from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

from runtime_paths import model_path

import pandas as pd

from t2_common import clean_float, dna, events_from_sequences, normalize_outcome_scores, outcome_rows_to_position_rows, write_csv


MODULES = {
    "ABE": {
        "subdir": "ABE_Proportion",
        "script": "TEST_ABE_Proportion.py",
        "sample": "ABE_Proportion_sample.txt",
        "input_len": 26,
        "slice_start": 1,
        "protospacer_start": 3,
        "window_start": 5,
        "ref_base": "A",
    },
    "CBE": {
        "subdir": "CBE_Proportion",
        "script": "TEST_CBE_Proportion.py",
        "sample": "CBE_Proportion_sample.txt",
        "input_len": 25,
        "slice_start": 2,
        "protospacer_start": 2,
        "window_start": 4,
        "ref_base": "C",
    },
}


def write_input(rows: list[dict], path: Path, seq_key: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("Target number\ttarget sequence\n")
        for idx, row in enumerate(rows, start=1):
            handle.write(f"{idx}\t{row[seq_key]}\n")


def patch_and_run_module(repo: Path, script_name: str, sample_name: str, input_path: Path) -> Path:
    import tensorflow.compat.v1 as tf_v1  # noqa: PLC0415

    tf_v1.disable_v2_behavior()
    sys.modules["tensorflow"] = tf_v1

    source_path = repo / script_name
    source = source_path.read_text(encoding="utf-8")
    source = source.replace(f"parameters           = {{'0': '{sample_name}'}}", "parameters           = {'0': '" + input_path.as_posix() + "'}")
    source = source.replace("path                 = './'", "path                 = ''")

    from lib2to3.refactor import RefactoringTool, get_fixers_from_package  # noqa: PLC0415

    output_xlsx = repo / "outputs" / "TEST_OUTPUT_fortest.xlsx"
    if output_xlsx.exists():
        output_xlsx.unlink()

    old_cwd = Path.cwd()
    try:
        os.chdir(repo)
        tool = RefactoringTool(get_fixers_from_package("lib2to3.fixes"))
        converted = str(tool.refactor_string(source, str(source_path)))
        globals_dict = {"__name__": "__main__", "__file__": str(source_path)}
        exec(compile(converted, str(source_path), "exec"), globals_dict)
    finally:
        os.chdir(old_cwd)

    if not output_xlsx.exists():
        raise RuntimeError(f"DeepBaseEditor proportion output was not created: {output_xlsx}")
    return output_xlsx


def deepbaseeditor_sequence(seq30: str, editor_class: str) -> str:
    meta = MODULES[editor_class]
    start = meta["slice_start"]
    return seq30[start : start + meta["input_len"]]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=model_path("BE_DEEPBASEEDITOR_REPO", "DeepBaseEditor-Paired-Library", "repo", "DeepBaseEditor"))
    parser.add_argument("--run-id", default="T3_exploratory_v0_deepbaseeditor")
    args = parser.parse_args()

    panel = pd.read_csv(args.panel, dtype=str, low_memory=False).fillna("")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_raw_rows: list[dict] = []
    all_outcome_rows: list[dict] = []
    all_skips: list[dict] = []

    for editor_class, meta in MODULES.items():
        module_repo = args.repo_root / meta["subdir"]
        selected: list[dict] = []
        seen_model_sequences: set[str] = set()
        for _, row in panel[panel["editor_class"].eq(editor_class)].iterrows():
            target_id = row["target_profile_id"]
            seq30 = row.get("sequence_context_30nt", "").upper()
            if len(seq30) != 30 or not dna(seq30):
                all_skips.append({"target_profile_id": target_id, "model_id": "deepbaseeditor", "reason": "DeepBaseEditor T3 requires concrete 30nt source target"})
                continue
            model_seq = deepbaseeditor_sequence(seq30, editor_class)
            if len(model_seq) != meta["input_len"] or not dna(model_seq):
                all_skips.append({"target_profile_id": target_id, "model_id": "deepbaseeditor", "reason": f"could not derive {meta['input_len']}nt DeepBaseEditor {editor_class} proportion input"})
                continue
            edit_window = model_seq[meta["window_start"] : meta["window_start"] + 8]
            if meta["ref_base"] not in edit_window:
                all_skips.append({"target_profile_id": target_id, "model_id": "deepbaseeditor", "reason": f"DeepBaseEditor {editor_class} fixed window has no substrate base"})
                continue
            if model_seq in seen_model_sequences:
                all_skips.append({"target_profile_id": target_id, "model_id": "deepbaseeditor", "reason": "duplicate DeepBaseEditor model input sequence skipped for exploratory run"})
                continue
            record = row.to_dict()
            record["_deepbaseeditor_seq"] = model_seq
            selected.append(record)
            seen_model_sequences.add(model_seq)

        if not selected:
            continue

        input_path = args.output_dir / f"raw_t3_deepbaseeditor_{editor_class}_input.txt"
        write_input(selected, input_path, "_deepbaseeditor_seq")
        output_xlsx = patch_and_run_module(module_repo, meta["script"], meta["sample"], input_path)
        copied_xlsx = args.output_dir / f"raw_t3_deepbaseeditor_{editor_class}_proportion.xlsx"
        shutil.copy2(output_xlsx, copied_xlsx)

        sheet = pd.read_excel(copied_xlsx, sheet_name=0, header=None).fillna("")
        selected_by_seq = {row["_deepbaseeditor_seq"]: row for row in selected}
        for _, pred in sheet.iterrows():
            reference = str(pred.iloc[0]).upper()
            outcome = str(pred.iloc[1]).upper()
            prop = clean_float(pred.iloc[3] if len(pred) > 3 else "")
            source_row = selected_by_seq.get(reference)
            if source_row is None:
                all_skips.append({"target_profile_id": "", "model_id": "deepbaseeditor", "reason": f"unmapped DeepBaseEditor output reference {reference}"})
                continue
            events = events_from_sequences(reference, outcome, meta["protospacer_start"], -3, 20)
            if events == "no_tracked_substrate_change":
                continue
            try:
                if float(prop) <= 0:
                    continue
            except Exception:
                continue
            target_id = source_row["target_profile_id"]
            all_raw_rows.append(
                {
                    "target_profile_id": target_id,
                    "editor_class": editor_class,
                    "model_reference_sequence": reference,
                    "model_outcome_sequence": outcome,
                    "predicted_outcome_proportion": prop,
                    "raw_output_path": str(copied_xlsx),
                }
            )
            all_outcome_rows.append(
                {
                    "prediction_id": f"deepbaseeditor_t3_outcome_{len(all_outcome_rows) + 1:08d}",
                    "target_profile_id": target_id,
                    "model_id": "deepbaseeditor",
                    "run_id": args.run_id,
                    "model_editor": f"{editor_class}_Proportion",
                    "model_cell_context": "",
                    "predicted_outcome_events": events,
                    "predicted_outcome_score": prop,
                    "predicted_outcome_frequency": prop,
                    "prediction_scale": "conditional_outcome_frequency_given_edited_reads",
                    "source_relation_to_model": source_row.get("source_relation_to_model_for_t3", "proxy_external_candidate"),
                    "notes": f"DeepBaseEditor {editor_class} proportion exploratory T3 profile; fixed 8bp module window marginalized to editable-position rates.",
                }
            )

    all_outcome_rows = normalize_outcome_scores(all_outcome_rows, score_key="predicted_outcome_score", output_key="predicted_outcome_frequency")
    position_rows = outcome_rows_to_position_rows(all_outcome_rows, id_prefix="deepbaseeditor_t3_position")
    for row in position_rows:
        row["notes"] = "DeepBaseEditor exploratory T3 position profile derived from ABE/CBE proportion module outputs."

    pd.DataFrame(all_raw_rows).to_csv(args.output_dir / "raw_t3_deepbaseeditor_proportion.csv", index=False)
    write_csv(args.output_dir / "predictions_t3_deepbaseeditor_outcomes.csv", all_outcome_rows)
    write_csv(args.output_dir / "predictions_t3_deepbaseeditor_positions.csv", position_rows)
    write_csv(args.output_dir / "skipped_t3_deepbaseeditor.csv", all_skips)
    print(
        json.dumps(
            {
                "model_id": "deepbaseeditor",
                "targets": panel["target_profile_id"].nunique(),
                "raw_outcome_rows": len(all_raw_rows),
                "outcome_predictions": len(all_outcome_rows),
                "position_predictions": len(position_rows),
                "skipped": len(all_skips),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
