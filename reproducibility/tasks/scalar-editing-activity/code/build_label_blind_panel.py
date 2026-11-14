from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


FORBIDDEN_SELECTION_TOKENS = ("measured", "efficiency", "label", "outcome_frequency", "edited_read")
SELECTION_COLUMNS = [
    "observation_id", "design_id", "source_family", "source_dataset", "source_file", "source_sheet",
    "source_row_key", "sequence_30nt", "protospacer_20nt", "pam", "pam_type", "target_base",
    "intended_edit", "editor", "editor_class", "cas_variant", "cell_context", "delivery",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-panel", type=Path, required=True)
    parser.add_argument("--crispron-train", type=Path, required=True)
    parser.add_argument("--crispron-test", type=Path, required=True)
    parser.add_argument("--old-evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--n", type=int, default=10000)
    return parser.parse_args()


def clean_dna(value):
    text = "" if pd.isna(value) else str(value).upper()
    return "".join(char for char in text if char in "ACGT")


def extract_30mer(context, guide):
    context, guide = clean_dna(context), clean_dna(guide)
    idx = context.find(guide)
    if idx < 4 or idx + 26 > len(context):
        return ""
    result = context[idx - 4:idx + 26]
    return result if len(result) == 30 else ""


def author_sets(path: Path):
    frame = pd.read_csv(path, sep="\t", usecols=["refs", "surro_target"])
    seq20 = frame["refs"].map(clean_dna)
    seq30 = pd.Series([extract_30mer(c, g) for c, g in zip(frame["surro_target"], frame["refs"])])
    return set(seq20[seq20.ne("")]), set(seq30[seq30.ne("")])


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def distribution(frame: pd.DataFrame, column: str, stage: str):
    counts = frame[column].fillna("NA").astype(str).value_counts(dropna=False)
    return pd.DataFrame({"stage": stage, "metadata_field": column, "level": counts.index, "observations": counts.values})


def main():
    args = parse_args()
    out = args.output
    for name in ["data", "metrics", "reports", "code", "logs", "manifest", "intermediate", "predictions/raw", "predictions/harmonized", "figures", "source_data", "supplementary", "superseded"]:
        (out / name).mkdir(parents=True, exist_ok=True)

    forbidden = [col for col in SELECTION_COLUMNS if any(token in col.lower() for token in FORBIDDEN_SELECTION_TOKENS)]
    if forbidden:
        raise RuntimeError(f"Selection column contract contains forbidden label-derived columns: {forbidden}")

    # Phase 1: selection-only read. No measured value or label-derived column is loaded.
    header = set(pd.read_csv(args.full_panel, nrows=0).columns)
    selection_cols = [col for col in SELECTION_COLUMNS if col in header]
    metadata = pd.read_csv(args.full_panel, usecols=selection_cols, low_memory=False)
    population = metadata[metadata["source_family"].eq("BE-Hive")].copy()
    population["sequence_30nt"] = population["sequence_30nt"].map(clean_dna)
    population["protospacer_20nt"] = population["protospacer_20nt"].map(clean_dna)

    train20, train30 = author_sets(args.crispron_train)
    test20, test30 = author_sets(args.crispron_test)
    population["known_crispron_train_overlap_20nt"] = population["protospacer_20nt"].isin(train20)
    population["known_crispron_test_overlap_20nt"] = population["protospacer_20nt"].isin(test20)
    population["known_crispron_train_overlap_30nt"] = population["sequence_30nt"].isin(train30)
    population["known_crispron_test_overlap_30nt"] = population["sequence_30nt"].isin(test30)
    flag_cols = [col for col in population.columns if col.startswith("known_crispron_")]
    population["known_crispron_development_overlap_any"] = population[flag_cols].any(axis=1)
    eligible = population[~population["known_crispron_development_overlap_any"]].copy()
    if len(eligible) < args.n:
        raise RuntimeError(f"Only {len(eligible)} eligible rows for requested n={args.n}")

    rng = np.random.default_rng(args.seed)
    chosen_positions = np.sort(rng.choice(len(eligible), size=args.n, replace=False))
    selected_ids = set(eligible.iloc[chosen_positions]["observation_id"].astype(str))
    population["selected_label_blind_random_panel"] = population["observation_id"].astype(str).isin(selected_ids)
    population["selection_seed"] = args.seed
    population["selection_probability"] = args.n / len(eligible)

    old = pd.read_csv(args.old_evaluation, usecols=["observation_id", "design_id", "source_family"]).drop_duplicates("observation_id")
    old_ids = set(old.loc[old["source_family"].eq("BE-Hive"), "observation_id"].astype(str))
    population["in_old_outcome_aware_10k"] = population["observation_id"].astype(str).isin(old_ids)
    population["selected_and_old_overlap"] = population["selected_label_blind_random_panel"] & population["in_old_outcome_aware_10k"]

    population.to_csv(out / "data" / "T1_V8_FULL_BEHIVE_SELECTION_AUDIT.csv.gz", index=False, compression="gzip")
    selected_metadata = population[population["selected_label_blind_random_panel"]].copy()
    selected_metadata.to_csv(out / "data" / "T1_V8_LABEL_BLIND_SELECTED_METADATA.csv.gz", index=False, compression="gzip")

    # Phase 2: labels are loaded only after observation IDs have been frozen.
    full = pd.read_csv(args.full_panel, low_memory=False)
    selected = full[full["observation_id"].astype(str).isin(selected_ids)].copy()
    selected = selected.set_index("observation_id").loc[sorted(selected_ids)].reset_index()
    if len(selected) != args.n or selected["observation_id"].nunique() != args.n:
        raise RuntimeError("Selected panel did not resolve to exactly one row per frozen observation ID")
    selected.to_csv(out / "data" / "T1_V8_LABEL_BLIND_ANALYSIS_PANEL.csv.gz", index=False, compression="gzip")
    selected.to_csv(out / "intermediate" / "T1_V8_LABEL_BLIND_ANALYSIS_PANEL.csv", index=False)

    adapter = pd.DataFrame({
        "benchmark_id": selected["observation_id"],
        "dataset_id": selected["source_dataset"],
        "source_type": selected["source_family"],
        "editor": selected["editor"],
        "editor_class": selected["editor_class"],
        "cell_context": selected["cell_context"],
        "delivery": selected["delivery"],
        "cas_variant": selected["cas_variant"],
        "sequence_context": selected["sequence_30nt"],
        "protospacer_20nt": selected["protospacer_20nt"],
        "pam": selected["pam"],
        "target_base": selected["target_base"],
        "target_position": selected.get("target_position", ""),
        "intended_edit": selected["intended_edit"],
        "reference_outcome": "",
        "measured_efficiency": selected["measured_total_efficiency"],
        "measured_outcome_frequency": "",
        "split": "label_blind_known_crispron_clean_random",
        "notes": "Label-blind random panel; known CRISPRon-BE development-sequence matches removed before random selection.",
    })
    adapter.to_csv(out / "data" / "T1_V8_COMMON_ADAPTER_PANEL.csv", index=False)

    summary = pd.DataFrame([{
        "full_BE-Hive_total_efficiency_observations": len(population),
        "full_BE-Hive_unique_designs": population["design_id"].nunique(),
        "known_CRISPRon_overlap_observations": int(population["known_crispron_development_overlap_any"].sum()),
        "known_CRISPRon_overlap_designs": population.loc[population["known_crispron_development_overlap_any"], "design_id"].nunique(),
        "eligible_after_known_sequence_exclusion_observations": len(eligible),
        "eligible_after_known_sequence_exclusion_designs": eligible["design_id"].nunique(),
        "selected_observations": len(selected_metadata),
        "selected_unique_designs": selected_metadata["design_id"].nunique(),
        "selected_overlap_with_old_outcome_aware_10k_observations": len(selected_ids & old_ids),
        "selected_overlap_with_old_outcome_aware_10k_designs": selected_metadata.loc[selected_metadata["in_old_outcome_aware_10k"], "design_id"].nunique(),
        "seed": args.seed,
        "selection_method": "uniform random without replacement from known-CRISPRon-clean eligible observation IDs; no measured value loaded",
    }])
    summary.to_csv(out / "metrics" / "LABEL_BLIND_PANEL_CONSTRUCTION_SUMMARY.csv", index=False)

    metadata_distributions = pd.concat([
        distribution(population, field, "before_known_sequence_exclusion") for field in ["editor", "cell_context", "source_file"]
    ] + [
        distribution(eligible, field, "eligible_after_known_sequence_exclusion") for field in ["editor", "cell_context", "source_file"]
    ] + [
        distribution(selected_metadata, field, "selected_label_blind_10k") for field in ["editor", "cell_context", "source_file"]
    ], ignore_index=True)
    metadata_distributions.to_csv(out / "metrics" / "LABEL_BLIND_PANEL_METADATA_DISTRIBUTIONS.csv", index=False)

    contract = {
        "selection_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "n": args.n,
        "selection_columns_loaded": selection_cols,
        "forbidden_tokens": list(FORBIDDEN_SELECTION_TOKENS),
        "measured_or_label_columns_loaded_during_selection": [],
        "labels_loaded_after_ids_frozen": True,
        "selection_input_sha256": sha256(args.full_panel),
        "train_sha256": sha256(args.crispron_train),
        "test_sha256": sha256(args.crispron_test),
    }
    (out / "manifest" / "LABEL_BLIND_SELECTION_CONTRACT.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
    print(json.dumps(summary.iloc[0].to_dict(), indent=2, default=str))


if __name__ == "__main__":
    main()
