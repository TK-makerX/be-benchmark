"""Candidate guide/window generation for the BE prediction system."""

from __future__ import annotations

import csv
import json
import math
import os
import sys
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .sequence import infer_editor_family, normalize_sequence


RELEASE_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_CORE = Path(os.environ.get("BE_BENCHMARK_CORE", RELEASE_ROOT / "workflows")).expanduser()
ADAPTERS_DIR = Path(os.environ.get("BE_ADAPTERS_DIR", BENCHMARK_CORE / "_model_adapters")).expanduser()
FORECAST_PYTHON = Path(os.environ.get("BE_FORECAST_PYTHON", sys.executable)).expanduser()
TF210_PYTHON = Path(os.environ.get("BE_TF210_PYTHON", sys.executable)).expanduser()
FORECAST_ADAPTER = ADAPTERS_DIR / "run_forecast_be_pilot.py"
DEEPBASEEDITOR_ADAPTER = ADAPTERS_DIR / "run_deepbaseeditor_abe_efficiency_pilot.py"


def revcomp(seq: str) -> str:
    table = str.maketrans("ACGTNacgtn", "TGCANtgcan")
    return seq.translate(table)[::-1].upper()


def is_ngg(pam: str) -> bool:
    pam = (pam or "").upper()
    return len(pam) == 3 and pam[1:] == "GG" and set(pam) <= set("ACGT")


def _editable_base_for_editor(editor_family: str) -> str:
    if editor_family == "ABE":
        return "A"
    if editor_family == "CBE":
        return "C"
    return "AC"


@dataclass
class Candidate:
    candidate_id: str
    strand: str
    guide_20nt: str
    pam_3nt: str
    target_23nt: str
    window_30nt: str
    source_window_start_0based: int
    source_window_end_0based_exclusive: int
    protospacer_start_0based_plus: int
    pam_start_0based_plus: int
    editable_base: str
    editable_positions_1based: List[int]
    editable_count: int
    mean_editable_position: Optional[float]
    center_distance_from_pos6: Optional[float]
    heuristic_rank_score: float
    warnings: List[str]

    def to_dict(self) -> Dict[str, Any]:
        row = asdict(self)
        row["editable_positions_1based"] = ",".join(str(x) for x in self.editable_positions_1based)
        row["warnings"] = "; ".join(self.warnings)
        return row


def _candidate_score(editable_positions: List[int]) -> tuple[float, Optional[float], Optional[float]]:
    if not editable_positions:
        return -100.0, None, None
    mean_pos = sum(editable_positions) / len(editable_positions)
    center_distance = abs(mean_pos - 6.0)
    score = len(editable_positions) * 10.0 - center_distance
    return score, mean_pos, center_distance


def _candidate_from_oriented_window(
    *,
    candidate_index: int,
    strand: str,
    guide: str,
    pam: str,
    window: str,
    source_start: int,
    source_end: int,
    protospacer_start_plus: int,
    pam_start_plus: int,
    editable_base: str,
) -> Candidate:
    edit_positions = [
        pos
        for pos, base in enumerate(guide, start=1)
        if 4 <= pos <= 8 and base in editable_base
    ]
    score, mean_pos, center_distance = _candidate_score(edit_positions)
    warnings: List[str] = []
    if not edit_positions:
        warnings.append(f"No editable {editable_base} base in positions 4-8")
    if not is_ngg(pam):
        warnings.append("PAM is not NGG-like")
    return Candidate(
        candidate_id=f"cand_{candidate_index:05d}",
        strand=strand,
        guide_20nt=guide,
        pam_3nt=pam,
        target_23nt=guide + pam,
        window_30nt=window,
        source_window_start_0based=source_start,
        source_window_end_0based_exclusive=source_end,
        protospacer_start_0based_plus=protospacer_start_plus,
        pam_start_0based_plus=pam_start_plus,
        editable_base=editable_base,
        editable_positions_1based=edit_positions,
        editable_count=len(edit_positions),
        mean_editable_position=mean_pos,
        center_distance_from_pos6=center_distance,
        heuristic_rank_score=score,
        warnings=warnings,
    )


def generate_candidates(
    sequence: str,
    editor: str = "ABE",
    strands: str = "both",
    max_candidates: Optional[int] = None,
) -> Dict[str, Any]:
    seq = normalize_sequence(sequence)
    editor_family = infer_editor_family(editor)
    editable_base = _editable_base_for_editor(editor_family)
    warnings: List[str] = []
    candidates: List[Candidate] = []
    if not seq:
        warnings.append("No sequence was provided.")
    if set(seq) - set("ACGT"):
        warnings.append("Candidate generation requires concrete A/C/G/T sequence; ambiguous bases were excluded from model-ready candidates.")
    if len(seq) < 30:
        warnings.append("Sequence is shorter than 30 nt; 30nt model-window candidates cannot be generated.")

    index = 1
    if len(seq) >= 30 and not (set(seq) - set("ACGT")):
        if strands in {"both", "plus", "+"}:
            for protospacer_start in range(4, len(seq) - 26 + 1):
                pam_start = protospacer_start + 20
                pam = seq[pam_start : pam_start + 3]
                if not is_ngg(pam):
                    continue
                window_start = protospacer_start - 4
                window_end = protospacer_start + 26
                guide = seq[protospacer_start : protospacer_start + 20]
                window = seq[window_start:window_end]
                candidates.append(
                    _candidate_from_oriented_window(
                        candidate_index=index,
                        strand="+",
                        guide=guide,
                        pam=pam,
                        window=window,
                        source_start=window_start,
                        source_end=window_end,
                        protospacer_start_plus=protospacer_start,
                        pam_start_plus=pam_start,
                        editable_base=editable_base,
                    )
                )
                index += 1
        if strands in {"both", "minus", "-"}:
            for pam_start_plus in range(3, len(seq) - 27 + 1):
                plus_pam = seq[pam_start_plus : pam_start_plus + 3]
                pam = revcomp(plus_pam)
                if not is_ngg(pam):
                    continue
                source_start = pam_start_plus - 3
                source_end = pam_start_plus + 27
                plus_segment = seq[source_start:source_end]
                window = revcomp(plus_segment)
                guide = window[4:24]
                candidates.append(
                    _candidate_from_oriented_window(
                        candidate_index=index,
                        strand="-",
                        guide=guide,
                        pam=pam,
                        window=window,
                        source_start=source_start,
                        source_end=source_end,
                        protospacer_start_plus=pam_start_plus + 3,
                        pam_start_plus=pam_start_plus,
                        editable_base=editable_base,
                    )
                )
                index += 1

    candidates.sort(
        key=lambda c: (
            c.editable_count > 0,
            c.heuristic_rank_score,
            -c.center_distance_from_pos6 if c.center_distance_from_pos6 is not None else -99,
        ),
        reverse=True,
    )
    total_candidates = len(candidates)
    if max_candidates is not None and max_candidates > 0:
        candidates = candidates[:max_candidates]
    for idx, candidate in enumerate(candidates, start=1):
        candidate.candidate_id = f"cand_{idx:05d}"
    return {
        "input": {
            "sequence_length": len(seq),
            "editor": editor,
            "editor_family": editor_family,
            "editable_base": editable_base,
            "strands": strands,
            "max_candidates": max_candidates,
        },
        "total_candidates_before_limit": total_candidates,
        "returned_candidates": len(candidates),
        "warnings": warnings,
        "candidates": [candidate.to_dict() for candidate in candidates],
    }


def _write_candidate_panel(path: Path, candidates: List[Dict[str, Any]], editor: str, editor_family: str, cas_variant: str = "") -> None:
    fields = [
        "benchmark_id",
        "sequence_context",
        "protospacer_20nt",
        "pam",
        "editor",
        "editor_class",
        "cas_variant",
        "cell_context",
        "reference_outcome",
        "intended_edit",
        "notes",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for candidate in candidates:
            writer.writerow(
                {
                    "benchmark_id": candidate["candidate_id"],
                    "sequence_context": candidate["window_30nt"],
                    "protospacer_20nt": candidate["guide_20nt"],
                    "pam": candidate["pam_3nt"],
                    "editor": editor,
                    "editor_class": editor_family,
                    "cas_variant": cas_variant or ("SpCas9" if candidate["pam_3nt"][1:] == "GG" else ""),
                    "cell_context": "",
                    "reference_outcome": "",
                    "intended_edit": "A>G" if editor_family == "ABE" else ("C>T" if editor_family == "CBE" else ""),
                    "notes": "candidate_generation_scoring_panel",
                }
            )


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _run_adapter(
    *,
    python_exe: Path,
    adapter: Path,
    panel: Path,
    out_dir: Path,
    timeout: int,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(python_exe), str(adapter), "--panel", str(panel), "--output-dir", str(out_dir)],
        cwd=str(ADAPTERS_DIR),
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )


def score_candidates(
    candidate_report: Dict[str, Any],
    editor: str,
    score_backends: Iterable[str],
    cas_variant: str = "",
) -> Dict[str, Any]:
    candidates = candidate_report.get("candidates", [])
    if not candidates:
        return candidate_report
    editor_family = candidate_report["input"]["editor_family"]
    backend_status: Dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="be_candidate_scoring_") as tmp:
        tmp_dir = Path(tmp)
        panel = tmp_dir / "candidate_panel.csv"
        _write_candidate_panel(panel, candidates, editor, editor_family, cas_variant=cas_variant)
        by_id = {row["candidate_id"]: row for row in candidates}
        for backend in score_backends:
            backend = backend.strip()
            if not backend:
                continue
            if backend == "forecast_be":
                proc = _run_adapter(
                    python_exe=FORECAST_PYTHON,
                    adapter=FORECAST_ADAPTER,
                    panel=panel,
                    out_dir=tmp_dir / "forecast_be",
                    timeout=180,
                )
                if proc.returncode != 0:
                    backend_status[backend] = {"status": "failed", "stderr": proc.stderr[-2000:]}
                    continue
                rows = _read_csv_rows(tmp_dir / "forecast_be" / "predictions_forecast_be_pilot.csv")
                scored = 0
                for row in rows:
                    if row.get("task_type") != "efficiency":
                        continue
                    candidate = by_id.get(row.get("benchmark_id", ""))
                    if not candidate:
                        continue
                    candidate["forecast_be_score"] = row.get("predicted_score", "")
                    candidate["forecast_be_efficiency"] = row.get("predicted_efficiency", "")
                    scored += 1
                backend_status[backend] = {"status": "completed", "scored_candidates": scored}
            elif backend == "deepbaseeditor":
                if editor_family != "ABE":
                    backend_status[backend] = {"status": "skipped", "reason": "DeepBaseEditor connector is ABE-only."}
                    continue
                proc = _run_adapter(
                    python_exe=TF210_PYTHON,
                    adapter=DEEPBASEEDITOR_ADAPTER,
                    panel=panel,
                    out_dir=tmp_dir / "deepbaseeditor",
                    timeout=300,
                )
                if proc.returncode != 0:
                    backend_status[backend] = {"status": "failed", "stderr": proc.stderr[-2000:]}
                    continue
                rows = _read_csv_rows(tmp_dir / "deepbaseeditor" / "predictions_deepbaseeditor_abe_efficiency_pilot.csv")
                scored = 0
                for row in rows:
                    candidate = by_id.get(row.get("benchmark_id", ""))
                    if not candidate:
                        continue
                    candidate["deepbaseeditor_score"] = row.get("predicted_score", "")
                    candidate["deepbaseeditor_efficiency"] = row.get("predicted_efficiency", "")
                    scored += 1
                backend_status[backend] = {"status": "completed", "scored_candidates": scored}
            else:
                backend_status[backend] = {"status": "not_implemented", "reason": "Candidate batch scoring supports forecast_be and deepbaseeditor in this MVP."}

    # Ranking: prefer model scores when available, otherwise heuristic score.
    def rank_key(row: Dict[str, Any]) -> tuple:
        model_score = None
        for key in ["deepbaseeditor_efficiency", "forecast_be_score"]:
            value = row.get(key)
            if value not in {None, ""}:
                try:
                    model_score = float(value)
                    break
                except (TypeError, ValueError):
                    pass
        return (
            model_score is not None,
            model_score if model_score is not None else -math.inf,
            float(row.get("heuristic_rank_score") or -math.inf),
        )

    candidates.sort(key=rank_key, reverse=True)
    for idx, row in enumerate(candidates, start=1):
        row["rank_after_scoring"] = idx
    candidate_report["scoring_backend_status"] = backend_status
    candidate_report["candidates"] = candidates
    return candidate_report


def write_candidate_outputs(report: Dict[str, Any], output_dir: Path) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "candidate_generation_result.json"
    csv_path = output_dir / "candidate_generation_candidates.csv"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    candidates = report.get("candidates", [])
    if candidates:
        fieldnames: List[str] = []
        for row in candidates:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(candidates)
    else:
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            handle.write("candidate_id\n")
    return {"json": str(json_path), "csv": str(csv_path)}

