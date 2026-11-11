#!/usr/bin/env python
"""
Audit local data sources for the unified base-editing benchmark matrix v5.

The goal is not to build the final matrix yet. This script inventories the
available local processed/supplementary sources, estimates task eligibility, and
flags whether a source can plausibly contribute to T1-T7.
"""

from __future__ import annotations

import gzip
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from runtime_paths import BEDICT_SUPPLEMENT, CORE_ROOT, CRISPRON_DEV_DATA, DATA_ROOT, REPORT_ROOT

import openpyxl
import pandas as pd


ROOT = CORE_ROOT
DATA = DATA_ROOT
MANUAL = DATA / "manual_downloads"
BEDICT2_XLSX = BEDICT_SUPPLEMENT
CRISPRON_DEV = CRISPRON_DEV_DATA
OUT_DIR = DATA / "unified_matrix_v5"
REPORT_DIR = REPORT_ROOT

DNA = set("ACGT")


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


def file_size_mb(path: Path) -> float:
    return path.stat().st_size / 1024 / 1024 if path.exists() else 0.0


def safe_rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except Exception:
        return str(path)


def count_text_rows(path: Path, sep: str = ",") -> Optional[int]:
    encodings = ["utf-8-sig", "utf-8", "latin1"]
    for enc in encodings:
        try:
            with path.open("r", encoding=enc, newline="") as handle:
                return max(0, sum(1 for _ in handle) - 1)
        except Exception:
            continue
    return None


def extract_30mer(seq: object, guide: object) -> str:
    seq = str(seq).upper()
    guide = str(guide).upper()
    idx = seq.find(guide)
    if idx < 4 or idx + 26 > len(seq):
        return ""
    out = seq[idx - 4 : idx + 26]
    if len(out) != 30 or set(out) - DNA:
        return ""
    return out


def fraction_columns(columns: Iterable[str]) -> List[str]:
    return [
        col
        for col in columns
        if str(col).startswith("Fraction edited ")
        and "logit" not in str(col).lower()
        and "norm" not in str(col).lower()
    ]


def task_flags(*tasks: str) -> Dict[str, bool]:
    return {f"candidate_{t}": t in tasks for t in [f"T{i}" for i in range(1, 8)]}


def base_record(
    source_family: str,
    source_dataset: str,
    source_file: Path,
    source_sheet: str = "",
    read_status: str = "ok",
    notes: str = "",
    source_row_count: Optional[int] = None,
    candidate_label_types: str = "",
    candidate_tasks: str = "",
    **task_kwargs: bool,
) -> Dict[str, object]:
    record = {
        "source_family": source_family,
        "source_dataset": source_dataset,
        "source_file": safe_rel(source_file),
        "source_sheet": source_sheet,
        "source_row_count": source_row_count,
        "source_size_mb": round(file_size_mb(source_file), 3),
        "candidate_label_types": candidate_label_types,
        "candidate_tasks": candidate_tasks,
        "read_status": read_status,
        "notes": notes,
    }
    for i in range(1, 8):
        record[f"candidate_T{i}"] = bool(task_kwargs.get(f"candidate_T{i}", False))
    return record


def audit_behive_efficiency() -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    folder = MANUAL / "be_hive_efficiency"
    if not folder.exists():
        return [
            base_record(
                "BE-Hive",
                "processed_efficiency",
                folder,
                read_status="missing",
                notes="folder missing",
            )
        ]
    for path in sorted(folder.glob("*.csv")):
        rec = base_record(
            "BE-Hive",
            "processed_efficiency",
            path,
            candidate_label_types="total_efficiency",
            candidate_tasks="T1;T2;T5;T6;T7_if_repeated_design",
            **task_flags("T1", "T2", "T5", "T6", "T7"),
        )
        try:
            df = pd.read_csv(
                path,
                usecols=lambda c: c
                in [
                    "gRNA (20nt)",
                    "Sequence context (56nt)",
                    "Sequence context (35 or 61 nt)",
                    "Name (unique)",
                ]
                or (
                    str(c).startswith("Fraction edited ")
                    and "logit" not in str(c).lower()
                    and "norm" not in str(c).lower()
                ),
            )
            frac_cols = fraction_columns(df.columns)
            if not frac_cols:
                raise ValueError("no fraction edited columns")
            seq_col = (
                "Sequence context (56nt)"
                if "Sequence context (56nt)" in df.columns
                else "Sequence context (35 or 61 nt)"
                if "Sequence context (35 or 61 nt)" in df.columns
                else ""
            )
            if not seq_col:
                raise ValueError("no sequence context column")
            measured = df[frac_cols].mean(axis=1, skipna=True)
            seqs = pd.Series(
                [
                    extract_30mer(seq, guide)
                    for seq, guide in zip(df[seq_col], df["gRNA (20nt)"])
                ]
            )
            valid = (
                seqs.str.len().eq(30)
                & measured.notna()
                & measured.between(0, 1)
                & df["gRNA (20nt)"].astype(str).str.len().eq(20)
            )
            if "AtoG" in path.name:
                valid &= seqs.str[6:14].str.contains("A", regex=False)
                edit = "A>G"
            elif "CtoT" in path.name:
                valid &= seqs.str[6:14].str.contains("C", regex=False)
                edit = "C>T"
            elif "CtoGA" in path.name:
                valid &= seqs.str[6:14].str.contains("C", regex=False)
                edit = "C>G/A"
            else:
                edit = "characterization"
            rec.update(
                {
                    "source_row_count": len(df),
                    "estimated_T1_valid_rows": int(valid.sum()),
                    "estimated_unique_design_label_rows": int(
                        pd.DataFrame(
                            {
                                "guide": df["gRNA (20nt)"].astype(str).str.upper(),
                                "seq30": seqs,
                                "label": measured,
                            }
                        )[valid]
                        .drop_duplicates()
                        .shape[0]
                    ),
                    "edit_inferred_from_filename": edit,
                    "replicate_fraction_columns": len(frac_cols),
                    "read_status": "ok",
                    "notes": "Full source can be converted to T1 by mean fraction-edited label; source-associated for BE-Hive-family models.",
                }
            )
        except Exception as exc:
            rec.update({"read_status": "error", "notes": repr(exc)})
        out.append(rec)
    return out


def audit_bedict2() -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    path = BEDICT2_XLSX
    if not path.exists():
        return [
            base_record(
                "BEDICT2",
                "supplement_absolute_efficiency",
                path,
                read_status="missing",
                notes="BEDICT2 supplement missing",
            )
        ]

    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    for ws in wb.worksheets:
        total_rows = max(0, (ws.max_row or 0) - 1)
        rec = base_record(
            "BEDICT2",
            "supplement_absolute_efficiency",
            path,
            source_sheet=ws.title,
            source_row_count=total_rows,
            candidate_label_types="total_efficiency;position_rate_possible",
            candidate_tasks="T1;T2;T4;T5;T6;T7_if_repeated_design",
            **task_flags("T1", "T2", "T4", "T5", "T6", "T7"),
        )
        it = ws.iter_rows(values_only=True)
        try:
            header = next(it)
        except StopIteration:
            rec.update({"read_status": "empty", "notes": "empty sheet"})
            out.append(rec)
            continue
        columns = {str(v): i for i, v in enumerate(header) if v is not None}
        required = ["refSeq", "Target-Sequence", "absolute_editing_efficiency"]
        if not all(col in columns for col in required):
            rec.update(
                {
                    "read_status": "skipped",
                    "estimated_T1_valid_rows": 0,
                    "estimated_NGG_valid_rows": 0,
                    "notes": "metadata or overview sheet; required BEDICT2 columns absent",
                }
            )
            out.append(rec)
            continue

        ip = columns["refSeq"]
        itarget = columns["Target-Sequence"]
        ieff = columns["absolute_editing_efficiency"]
        valid = 0
        ngg = 0
        for row in it:
            proto = row[ip] if ip < len(row) else None
            target = row[itarget] if itarget < len(row) else None
            eff = row[ieff] if ieff < len(row) else None
            try:
                e = float(eff) / 100.0
            except Exception:
                continue
            if e < 0 or e > 1:
                continue
            proto_s = str(proto).upper() if proto is not None else ""
            if len(proto_s) != 20:
                continue
            seq30 = extract_30mer(target, proto_s)
            if not seq30:
                continue
            if "A" not in seq30[6:14]:
                continue
            valid += 1
            if seq30[25:27] == "GG":
                ngg += 1
        rec.update(
            {
                "estimated_T1_valid_rows": valid,
                "estimated_NGG_valid_rows": ngg,
                "read_status": "ok",
                "notes": "Full sheet can be converted to ABE T1; SpCas9/NGG subset is much smaller than SpRY/SpG/PAM-relaxed subset.",
            }
        )
        out.append(rec)
    wb.close()
    return out


def audit_crispron_dev() -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    if not CRISPRON_DEV.exists():
        return [
            base_record(
                "CRISPRon-BE",
                "model_development",
                CRISPRON_DEV,
                read_status="missing",
                notes="folder missing",
            )
        ]
    for path in sorted(CRISPRON_DEV.glob("*.tsv.gz")):
        rec = base_record(
            "CRISPRon-BE",
            "model_development",
            path,
            candidate_label_types="total_efficiency;outcome_frequency",
            candidate_tasks="T1;T2;T3;T5;T6",
            **task_flags("T1", "T2", "T3", "T5", "T6"),
        )
        try:
            df = pd.read_csv(path, sep="\t", compression="gzip")
            rec.update(
                {
                    "source_row_count": len(df),
                    "estimated_unique_refs": int(df["refs"].nunique()) if "refs" in df else None,
                    "estimated_T1_valid_rows": int(df["refs"].nunique()) if "refs" in df else None,
                    "estimated_outcome_rows": int(len(df)),
                    "source_breakdown": json.dumps(
                        df["source"].value_counts().to_dict(), ensure_ascii=False
                    )
                    if "source" in df
                    else "",
                    "read_status": "ok",
                    "notes": "Outcome-level rows; T1 requires collapsing outcome rows to unique refs/editingeff. Train rows are source-associated for CRISPRon-BE.",
                }
            )
        except Exception as exc:
            rec.update({"read_status": "error", "notes": repr(exc)})
        out.append(rec)
    return out


def audit_simple_groups() -> List[Dict[str, object]]:
    groups = {
        "FORECasT-BE": (MANUAL / "forecast_be", "processed_outcome_matrix", "outcome_frequency;position_rate_possible", "T3;T4;T5;T6"),
        "CGBE": (MANUAL / "cgbe_smart_processed", "processed_efficiency_outcome", "total_efficiency;outcome_frequency_possible", "T1;T2;T3;T5;T6"),
        "BE-DICT": (MANUAL / "be_dict_supplement", "supplementary", "total_efficiency;position_rate;outcome_frequency", "T1;T2;T3;T4;T5;T6;T7"),
        "BEguider": (MANUAL / "beguider", "supplementary_and_github", "measured_or_prediction_mixed", "T1_possible;T2_possible;T5;T6"),
        "BE_Endo": (MANUAL / "be_endo_supplement", "supplementary", "total_efficiency_possible", "T1_possible;T2_possible;T5;T6"),
    }
    out: List[Dict[str, object]] = []
    for family, (folder, dataset, labels, tasks) in groups.items():
        if not folder.exists():
            out.append(
                base_record(
                    family,
                    dataset,
                    folder,
                    read_status="missing",
                    candidate_label_types=labels,
                    candidate_tasks=tasks,
                    notes="folder missing",
                )
            )
            continue
        for path in sorted(folder.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in {".csv", ".tsv", ".xlsx", ".xls"}:
                continue
            flags = task_flags(
                *[
                    t
                    for t in [f"T{i}" for i in range(1, 8)]
                    if t in tasks.replace("_possible", "")
                ]
            )
            rec = base_record(
                family,
                dataset,
                path,
                candidate_label_types=labels,
                candidate_tasks=tasks,
                **flags,
            )
            try:
                if path.suffix.lower() in {".csv", ".tsv"}:
                    rec["source_row_count"] = count_text_rows(
                        path, sep="\t" if path.suffix.lower() == ".tsv" else ","
                    )
                    rec["read_status"] = "ok"
                    rec["notes"] = "Inventory only; schema-specific conversion required."
                elif path.suffix.lower() == ".xlsx":
                    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
                    total = 0
                    sheet_summaries = []
                    for ws in wb.worksheets:
                        rows = max(0, (ws.max_row or 0) - 1)
                        total += rows
                        sheet_summaries.append(f"{ws.title}:{rows}")
                    wb.close()
                    rec["source_row_count"] = total
                    rec["source_sheet"] = ";".join(sheet_summaries[:12])
                    rec["read_status"] = "ok"
                    rec["notes"] = "Workbook inventory only; measured-vs-predicted label separation required."
                else:
                    rec["read_status"] = "skipped"
                    rec["notes"] = "unsupported file extension for audit"
            except Exception as exc:
                rec["read_status"] = "error"
                rec["notes"] = repr(exc)
            out.append(rec)
    return out


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    aggregations = {
        "source_row_count": "sum",
        "source_size_mb": "sum",
        "estimated_T1_valid_rows": "sum",
        "estimated_NGG_valid_rows": "sum",
        "estimated_unique_refs": "sum",
        "estimated_outcome_rows": "sum",
    }
    for col in list(aggregations):
        if col not in df.columns:
            df[col] = 0
    unique_file_size = (
        df.drop_duplicates(["source_family", "source_file"])
        .groupby("source_family", dropna=False)["source_size_mb"]
        .sum()
        .to_dict()
    )
    summary = (
        df.groupby("source_family", dropna=False)
        .agg(
            files=("source_file", "count"),
            rows=("source_row_count", "sum"),
            est_t1_rows=("estimated_T1_valid_rows", "sum"),
            est_ngg_rows=("estimated_NGG_valid_rows", "sum"),
            est_unique_refs=("estimated_unique_refs", "sum"),
            est_outcome_rows=("estimated_outcome_rows", "sum"),
        )
        .reset_index()
    )
    summary["size_mb"] = summary["source_family"].map(unique_file_size).fillna(0.0).round(3)
    summary = summary[
        [
            "source_family",
            "files",
            "rows",
            "size_mb",
            "est_t1_rows",
            "est_ngg_rows",
            "est_unique_refs",
            "est_outcome_rows",
        ]
    ]
    return summary


def write_report(inventory: pd.DataFrame, summary: pd.DataFrame) -> None:
    report = REPORT_DIR / "UNIFIED_MATRIX_CANDIDATE_AUDIT_V5.md"

    def md_table(table: pd.DataFrame) -> str:
        if table.empty:
            return "_No rows._"
        t = table.copy().fillna("")
        lines = [
            "| " + " | ".join(map(str, t.columns)) + " |",
            "| " + " | ".join(["---"] * len(t.columns)) + " |",
        ]
        for _, row in t.iterrows():
            lines.append(
                "| "
                + " | ".join(str(row[c]).replace("|", "\\|").replace("\n", " ") for c in t.columns)
                + " |"
            )
        return "\n".join(lines)

    top = inventory.sort_values(
        by=["estimated_T1_valid_rows", "source_row_count"],
        ascending=False,
        na_position="last",
    ).head(40)
    columns = [
        "source_family",
        "source_dataset",
        "source_file",
        "source_sheet",
        "source_row_count",
        "estimated_T1_valid_rows",
        "estimated_NGG_valid_rows",
        "estimated_unique_refs",
        "estimated_outcome_rows",
        "candidate_tasks",
        "read_status",
    ]
    columns = [c for c in columns if c in top.columns]

    text = f"""# Unified matrix candidate audit v5

Generated: {datetime.now().isoformat(timespec="seconds")}

## Purpose

This audit checks how much local data can plausibly feed the unified benchmark
matrix before any full-scale model run. It is source inventory and eligibility
assessment, not final performance evaluation.

## Source-level summary

{md_table(summary)}

## Largest / most useful candidate sources

{md_table(top[columns])}

## Interpretation

- The previous merged T1 result was pilot-scale and should not be used as the final
  manuscript-scale benchmark.
- BE-Hive efficiency and BEDICT2 alone contain enough local rows to exceed BEEP's
  34,040 BE-gRNA-target combination scale after careful conversion and QC.
- The full matrix must still use source-balanced evaluation, otherwise BE-Hive-like
  sources will dominate the numerical metrics.
- FORECasT-BE, BE-DICT, BEguider and CGBE are inventoried here but require schema-
  specific conversion before they can enter task views.

## Output file

- Inventory CSV: `{OUT_DIR / 'source_candidate_inventory_v5.csv'}`
"""
    report.write_text(text, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    records: List[Dict[str, object]] = []
    records.extend(audit_behive_efficiency())
    records.extend(audit_bedict2())
    records.extend(audit_crispron_dev())
    records.extend(audit_simple_groups())

    inventory = pd.DataFrame(records)
    for col in [
        "estimated_T1_valid_rows",
        "estimated_NGG_valid_rows",
        "estimated_unique_refs",
        "estimated_outcome_rows",
    ]:
        if col not in inventory.columns:
            inventory[col] = 0
        inventory[col] = pd.to_numeric(inventory[col], errors="coerce").fillna(0).astype(int)
    inventory["source_row_count"] = pd.to_numeric(
        inventory["source_row_count"], errors="coerce"
    ).fillna(0).astype(int)

    inventory.to_csv(OUT_DIR / "source_candidate_inventory_v5.csv", index=False)
    summary = summarize(inventory)
    summary.to_csv(OUT_DIR / "source_candidate_summary_v5.csv", index=False)
    write_report(inventory, summary)

    print(f"Inventory rows: {len(inventory)}")
    print(f"Output: {OUT_DIR}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
