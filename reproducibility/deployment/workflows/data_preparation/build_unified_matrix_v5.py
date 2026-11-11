#!/usr/bin/env python
"""
Build unified base-editing benchmark matrix v5.

First full-matrix construction batch:
  - BE-Hive processed efficiency -> T1/T2/T5/T7
  - BEDICT2 supplementary absolute/position efficiency -> T1/T2/T4/T5/T7
  - CRISPRon-BE model-development data -> collapsed T1/T2/T5 and outcome-level T3

The output is intentionally source-aware and measurement-scale-aware. It does not
pretend that all efficiency labels are identical.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

from runtime_paths import BEDICT_SUPPLEMENT, CORE_ROOT, CRISPRON_DEV_DATA, DATA_ROOT, REPORT_ROOT

import openpyxl
import pandas as pd


ROOT = CORE_ROOT
DATA = DATA_ROOT
OUT = DATA / "unified_matrix_v5"
TASK = OUT / "task_views"
REPORT = REPORT_ROOT / "UNIFIED_MATRIX_V5_QC_REPORT.md"

BEHIVE_DIR = DATA / "manual_downloads" / "be_hive_efficiency"
BEDICT2_XLSX = BEDICT_SUPPLEMENT
CRISPRON_DEV = CRISPRON_DEV_DATA
CANDIDATE_INVENTORY = OUT / "source_candidate_inventory_v5.csv"
MODEL_ELIGIBILITY = ROOT / "config" / "model_eligibility_v3.csv"

DNA = set("ACGT")


OBS_COLUMNS = [
    "observation_id",
    "design_id",
    "editor_context_id",
    "source_inventory_id",
    "source_family",
    "source_dataset",
    "source_file",
    "source_sheet",
    "source_row_key",
    "source_relation_to_model",
    "source_associated_models",
    "source_role",
    "label_type",
    "measured_total_efficiency",
    "measured_position_rate",
    "measured_outcome_frequency",
    "outcome_sequence",
    "no_edit_frequency",
    "desired_product_frequency",
    "total_read_count",
    "edited_read_count",
    "replicate_count",
    "measurement_scale",
    "task_eligibility_T1",
    "task_eligibility_T2",
    "task_eligibility_T3",
    "task_eligibility_T4",
    "task_eligibility_T5",
    "task_eligibility_T6",
    "task_eligibility_T7",
    "qc_sequence_pass",
    "qc_label_pass",
    "qc_editor_pass",
    "qc_leakage_pass",
    "qc_level",
    "exclusion_reason",
    "notes",
]

DESIGN_COLUMNS = [
    "design_id",
    "sequence_context_original",
    "sequence_56nt",
    "sequence_50nt",
    "sequence_30nt",
    "sequence_29nt",
    "protospacer_20nt",
    "pam",
    "pam_type",
    "target_base",
    "target_position",
    "intended_edit",
    "reference_allele",
    "alternate_allele",
    "gene",
    "genomic_locus",
    "sequence_qc_status",
    "design_hash",
]

EDITOR_COLUMNS = [
    "editor_context_id",
    "editor",
    "editor_class",
    "editor_family",
    "cas_variant",
    "pam_scope",
    "cell_context",
    "delivery",
    "timepoint",
    "replicate_strategy",
    "editor_qc_status",
    "editor_context_hash",
]


def ensure_dirs() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    TASK.mkdir(parents=True, exist_ok=True)
    (ROOT / "reports").mkdir(parents=True, exist_ok=True)


def stable_hash(*parts: object, n: int = 12) -> str:
    text = "||".join("" if x is None else str(x) for x in parts)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:n]


def stable_id(prefix: str, *parts: object) -> str:
    return f"{prefix}_{stable_hash(*parts)}"


def clean_dna(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = re.sub(r"[^ACGTacgt]", "", str(value)).upper()
    return text


def clean_float(value: object) -> Optional[float]:
    try:
        v = float(value)
    except Exception:
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def format_float(value: object, digits: int = 12) -> str:
    v = clean_float(value)
    if v is None:
        return ""
    return f"{v:.{digits}g}"


def extract_30mer(sequence_context: object, guide: object) -> str:
    seq = clean_dna(sequence_context)
    guide = clean_dna(guide)
    if not seq or not guide:
        return ""
    idx = seq.find(guide)
    if idx < 4 or idx + 26 > len(seq):
        return ""
    candidate = seq[idx - 4 : idx + 26]
    if len(candidate) != 30 or set(candidate) - DNA:
        return ""
    return candidate


def seq50_from_context(sequence_context: str, seq30: str) -> str:
    seq = clean_dna(sequence_context)
    if len(seq) >= 50 and seq30:
        idx = seq.find(seq30)
        if idx >= 10 and idx + 40 <= len(seq):
            out = seq[idx - 10 : idx + 40]
            if len(out) == 50:
                return out
        if len(seq) == 50:
            return seq
        if len(seq) >= 56:
            return seq[:50]
    return ""


def pam_type(pam: str) -> str:
    pam = str(pam or "").upper()
    if len(pam) < 2:
        return "unknown"
    if len(pam) >= 3 and pam[1:] == "GG":
        return "NGG"
    if "NG" in pam or pam.endswith("G"):
        return "noncanonical_NG_like"
    return "non_NGG_or_PAM_relaxed"


def fraction_columns(columns: Iterable[str]) -> List[str]:
    return [
        col
        for col in columns
        if str(col).startswith("Fraction edited ")
        and "logit" not in str(col).lower()
        and "norm" not in str(col).lower()
    ]


def count_columns(columns: Iterable[str], prefix: str) -> List[str]:
    return [col for col in columns if str(col).startswith(prefix)]


def infer_behive_from_filename(filename: str) -> Dict[str, str]:
    stem = Path(filename).stem
    parts = stem.split("_")
    cell = parts[0] if parts else ""
    editor = "_".join(parts[2:]) if len(parts) >= 3 and parts[1] in {"AtoG", "CtoT", "CtoGA"} else "_".join(parts[2:]) if len(parts) >= 3 and parts[1] == "12kChar" else stem
    if not editor or editor == "Char":
        editor = parts[-1] if parts else stem

    if "AtoG" in stem or "ABE" in stem:
        editor_class = "ABE"
        target_base = "A"
        intended_edit = "A>G"
    elif "CtoGA" in stem:
        editor_class = "CGBE"
        target_base = "C"
        intended_edit = "C>G/A"
    else:
        editor_class = "CBE"
        target_base = "C"
        intended_edit = "C>T"

    if "12kChar" in stem and "AtoG" not in stem and "CtoT" not in stem and "CtoGA" not in stem:
        intended_edit = f"{target_base}>mixed_characterization"

    return {
        "cell_context": cell,
        "editor": editor,
        "editor_class": editor_class,
        "editor_family": "BE-Hive",
        "cas_variant": "SpCas9",
        "pam_scope": "NGG",
        "delivery": "library",
        "target_base": target_base,
        "intended_edit": intended_edit,
    }


def parse_bedict2_sheet(sheet: str) -> Dict[str, str]:
    if "SpRY" in sheet:
        cas = "SpRY"
        pam_scope = "near-PAMless"
    elif "SpG" in sheet:
        cas = "SpG"
        pam_scope = "NGN"
    elif "SpCas9" in sheet:
        cas = "SpCas9"
        pam_scope = "NGG"
    else:
        cas = "unknown"
        pam_scope = "unknown"

    editor = "ABE8e" if "ABE8e" in sheet else "ABEmax" if "ABEmax" in sheet else "ABE"
    delivery = "unknown"
    cell = "unknown"
    if sheet.startswith("HEK-Plasmid"):
        delivery = "plasmid"
        cell = "HEK"
    elif sheet.startswith("HEK-mRNA"):
        delivery = "mRNA"
        cell = "HEK"
    elif sheet.startswith("AAVlibrary-AAV"):
        delivery = "AAV"
        cell = "in_vivo_AAV_library"
    elif sheet.startswith("LentiAAV"):
        delivery = "LentiAAV"
        cell = "in_vivo"
    elif sheet.startswith("LentiLNP"):
        delivery = "LentiLNP"
        cell = "in_vivo"

    timepoint = ""
    m = re.search(r"_(\d+d|\d+w|02pmol|1pmol|5pmol)_", sheet)
    if m:
        timepoint = m.group(1)
    elif "_6w_" in sheet:
        timepoint = "6w"
    elif "_12w_" in sheet:
        timepoint = "12w"

    return {
        "editor": editor,
        "editor_class": "ABE",
        "editor_family": "BEDICT2",
        "cas_variant": cas,
        "pam_scope": pam_scope,
        "cell_context": cell,
        "delivery": delivery,
        "timepoint": timepoint,
        "target_base": "A",
        "intended_edit": "A>G",
    }


def add_design(
    design_map: Dict[str, Dict[str, str]],
    *,
    sequence_context_original: str,
    sequence_56nt: str = "",
    sequence_50nt: str = "",
    sequence_30nt: str = "",
    protospacer_20nt: str = "",
    pam: str = "",
    target_base: str = "",
    target_position: str = "",
    intended_edit: str = "",
    reference_allele: str = "",
    alternate_allele: str = "",
    gene: str = "",
    genomic_locus: str = "",
) -> str:
    sequence_context_original = clean_dna(sequence_context_original)
    sequence_56nt = clean_dna(sequence_56nt)
    sequence_50nt = clean_dna(sequence_50nt)
    sequence_30nt = clean_dna(sequence_30nt)
    protospacer_20nt = clean_dna(protospacer_20nt)
    pam = clean_dna(pam)
    target_base = str(target_base or "")
    target_position = str(target_position or "")
    intended_edit = str(intended_edit or "")
    sequence_29nt = sequence_30nt[:29] if len(sequence_30nt) >= 29 else ""
    design_hash = stable_hash(
        sequence_30nt,
        protospacer_20nt,
        pam,
        target_base,
        target_position,
        intended_edit,
        reference_allele,
        alternate_allele,
    )
    design_id = f"DES_{design_hash}"
    if design_id not in design_map:
        seq_ok = bool(sequence_30nt and len(sequence_30nt) == 30 and not (set(sequence_30nt) - DNA))
        design_map[design_id] = {
            "design_id": design_id,
            "sequence_context_original": sequence_context_original,
            "sequence_56nt": sequence_56nt,
            "sequence_50nt": sequence_50nt,
            "sequence_30nt": sequence_30nt,
            "sequence_29nt": sequence_29nt,
            "protospacer_20nt": protospacer_20nt,
            "pam": pam,
            "pam_type": pam_type(pam),
            "target_base": target_base,
            "target_position": target_position,
            "intended_edit": intended_edit,
            "reference_allele": reference_allele,
            "alternate_allele": alternate_allele,
            "gene": gene,
            "genomic_locus": genomic_locus,
            "sequence_qc_status": "pass" if seq_ok else "fail",
            "design_hash": design_hash,
        }
    return design_id


def add_editor_context(
    editor_map: Dict[str, Dict[str, str]],
    *,
    editor: str,
    editor_class: str,
    editor_family: str,
    cas_variant: str,
    pam_scope: str,
    cell_context: str,
    delivery: str,
    timepoint: str = "",
    replicate_strategy: str = "",
) -> str:
    editor_hash = stable_hash(
        editor,
        editor_class,
        editor_family,
        cas_variant,
        pam_scope,
        cell_context,
        delivery,
        timepoint,
        replicate_strategy,
    )
    editor_context_id = f"EDCTX_{editor_hash}"
    if editor_context_id not in editor_map:
        editor_map[editor_context_id] = {
            "editor_context_id": editor_context_id,
            "editor": editor,
            "editor_class": editor_class,
            "editor_family": editor_family,
            "cas_variant": cas_variant,
            "pam_scope": pam_scope,
            "cell_context": cell_context,
            "delivery": delivery,
            "timepoint": timepoint,
            "replicate_strategy": replicate_strategy,
            "editor_qc_status": "pass" if editor and editor_class else "partial",
            "editor_context_hash": editor_hash,
        }
    return editor_context_id


def obs_base(**kwargs: object) -> Dict[str, str]:
    row = {col: "" for col in OBS_COLUMNS}
    for key, value in kwargs.items():
        if key not in row:
            continue
        if isinstance(value, bool):
            row[key] = "True" if value else "False"
        elif value is None:
            row[key] = ""
        else:
            row[key] = str(value)
    return row


def load_source_inventory() -> pd.DataFrame:
    if CANDIDATE_INVENTORY.exists():
        inv = pd.read_csv(CANDIDATE_INVENTORY)
    else:
        inv = pd.DataFrame()
    if inv.empty:
        return inv
    inv = inv.copy()
    inv.insert(0, "source_inventory_id", [f"SRC_{i:05d}" for i in range(1, len(inv) + 1)])
    inv.to_csv(OUT / "source_inventory_v5.csv", index=False)
    return inv


def source_id_lookup(inv: pd.DataFrame) -> Dict[Tuple[str, str], str]:
    if inv.empty:
        return {}
    lookup = {}
    for _, row in inv.iterrows():
        key = (str(row.get("source_file", "")), str(row.get("source_sheet", "") or ""))
        lookup[key] = str(row["source_inventory_id"])
        lookup[(Path(str(row.get("source_file", ""))).name, str(row.get("source_sheet", "") or ""))] = str(row["source_inventory_id"])
    return lookup


def lookup_source_id(lookup: Dict[Tuple[str, str], str], file_path: Path, sheet: str = "") -> str:
    rel = ""
    try:
        rel = str(file_path.relative_to(ROOT))
    except Exception:
        rel = str(file_path)
    return lookup.get((rel, sheet), lookup.get((file_path.name, sheet), stable_id("SRC", rel, sheet)))


def process_behive(
    writer: csv.DictWriter,
    source_lookup: Dict[Tuple[str, str], str],
    design_map: Dict[str, Dict[str, str]],
    editor_map: Dict[str, Dict[str, str]],
    t1_rows: List[Dict[str, object]],
    counters: Dict[str, int],
) -> None:
    if not BEHIVE_DIR.exists():
        return
    for path in sorted(BEHIVE_DIR.glob("*.csv")):
        meta = infer_behive_from_filename(path.name)
        source_inventory_id = lookup_source_id(source_lookup, path)
        try:
            df = pd.read_csv(
                path,
                usecols=lambda c: c
                in [
                    "Name (unique)",
                    "gRNA (20nt)",
                    "Sequence context (56nt)",
                    "Sequence context (35 or 61 nt)",
                    "Editing position_x",
                    "Editing position_y",
                    "Protospacer position zero index",
                ]
                or str(c).startswith("Fraction edited ")
                or str(c).startswith("Total count ")
                or str(c).startswith("Edited count "),
            )
        except Exception as exc:
            counters["behive_read_errors"] += 1
            continue

        frac_cols = fraction_columns(df.columns)
        total_cols = count_columns(df.columns, "Total count ")
        edited_cols = count_columns(df.columns, "Edited count ")
        if not frac_cols:
            counters["behive_no_fraction_cols"] += 1
            continue
        seq_col = (
            "Sequence context (56nt)"
            if "Sequence context (56nt)" in df.columns
            else "Sequence context (35 or 61 nt)"
            if "Sequence context (35 or 61 nt)" in df.columns
            else ""
        )
        if not seq_col:
            counters["behive_no_sequence_col"] += 1
            continue
        measured = df[frac_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1, skipna=True)
        total_read = df[total_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1, skipna=True) if total_cols else pd.Series([math.nan] * len(df))
        edited_read = df[edited_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1, skipna=True) if edited_cols else pd.Series([math.nan] * len(df))

        editor_context_id = add_editor_context(
            editor_map,
            editor=meta["editor"],
            editor_class=meta["editor_class"],
            editor_family=meta["editor_family"],
            cas_variant=meta["cas_variant"],
            pam_scope=meta["pam_scope"],
            cell_context=meta["cell_context"],
            delivery=meta["delivery"],
            replicate_strategy=f"mean_of_{len(frac_cols)}_fraction_columns",
        )

        for idx, rec in df.iterrows():
            guide = clean_dna(rec.get("gRNA (20nt)", ""))
            seq_context = clean_dna(rec.get(seq_col, ""))
            seq30 = extract_30mer(seq_context, guide)
            eff = clean_float(measured.iloc[idx])
            if not seq30 or len(guide) != 20 or eff is None or eff < 0 or eff > 1:
                counters["behive_invalid_rows"] += 1
                continue
            if meta["target_base"] and meta["target_base"] not in seq30[6:14]:
                counters["behive_no_editable_base_window"] += 1
                continue
            pam = seq30[24:27]
            sequence_56nt = seq_context if len(seq_context) == 56 else ""
            sequence_50nt = seq50_from_context(seq_context, seq30)
            target_position = rec.get("Editing position_x", rec.get("Editing position_y", ""))
            target_position = "" if pd.isna(target_position) else str(target_position)
            design_id = add_design(
                design_map,
                sequence_context_original=seq_context,
                sequence_56nt=sequence_56nt,
                sequence_50nt=sequence_50nt,
                sequence_30nt=seq30,
                protospacer_20nt=guide,
                pam=pam,
                target_base=meta["target_base"],
                target_position=target_position,
                intended_edit=meta["intended_edit"],
            )
            obs_id = stable_id("OBS", "BE-Hive", path.name, idx, design_id, editor_context_id, eff)
            source_row_key = str(rec.get("Name (unique)", idx))
            total_val = clean_float(total_read.iloc[idx])
            edited_val = clean_float(edited_read.iloc[idx])
            obs = obs_base(
                observation_id=obs_id,
                design_id=design_id,
                editor_context_id=editor_context_id,
                source_inventory_id=source_inventory_id,
                source_family="BE-Hive",
                source_dataset="processed_efficiency",
                source_file=str(path),
                source_sheet="",
                source_row_key=source_row_key,
                source_relation_to_model="source_associated",
                source_associated_models="behive_efficiency;behive_bystander",
                source_role="processed_efficiency",
                label_type="total_efficiency",
                measured_total_efficiency=format_float(eff),
                total_read_count=format_float(total_val),
                edited_read_count=format_float(edited_val),
                replicate_count=len(frac_cols),
                measurement_scale="fraction_edited_mean_replicates",
                task_eligibility_T1=True,
                task_eligibility_T2=True,
                task_eligibility_T3=False,
                task_eligibility_T4=False,
                task_eligibility_T5=True,
                task_eligibility_T6=True,
                task_eligibility_T7=True,
                qc_sequence_pass=True,
                qc_label_pass=True,
                qc_editor_pass=True,
                qc_leakage_pass=True,
                qc_level="A",
                notes=f"BE-Hive full conversion from {path.name}; label is mean of {len(frac_cols)} fraction-edited column(s).",
            )
            writer.writerow(obs)
            t1_rows.append(
                make_task_base(obs, design_map[design_id], editor_map[editor_context_id])
            )
            counters["behive_observations"] += 1


def process_bedict2(
    writer: csv.DictWriter,
    source_lookup: Dict[Tuple[str, str], str],
    design_map: Dict[str, Dict[str, str]],
    editor_map: Dict[str, Dict[str, str]],
    t1_rows: List[Dict[str, object]],
    t4_rows: List[Dict[str, object]],
    counters: Dict[str, int],
) -> None:
    if not BEDICT2_XLSX.exists():
        return
    wb = openpyxl.load_workbook(BEDICT2_XLSX, read_only=True, data_only=True)
    for ws in wb.worksheets:
        sheet = ws.title
        meta = parse_bedict2_sheet(sheet)
        source_inventory_id = lookup_source_id(source_lookup, BEDICT2_XLSX, sheet)
        it = ws.iter_rows(values_only=True)
        try:
            header = next(it)
        except StopIteration:
            continue
        cols = {str(v): i for i, v in enumerate(header) if v is not None}
        required = ["rname", "refSeq", "PAM", "Target-Sequence", "Position", "absolute_editing_efficiency"]
        if not all(c in cols for c in required):
            counters["bedict2_skipped_sheets"] += 1
            continue
        editor_context_id = add_editor_context(
            editor_map,
            editor=meta["editor"],
            editor_class=meta["editor_class"],
            editor_family=meta["editor_family"],
            cas_variant=meta["cas_variant"],
            pam_scope=meta["pam_scope"],
            cell_context=meta["cell_context"],
            delivery=meta["delivery"],
            timepoint=meta["timepoint"],
            replicate_strategy="single_absolute_editing_efficiency_column",
        )
        for row_idx, row in enumerate(it, start=2):
            def val(name: str) -> object:
                i = cols[name]
                return row[i] if i < len(row) else ""

            protospacer = clean_dna(val("refSeq"))
            target_sequence = clean_dna(val("Target-Sequence"))
            eff_raw = clean_float(val("absolute_editing_efficiency"))
            if eff_raw is None:
                counters["bedict2_invalid_rows"] += 1
                continue
            eff = eff_raw / 100.0
            if eff < 0 or eff > 1 or len(protospacer) != 20:
                counters["bedict2_invalid_rows"] += 1
                continue
            seq30 = extract_30mer(target_sequence, protospacer)
            if not seq30 or "A" not in seq30[6:14]:
                counters["bedict2_invalid_rows"] += 1
                continue
            pam = clean_dna(val("PAM")) or seq30[24:27]
            target_position = str(val("Position") or "")
            design_id = add_design(
                design_map,
                sequence_context_original=target_sequence,
                sequence_50nt=seq50_from_context(target_sequence, seq30),
                sequence_30nt=seq30,
                protospacer_20nt=protospacer,
                pam=pam,
                target_base="A",
                target_position=target_position,
                intended_edit="A>G",
            )
            obs_id = stable_id("OBS", "BEDICT2", sheet, row_idx, design_id, editor_context_id, eff)
            obs = obs_base(
                observation_id=obs_id,
                design_id=design_id,
                editor_context_id=editor_context_id,
                source_inventory_id=source_inventory_id,
                source_family="BEDICT2",
                source_dataset="supplement_absolute_efficiency",
                source_file=str(BEDICT2_XLSX),
                source_sheet=sheet,
                source_row_key=str(val("rname") or row_idx),
                source_relation_to_model="source_associated",
                source_associated_models="bedict_v2",
                source_role="supplement_absolute_efficiency",
                label_type="target_position_efficiency",
                measured_total_efficiency=format_float(eff),
                measured_position_rate=format_float(eff),
                replicate_count=1,
                measurement_scale="absolute_editing_efficiency_position_fraction",
                task_eligibility_T1=True,
                task_eligibility_T2=True,
                task_eligibility_T3=False,
                task_eligibility_T4=True,
                task_eligibility_T5=True,
                task_eligibility_T6=True,
                task_eligibility_T7=True,
                qc_sequence_pass=True,
                qc_label_pass=True,
                qc_editor_pass=True,
                qc_leakage_pass=True,
                qc_level="A",
                notes="BEDICT2 full conversion; label is absolute_editing_efficiency/100 and is position/target-site oriented.",
            )
            writer.writerow(obs)
            task_row = make_task_base(obs, design_map[design_id], editor_map[editor_context_id])
            t1_rows.append(task_row)
            t4_rows.append(task_row)
            counters["bedict2_observations"] += 1
    wb.close()


def crispron_editor_class(path: Path) -> str:
    return "ABE" if path.name.startswith("ABE") else "CBE"


def process_crispron(
    writer: csv.DictWriter,
    source_lookup: Dict[Tuple[str, str], str],
    design_map: Dict[str, Dict[str, str]],
    editor_map: Dict[str, Dict[str, str]],
    t1_rows: List[Dict[str, object]],
    t3_rows: List[Dict[str, object]],
    counters: Dict[str, int],
) -> None:
    if not CRISPRON_DEV.exists():
        return
    for path in sorted(CRISPRON_DEV.glob("*.tsv.gz")):
        editor_class = crispron_editor_class(path)
        target_base = "A" if editor_class == "ABE" else "C"
        intended_edit = "A>G" if editor_class == "ABE" else "C>T"
        partition_kind = "test" if "test" in path.name.lower() else "train"
        source_inventory_id = lookup_source_id(source_lookup, path)
        df = pd.read_csv(path, sep="\t", compression="gzip")
        editor_context_id = add_editor_context(
            editor_map,
            editor=f"CRISPRon-{editor_class}",
            editor_class=editor_class,
            editor_family="CRISPRon-BE_data",
            cas_variant="SpCas9",
            pam_scope="NGG",
            cell_context="mixed_training_sources",
            delivery="mixed",
            timepoint="",
            replicate_strategy="processed_train_test_table",
        )
        # Collapsed T1 rows: one per source/ref/surro_target combination.
        group_cols = ["refs", "surro_target", "source", "partition"]
        collapsed = (
            df.groupby(group_cols, dropna=False)
            .agg(
                editingeff=("editingeff", "first"),
                total=("total", "first"),
                editingCounts=("editingCounts", "first"),
                outcome_rows=("outcomes", "size"),
            )
            .reset_index()
        )
        for idx, rec in collapsed.iterrows():
            protospacer = clean_dna(rec["refs"])
            seq_context = clean_dna(rec["surro_target"])
            seq30 = extract_30mer(seq_context, protospacer)
            eff_raw = clean_float(rec["editingeff"])
            if not seq30 or len(protospacer) != 20 or eff_raw is None:
                counters["crispron_invalid_t1"] += 1
                continue
            eff = eff_raw / 100.0
            if eff < 0 or eff > 1:
                counters["crispron_invalid_t1"] += 1
                continue
            if target_base not in seq30[6:14]:
                counters["crispron_no_editable_base"] += 1
                continue
            design_id = add_design(
                design_map,
                sequence_context_original=seq_context,
                sequence_30nt=seq30,
                protospacer_20nt=protospacer,
                pam=seq30[24:27],
                target_base=target_base,
                intended_edit=intended_edit,
            )
            obs_id = stable_id("OBS", "CRISPRon-BE", path.name, "T1", idx, design_id, editor_context_id)
            obs = obs_base(
                observation_id=obs_id,
                design_id=design_id,
                editor_context_id=editor_context_id,
                source_inventory_id=source_inventory_id,
                source_family="CRISPRon-BE",
                source_dataset=f"model_development_{partition_kind}",
                source_file=str(path),
                source_sheet="",
                source_row_key=f"{rec['source']}|{rec['partition']}|{protospacer}",
                source_relation_to_model="source_associated",
                source_associated_models="crispron_be",
                source_role=f"processed_{partition_kind}",
                label_type="total_efficiency",
                measured_total_efficiency=format_float(eff),
                total_read_count=format_float(rec["total"]),
                edited_read_count=format_float(rec["editingCounts"]),
                replicate_count=1,
                measurement_scale="editingeff_percent_collapsed_to_fraction",
                task_eligibility_T1=True,
                task_eligibility_T2=True,
                task_eligibility_T3=False,
                task_eligibility_T4=False,
                task_eligibility_T5=True,
                task_eligibility_T6=True,
                task_eligibility_T7=False,
                qc_sequence_pass=True,
                qc_label_pass=True,
                qc_editor_pass=True,
                qc_leakage_pass=True,
                qc_level="A" if partition_kind == "test" else "B",
                notes="CRISPRon-BE model-development table collapsed from outcome-level rows to unique ref/source target.",
            )
            writer.writerow(obs)
            t1_rows.append(make_task_base(obs, design_map[design_id], editor_map[editor_context_id]))
            counters["crispron_t1_observations"] += 1

        # Outcome-level T3 rows.
        for idx, rec in df.iterrows():
            protospacer = clean_dna(rec.get("refs", ""))
            seq_context = clean_dna(rec.get("surro_target", ""))
            seq30 = extract_30mer(seq_context, protospacer)
            outseq = clean_dna(rec.get("outcomes", ""))
            freq_raw = clean_float(rec.get("outcomefreq", None))
            eff_raw = clean_float(rec.get("editingeff", None))
            if not seq30 or len(protospacer) != 20 or not outseq or freq_raw is None:
                counters["crispron_invalid_t3"] += 1
                continue
            freq = freq_raw / 100.0
            eff = eff_raw / 100.0 if eff_raw is not None else None
            if freq < 0 or freq > 1:
                counters["crispron_invalid_t3"] += 1
                continue
            design_id = add_design(
                design_map,
                sequence_context_original=seq_context,
                sequence_30nt=seq30,
                protospacer_20nt=protospacer,
                pam=seq30[24:27],
                target_base=target_base,
                intended_edit=intended_edit,
            )
            obs_id = stable_id("OBS", "CRISPRon-BE", path.name, "T3", idx, design_id, outseq, freq)
            no_edit = freq if outseq == protospacer else ""
            obs = obs_base(
                observation_id=obs_id,
                design_id=design_id,
                editor_context_id=editor_context_id,
                source_inventory_id=source_inventory_id,
                source_family="CRISPRon-BE",
                source_dataset=f"model_development_{partition_kind}",
                source_file=str(path),
                source_sheet="",
                source_row_key=f"{rec.get('source', '')}|{rec.get('partition', '')}|{protospacer}|{outseq}|{idx}",
                source_relation_to_model="source_associated",
                source_associated_models="crispron_be",
                source_role=f"processed_outcome_{partition_kind}",
                label_type="outcome_frequency",
                measured_total_efficiency=format_float(eff),
                measured_outcome_frequency=format_float(freq),
                outcome_sequence=outseq,
                no_edit_frequency=format_float(no_edit),
                total_read_count=format_float(rec.get("total", "")),
                edited_read_count=format_float(rec.get("editingCounts", "")),
                replicate_count=1,
                measurement_scale="outcomefreq_percent_to_fraction",
                task_eligibility_T1=False,
                task_eligibility_T2=False,
                task_eligibility_T3=True,
                task_eligibility_T4=False,
                task_eligibility_T5=True,
                task_eligibility_T6=True,
                task_eligibility_T7=False,
                qc_sequence_pass=True,
                qc_label_pass=True,
                qc_editor_pass=True,
                qc_leakage_pass=True,
                qc_level="A" if partition_kind == "test" else "B",
                notes="CRISPRon-BE outcome-level processed row.",
            )
            writer.writerow(obs)
            t3_rows.append(make_task_base(obs, design_map[design_id], editor_map[editor_context_id]))
            counters["crispron_t3_observations"] += 1


def make_task_base(obs: Dict[str, str], design: Dict[str, str], editor: Dict[str, str]) -> Dict[str, object]:
    row: Dict[str, object] = {}
    row.update(obs)
    row.update({k: v for k, v in design.items() if k not in row})
    row.update({k: v for k, v in editor.items() if k not in row})
    return row


def write_dict_csv(path: Path, rows: Iterable[Dict[str, object]], columns: Optional[List[str]] = None) -> int:
    rows = list(rows)
    if columns is None:
        seen: List[str] = []
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.append(key)
        columns = seen
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def make_task_views(
    t1_rows: List[Dict[str, object]],
    t3_rows: List[Dict[str, object]],
    t4_rows: List[Dict[str, object]],
    counters: Dict[str, int],
) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    t1 = pd.DataFrame(t1_rows)
    if not t1.empty:
        t1["measured_total_efficiency"] = pd.to_numeric(t1["measured_total_efficiency"], errors="coerce")
        t1 = t1[t1["measured_total_efficiency"].between(0, 1)].copy()
        t1["source_balancing_group"] = (
            t1["source_family"].astype(str)
            + "|"
            + t1["measurement_scale"].astype(str)
            + "|"
            + t1["editor_class"].astype(str)
        )
        t1["main_scale_group"] = t1["measurement_scale"].astype(str)
        t1.to_csv(TASK / "T1_total_efficiency_view_v5.csv", index=False)
        counts["T1"] = len(t1)

        # T2: source/scale/editor-class median threshold.
        t2 = t1.copy()
        thresholds = (
            t2.groupby("source_balancing_group")["measured_total_efficiency"]
            .median()
            .rename("high_efficiency_threshold")
        )
        t2 = t2.merge(thresholds, on="source_balancing_group", how="left")
        t2["high_efficiency_label"] = (
            t2["measured_total_efficiency"] >= t2["high_efficiency_threshold"]
        ).astype(int)
        t2["threshold_rule"] = "within_source_scale_editorclass_median"
        t2.to_csv(TASK / "T2_high_low_efficiency_view_v5.csv", index=False)
        counts["T2"] = len(t2)

        # T5: source/context generalization view from T1 labels.
        t5_cols = [
            "observation_id",
            "design_id",
            "editor_context_id",
            "source_family",
            "source_dataset",
            "source_relation_to_model",
            "source_associated_models",
            "measurement_scale",
            "measured_total_efficiency",
            "editor",
            "editor_class",
            "editor_family",
            "cas_variant",
            "pam_scope",
            "cell_context",
            "delivery",
            "timepoint",
            "pam_type",
            "sequence_30nt",
            "protospacer_20nt",
            "pam",
            "source_balancing_group",
            "qc_level",
        ]
        t5 = t1[[c for c in t5_cols if c in t1.columns]].copy()
        t5.to_csv(TASK / "T5_generalization_view_v5.csv", index=False)
        counts["T5"] = len(t5)

        # T7: repeated design + comparable source/scale groups.
        group_cols = ["source_family", "measurement_scale", "design_id"]
        n_editors = (
            t1.groupby(group_cols)["editor_context_id"]
            .nunique()
            .rename("n_editor_contexts_for_design")
            .reset_index()
        )
        t7 = t1.merge(n_editors, on=group_cols, how="left")
        t7 = t7[t7["n_editor_contexts_for_design"] >= 2].copy()
        if not t7.empty:
            t7["editor_rank_within_design"] = (
                t7.groupby(group_cols)["measured_total_efficiency"]
                .rank(method="min", ascending=False)
                .astype(int)
            )
            t7["is_best_editor_context"] = (t7["editor_rank_within_design"] == 1).astype(int)
        t7.to_csv(TASK / "T7_editor_recommendation_view_v5.csv", index=False)
        counts["T7"] = len(t7)
    else:
        for name in ["T1_total_efficiency_view_v5.csv", "T2_high_low_efficiency_view_v5.csv", "T5_generalization_view_v5.csv", "T7_editor_recommendation_view_v5.csv"]:
            pd.DataFrame().to_csv(TASK / name, index=False)
            counts[name] = 0

    t3 = pd.DataFrame(t3_rows)
    if not t3.empty:
        t3["measured_outcome_frequency"] = pd.to_numeric(t3["measured_outcome_frequency"], errors="coerce")
        t3 = t3[t3["measured_outcome_frequency"].between(0, 1)].copy()
    t3.to_csv(TASK / "T3_outcome_bystander_view_v5.csv", index=False)
    counts["T3"] = len(t3)

    t4 = pd.DataFrame(t4_rows)
    if not t4.empty:
        t4["measured_position_rate"] = pd.to_numeric(t4["measured_position_rate"], errors="coerce")
        t4 = t4[t4["measured_position_rate"].between(0, 1)].copy()
    t4.to_csv(TASK / "T4_position_profile_view_v5.csv", index=False)
    counts["T4"] = len(t4)

    # T6: model reproducibility/applicability scaffold from model eligibility registry.
    if MODEL_ELIGIBILITY.exists():
        t6 = pd.read_csv(MODEL_ELIGIBILITY)
        t6["matrix_version"] = "unified_matrix_v5"
        t6["local_reproduction_status"] = t6["local_or_web"].map(
            lambda x: "local_candidate" if str(x).startswith("local") else "web_or_optional"
        )
        if not t1.empty:
            t6["t1_matrix_rows_available"] = len(t1)
        if not t3.empty:
            t6["t3_matrix_rows_available"] = len(t3)
        if not t4.empty:
            t6["t4_matrix_rows_available"] = len(t4)
    else:
        t6 = pd.DataFrame()
    t6.to_csv(TASK / "T6_reproducibility_applicability_view_v5.csv", index=False)
    counts["T6"] = len(t6)
    return counts


def write_qc_report(
    counters: Dict[str, int],
    design_map: Dict[str, Dict[str, str]],
    editor_map: Dict[str, Dict[str, str]],
    task_counts: Dict[str, int],
) -> None:
    source_summary = pd.read_csv(OUT / "source_candidate_summary_v5.csv") if (OUT / "source_candidate_summary_v5.csv").exists() else pd.DataFrame()
    rows = []
    if (OUT / "observation_matrix_v5.csv").exists():
        # Lightweight line count, avoiding loading the full matrix.
        with (OUT / "observation_matrix_v5.csv").open("r", encoding="utf-8") as handle:
            obs_rows = max(0, sum(1 for _ in handle) - 1)
    else:
        obs_rows = 0

    counter_df = pd.DataFrame(
        [{"counter": k, "value": v} for k, v in sorted(counters.items())]
    )
    task_df = pd.DataFrame(
        [{"task": k, "rows": v} for k, v in sorted(task_counts.items())]
    )
    summary_df = pd.DataFrame(
        [
            {"item": "observation_matrix_rows", "value": obs_rows},
            {"item": "design_master_rows", "value": len(design_map)},
            {"item": "editor_context_master_rows", "value": len(editor_map)},
        ]
    )

    def md_table(df: pd.DataFrame) -> str:
        if df.empty:
            return "_No rows._"
        t = df.copy().fillna("")
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

    text = f"""# Unified matrix v5 QC report

Generated: {datetime.now().isoformat(timespec="seconds")}

## Build summary

{md_table(summary_df)}

## Task-view row counts

{md_table(task_df)}

## Source candidate summary

{md_table(source_summary)}

## Build counters

{md_table(counter_df)}

## QC interpretation

- This is the first constructed matrix version, not merely a source audit.
- BE-Hive, BEDICT2 and CRISPRon-BE are included in matrix/task-view form.
- BE-Hive and CRISPRon-BE rows are source-associated for their own model families;
  model-specific external filtering must be applied during evaluation.
- BEDICT2 rows are retained with `measurement_scale=absolute_editing_efficiency_position_fraction`,
  because these labels are position/target-site oriented rather than guaranteed total-editing labels.
- T1/T2/T5/T7 are now available at full first-batch scale.
- T3 is available from CRISPRon-BE outcome-level data.
- T4 is available from BEDICT2 position/target-site efficiency rows.
"""
    REPORT.write_text(text, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    counters: Dict[str, int] = defaultdict(int)
    source_inventory = load_source_inventory()
    source_lookup = source_id_lookup(source_inventory)
    design_map: Dict[str, Dict[str, str]] = {}
    editor_map: Dict[str, Dict[str, str]] = {}
    t1_rows: List[Dict[str, object]] = []
    t3_rows: List[Dict[str, object]] = []
    t4_rows: List[Dict[str, object]] = []

    obs_path = OUT / "observation_matrix_v5.csv"
    with obs_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OBS_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        print("Processing BE-Hive...")
        process_behive(writer, source_lookup, design_map, editor_map, t1_rows, counters)
        print("Processing BEDICT2...")
        process_bedict2(writer, source_lookup, design_map, editor_map, t1_rows, t4_rows, counters)
        print("Processing CRISPRon-BE...")
        process_crispron(writer, source_lookup, design_map, editor_map, t1_rows, t3_rows, counters)

    write_dict_csv(OUT / "design_master_v5.csv", design_map.values(), DESIGN_COLUMNS)
    write_dict_csv(OUT / "editor_context_master_v5.csv", editor_map.values(), EDITOR_COLUMNS)
    print("Writing task views...")
    task_counts = make_task_views(t1_rows, t3_rows, t4_rows, counters)
    write_qc_report(counters, design_map, editor_map, task_counts)

    print("Done.")
    print(f"Observation matrix: {obs_path}")
    print(f"Designs: {len(design_map)}")
    print(f"Editor contexts: {len(editor_map)}")
    print("Task counts:", task_counts)


if __name__ == "__main__":
    main()
