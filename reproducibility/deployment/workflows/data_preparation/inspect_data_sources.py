"""
Inspect local Tier A base-editor benchmark data sources.

Outputs:
  - benchmark_core/reports/data_inventory_tier_a.csv
  - benchmark_core/reports/tier_a_schema_mapping.csv
  - benchmark_core/reports/TIER_A_SCHEMA_MAPPING.md

This script is intentionally read-only for source data. It only writes reports
under benchmark_core/reports.
"""

from __future__ import annotations

import csv
import gzip
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from runtime_paths import MODELS_ROOT, REPORT_ROOT

import openpyxl
import pandas as pd


PROJECT_ROOT = MODELS_ROOT
REPORT_DIR = REPORT_ROOT

MAX_SAMPLE_ROWS = 50
MAX_SAMPLE_COLS = 220
MAX_PREVIEW_COLS = 80
MAX_SAMPLE_VALUES = 5


SCHEMA_FIELDS = [
    "benchmark_id",
    "dataset_id",
    "source_type",
    "editor",
    "editor_class",
    "cell_context",
    "delivery",
    "cas_variant",
    "sequence_context",
    "protospacer_20nt",
    "pam",
    "target_base",
    "target_position",
    "intended_edit",
    "reference_outcome",
    "measured_efficiency",
    "measured_outcome_frequency",
    "split",
    "notes",
]


@dataclass(frozen=True)
class DataSource:
    source_id: str
    title: str
    root_or_file: Path
    patterns: tuple[str, ...]
    benchmark_role: str
    leakage_note: str


TIER_A_SOURCES = [
    DataSource(
        source_id="BEDICT2_SUPP_LOCAL",
        title="BEDICT2.0 paper supplements",
        root_or_file=PROJECT_ROOT / "BEDICT-V2" / "resources" / "paper_supplements",
        patterns=("*.xlsx",),
        benchmark_role="independent/challenge for non-BEDICT models",
        leakage_note="high for BEDICT-V2 / BEDICT lineage",
    ),
    DataSource(
        source_id="CRISPRONBE_SUPP_LOCAL",
        title="CRISPRon-BE processed model-development data",
        root_or_file=PROJECT_ROOT
        / "CRISPRon-BE"
        / "repo"
        / "data_for_model_development_V1.0",
        patterns=("*.xlsx", "*.tsv.gz"),
        benchmark_role="multi-dataset benchmark and split template",
        leakage_note="high for CRISPRon-BE if not using held-out logic",
    ),
    DataSource(
        source_id="EPCNNBE_LOCAL_TESTS",
        title="EPCNNBE local train/test and compared datasets",
        root_or_file=PROJECT_ROOT / "EPCNNBE" / "repo" / "tests",
        patterns=("*.csv", "*.xlsx"),
        benchmark_role="first normalizer target and reproducibility smoke benchmark",
        leakage_note="high for EPCNNBE; likely overlaps historical DeepBaseEditor data",
    ),
    DataSource(
        source_id="BE_ENDO_EXAMPLE_LOCAL",
        title="BE_Endo local smoke/example data",
        root_or_file=PROJECT_ROOT / "BE_Endo" / "outputs" / "CBE_BE_Seq_example_eff.tsv",
        patterns=("*.tsv",),
        benchmark_role="adapter validation only",
        leakage_note="too small for scientific benchmark",
    ),
]


FIELD_RULES: list[tuple[str, list[str], str]] = [
    (
        "benchmark_id",
        [
            r"\bid\b",
            r"original.*id",
            r"sample.*id",
            r"guide.*id",
            r"grna.*id",
            r"sg.*id",
            r"target.*id",
            r"unique.*id",
            r"site.*id",
        ],
        "identifier-like column",
    ),
    (
        "dataset_id",
        [r"dataset", r"study", r"source", r"library", r"screen", r"experiment", r"batch"],
        "dataset/study label",
    ),
    (
        "editor",
        [
            r"editor",
            r"base.*editor",
            r"\babe",
            r"\bcbe",
            r"be4",
            r"abe8",
            r"abemax",
            r"cbe4",
            r"target.?aid",
            r"spg",
            r"spry",
            r"ng\b",
        ],
        "editor or editor-setting column",
    ),
    (
        "editor_class",
        [r"editor.*class", r"base.*class", r"\babe\b", r"\bcbe\b", r"cgbe"],
        "editor class can often be inferred from editor/file name",
    ),
    (
        "cell_context",
        [
            r"cell",
            r"hek",
            r"k562",
            r"u2os",
            r"mouse",
            r"liver",
            r"in.?vivo",
            r"in.?vitro",
            r"context",
            r"tissue",
        ],
        "cell/tissue/experimental context",
    ),
    ("delivery", [r"delivery", r"aav", r"lnp", r"mrna", r"plasmid", r"electroporation"], "delivery context"),
    ("cas_variant", [r"cas", r"pam", r"spcas9", r"spry", r"spg", r"ng", r"nrrh", r"nrch"], "Cas/PAM variant"),
    (
        "sequence_context",
        [
            r"sequence",
            r"seq\b",
            r"target.*seq",
            r"context.*seq",
            r"protospacer.*pam",
            r"input",
            r"reference",
            r"wt",
        ],
        "longest measured/supplied DNA sequence",
    ),
    (
        "protospacer_20nt",
        [r"protospacer", r"spacer", r"guide", r"grna", r"sg.?rna", r"target.?20", r"guide.?seq", r"^target$"],
        "20-nt guide/protospacer",
    ),
    ("pam", [r"\bpam\b", r"pam.*seq", r"ngg", r"nann", r"ngnn"], "PAM sequence"),
    ("target_base", [r"target.*base", r"edit.*base", r"substrate", r"ref.*base"], "target base"),
    (
        "target_position",
        [r"position", r"\bpos\b", r"edit.*site", r"target.*site", r"window", r"nt.*position"],
        "editing/target position",
    ),
    (
        "intended_edit",
        [r"intended.*edit", r"edit.*type", r"mutation", r"variant", r"a.?to.?g", r"c.?to.?t", r"c.?to.?g"],
        "intended base conversion",
    ),
    (
        "reference_outcome",
        [r"^outcome$", r"outcome.*seq", r"edited.*seq", r"product", r"allele", r"mutant", r"genotype", r"observed.*seq"],
        "edited outcome/product sequence",
    ),
    (
        "measured_efficiency",
        [
            r"efficien",
            r"\beff\b",
            r"grna.*eff",
            r"editingeff",
            r"activity",
            r"editing.*rate",
            r"edit.*freq",
            r"edit.*percent",
            r"total.*edit",
            r"mean.*edit",
            r"observed",
            r"measured",
        ],
        "observed total editing efficiency/activity",
    ),
    (
        "measured_outcome_frequency",
        [r"outcome.*freq", r"outcomefreq", r"allele.*freq", r"proportion", r"\bpor\b", r"\bof\b", r"\bfreq\b", r"ratio", r"fraction", r"frequency"],
        "specific outcome frequency/proportion",
    ),
    ("split", [r"\bsplit\b", r"\btrain\b", r"\btest\b", r"valid", r"held.?out"], "train/test split label"),
]


DNA_RE = re.compile(r"^[ACGTUNacgtunNRYKMSWBDHVrykmswbdhv.-]+$")


def discover_files(source: DataSource) -> list[Path]:
    path = source.root_or_file
    if path.is_file():
        return [path]
    files: list[Path] = []
    for pattern in source.patterns:
        files.extend(path.glob(pattern))
    return sorted(set(files))


def safe_str(value: Any, limit: int = 140) -> str:
    if value is None:
        return ""
    text = str(value).replace("\r", " ").replace("\n", " ").strip()
    if len(text) > limit:
        return text[: limit - 1] + "…"
    return text


def normalize_column_name(name: Any, fallback: str) -> str:
    text = safe_str(name, 120)
    return text if text else fallback


def deduplicate_columns(columns: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out: list[str] = []
    for col in columns:
        base = col if col else "unnamed"
        seen[base] = seen.get(base, 0) + 1
        out.append(base if seen[base] == 1 else f"{base}.{seen[base]}")
    return out


def infer_editor_class_from_text(text: str) -> str:
    t = text.lower()
    has_abe = "abe" in t
    has_cbe = "cbe" in t or "be4" in t or "cytosine" in t
    has_cgbe = "cgbe" in t or "c-to-g" in t or "c2g" in t
    if has_cgbe:
        return "CGBE"
    if has_abe and has_cbe:
        return "ABE/CBE"
    if has_abe:
        return "ABE"
    if has_cbe:
        return "CBE"
    return ""


def infer_split_from_text(text: str) -> str:
    t = text.lower()
    if "train" in t:
        return "train"
    if "valid" in t or "validation" in t:
        return "validation"
    if "test" in t:
        return "test"
    if "endo" in t or "external" in t:
        return "external_test"
    return ""


def detect_header_row(rows: list[list[Any]]) -> int:
    """Return 0-based header row index from a small preview."""
    best_i = 0
    best_score = -1.0
    for i, row in enumerate(rows[:20]):
        nonempty = [safe_str(v) for v in row if safe_str(v)]
        if not nonempty:
            continue
        alpha = sum(any(ch.isalpha() for ch in x) for x in nonempty)
        unique = len(set(nonempty))
        # Favor text-rich and mostly unique rows near the top.
        score = alpha * 2 + unique - i * 0.15
        if score > best_score:
            best_i = i
            best_score = score
    return best_i


def simple_dtype(values: list[Any]) -> str:
    nonempty = [v for v in values if safe_str(v)]
    if not nonempty:
        return "empty"
    numeric = 0
    dna_like = 0
    for v in nonempty:
        text = safe_str(v)
        try:
            float(text)
            numeric += 1
        except Exception:
            pass
        if len(text) >= 8 and DNA_RE.match(text):
            dna_like += 1
    if dna_like >= max(1, len(nonempty) // 2):
        return "dna_sequence_like"
    if numeric == len(nonempty):
        return "numeric"
    if numeric:
        return "mixed_numeric_text"
    return "text"


def sample_values(values: Iterable[Any]) -> str:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = safe_str(value, 80)
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
        if len(out) >= MAX_SAMPLE_VALUES:
            break
    return " | ".join(out)


def map_column(column: str, dtype: str, samples: str, file_context: str) -> tuple[str, float, str]:
    # Column-level mapping should depend on the column name and observed values.
    # File/sheet names are used separately for table-level editor/split guesses;
    # including them here makes every column in a train/test/ABE/CBE file look
    # like a split/editor column.
    _ = file_context
    haystack = f"{column} {samples}".lower()
    best_field = ""
    best_score = 0.0
    best_reason = ""
    for field, patterns, reason in FIELD_RULES:
        score = 0.0
        for pat in patterns:
            if re.search(pat, haystack, flags=re.IGNORECASE):
                score += 1.0
        if field in {"sequence_context", "protospacer_20nt", "pam"} and dtype == "dna_sequence_like":
            score += 1.0
        if field in {"measured_efficiency", "measured_outcome_frequency", "target_position"} and dtype == "numeric":
            score += 0.2
        if score > best_score:
            best_field = field
            best_score = score
            best_reason = reason
    if not best_field or best_score < 1.0:
        return "", 0.0, "no confident mapping"
    confidence = min(0.95, 0.35 + best_score * 0.15)
    return best_field, round(confidence, 2), best_reason


def read_text_table(path: Path) -> tuple[list[str], list[list[Any]], int]:
    sep = "\t" if ".tsv" in "".join(path.suffixes).lower() else ","
    compression = "gzip" if path.suffix == ".gz" else None
    df = pd.read_csv(path, sep=sep, compression=compression, nrows=MAX_SAMPLE_ROWS, dtype=str)
    columns = [normalize_column_name(c, f"Column_{i+1}") for i, c in enumerate(df.columns)]
    sample_rows = df.fillna("").values.tolist()

    opener = gzip.open if compression == "gzip" else open
    with opener(path, "rt", encoding="utf-8", errors="replace", newline="") as handle:
        row_count = sum(1 for _ in handle)
    row_count = max(0, row_count - 1)
    return columns, sample_rows, row_count


def inspect_text_file(source: DataSource, path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    try:
        columns, sample_rows, row_count = read_text_table(path)
        columns = deduplicate_columns(columns)
    except Exception as exc:
        rel = str(path)
        return [
            {
                "source_id": source.source_id,
                "source_title": source.title,
                "file_path": rel,
                "sheet_or_table": "",
                "format": "".join(path.suffixes),
                "row_count": "",
                "col_count": "",
                "header_row": "",
                "columns_preview": "",
                "editor_class_guess": infer_editor_class_from_text(rel),
                "split_guess": infer_split_from_text(rel),
                "mapping_readiness": "read_error",
                "benchmark_role": source.benchmark_role,
                "leakage_note": source.leakage_note,
                "notes": f"Could not read text table: {exc}",
            }
        ], []

    file_context = f"{path.name} {source.title}"
    inventory = make_inventory_row(source, path, "", "".join(path.suffixes), row_count, len(columns), 1, columns)
    mapping = make_mapping_rows(source, path, "", columns, sample_rows, file_context)
    return [inventory], mapping


def inspect_xlsx_file(source: DataSource, path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    inventory_rows: list[dict[str, Any]] = []
    mapping_rows: list[dict[str, Any]] = []
    try:
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    except Exception as exc:
        inventory_rows.append(
            {
                "source_id": source.source_id,
                "source_title": source.title,
                "file_path": str(path),
                "sheet_or_table": "",
                "format": path.suffix,
                "row_count": "",
                "col_count": "",
                "header_row": "",
                "columns_preview": "",
                "editor_class_guess": infer_editor_class_from_text(str(path)),
                "split_guess": infer_split_from_text(str(path)),
                "mapping_readiness": "read_error",
                "benchmark_role": source.benchmark_role,
                "leakage_note": source.leakage_note,
                "notes": f"Could not read workbook: {exc}",
            }
        )
        return inventory_rows, mapping_rows

    for ws in wb.worksheets:
        sample_matrix: list[list[Any]] = []
        for row in ws.iter_rows(
            min_row=1,
            max_row=min(ws.max_row or 1, MAX_SAMPLE_ROWS),
            max_col=min(ws.max_column or 1, MAX_SAMPLE_COLS),
            values_only=True,
        ):
            sample_matrix.append(list(row))

        if sample_matrix:
            header_idx = detect_header_row(sample_matrix)
            raw_header = sample_matrix[header_idx]
            columns = [
                normalize_column_name(value, f"Column_{i+1}")
                for i, value in enumerate(raw_header[: min(len(raw_header), MAX_SAMPLE_COLS)])
            ]
            columns = deduplicate_columns(columns)
            sample_rows = sample_matrix[header_idx + 1 :]
        else:
            header_idx = 0
            columns = []
            sample_rows = []

        row_count = max(0, (ws.max_row or 0) - (header_idx + 1))
        col_count = ws.max_column or len(columns)
        inventory_rows.append(
            make_inventory_row(source, path, ws.title, path.suffix, row_count, col_count, header_idx + 1, columns)
        )
        file_context = f"{path.name} {ws.title} {source.title}"
        mapping_rows.extend(make_mapping_rows(source, path, ws.title, columns, sample_rows, file_context))

    return inventory_rows, mapping_rows


def make_inventory_row(
    source: DataSource,
    path: Path,
    sheet_or_table: str,
    fmt: str,
    row_count: int,
    col_count: int,
    header_row: int,
    columns: list[str],
) -> dict[str, Any]:
    context = f"{path.name} {sheet_or_table} {source.title}"
    column_text = " | ".join(columns[:MAX_PREVIEW_COLS])
    mapped_fields = sorted(
        {
            map_column(col, "unknown", "", context)[0]
            for col in columns
            if map_column(col, "unknown", "", context)[0]
        }
    )
    required_hits = {"editor_class", "sequence_context"} & set(mapped_fields)
    readiness = "needs_manual_mapping"
    if {"sequence_context"} <= set(mapped_fields) and len(mapped_fields) >= 4:
        readiness = "good_candidate"
    if len(required_hits) == 0 and len(mapped_fields) < 2:
        readiness = "low_mapping_signal"

    return {
        "source_id": source.source_id,
        "source_title": source.title,
        "file_path": str(path),
        "sheet_or_table": sheet_or_table,
        "format": fmt,
        "row_count": row_count,
        "col_count": col_count,
        "header_row": header_row,
        "columns_preview": column_text,
        "mapped_fields_guess": " | ".join(mapped_fields),
        "editor_class_guess": infer_editor_class_from_text(context),
        "split_guess": infer_split_from_text(context),
        "mapping_readiness": readiness,
        "benchmark_role": source.benchmark_role,
        "leakage_note": source.leakage_note,
        "notes": "",
    }


def make_mapping_rows(
    source: DataSource,
    path: Path,
    sheet_or_table: str,
    columns: list[str],
    sample_rows: list[list[Any]],
    file_context: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, column in enumerate(columns[:MAX_SAMPLE_COLS]):
        values = []
        for row in sample_rows:
            if i < len(row):
                values.append(row[i])
        dtype = simple_dtype(values)
        samples = sample_values(values)
        field, confidence, reason = map_column(column, dtype, samples, file_context)
        if not field:
            # Keep low-signal columns too, but make filtering easy.
            field = ""
        rows.append(
            {
                "source_id": source.source_id,
                "file_path": str(path),
                "sheet_or_table": sheet_or_table,
                "source_column": column,
                "source_column_index_1based": i + 1,
                "inferred_dtype": dtype,
                "sample_values": samples,
                "proposed_schema_field": field,
                "confidence": confidence,
                "mapping_reason": reason,
                "manual_review_needed": "yes" if confidence < 0.65 else "no",
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def postprocess_inventory_with_mapping(
    inventory_rows: list[dict[str, Any]], mapping_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    by_table: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in mapping_rows:
        key = (str(row["source_id"]), str(row["file_path"]), str(row.get("sheet_or_table") or ""))
        by_table.setdefault(key, []).append(row)

    out: list[dict[str, Any]] = []
    for inv in inventory_rows:
        inv = dict(inv)
        key = (str(inv["source_id"]), str(inv["file_path"]), str(inv.get("sheet_or_table") or ""))
        rows = by_table.get(key, [])
        fields = sorted(
            {
                str(r["proposed_schema_field"])
                for r in rows
                if str(r.get("proposed_schema_field") or "")
                and float(r.get("confidence") or 0) >= 0.5
            }
        )
        inv["mapped_fields_guess"] = " | ".join(fields)

        has_sequence = bool({"sequence_context", "protospacer_20nt"} & set(fields))
        has_measurement = bool(
            {"measured_efficiency", "measured_outcome_frequency", "reference_outcome"} & set(fields)
        )
        has_editor_context = bool(inv.get("editor_class_guess")) or "editor" in fields
        has_rows = str(inv.get("row_count", "")).isdigit() and int(inv.get("row_count") or 0) > 0

        if inv.get("mapping_readiness") == "read_error":
            readiness = "read_error"
        elif has_sequence and has_measurement and has_editor_context and has_rows:
            readiness = "good_candidate"
        elif has_sequence and has_measurement and has_rows:
            readiness = "needs_editor_context"
        elif has_sequence and has_rows:
            readiness = "needs_measurement_mapping"
        elif fields:
            readiness = "needs_manual_mapping"
        else:
            readiness = "low_mapping_signal"
        inv["mapping_readiness"] = readiness
        out.append(inv)
    return out


def md_escape(text: Any) -> str:
    s = safe_str(text, 180)
    return s.replace("|", "\\|")


def write_markdown_summary(inventory: list[dict[str, Any]], mapping: list[dict[str, Any]], output_path: Path) -> None:
    by_source: dict[str, list[dict[str, Any]]] = {}
    for row in inventory:
        by_source.setdefault(str(row["source_id"]), []).append(row)

    good = [r for r in inventory if r["mapping_readiness"] == "good_candidate"]
    needs = [r for r in inventory if r["mapping_readiness"] == "needs_manual_mapping"]
    low = [r for r in inventory if r["mapping_readiness"] == "low_mapping_signal"]
    errors = [r for r in inventory if r["mapping_readiness"] == "read_error"]

    lines: list[str] = []
    lines.append("# Tier A local data inventory and schema mapping")
    lines.append("")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Inventory tables/sheets inspected: {len(inventory)}")
    lines.append(f"- Column mapping rows generated: {len(mapping)}")
    lines.append(f"- Good schema candidates: {len(good)}")
    lines.append(f"- Need manual mapping: {len(needs)}")
    lines.append(f"- Low mapping signal: {len(low)}")
    lines.append(f"- Read errors: {len(errors)}")
    lines.append("")
    lines.append("## Source-level inventory")
    lines.append("")
    lines.append("| Source | Tables/sheets | Total rows | Notes |")
    lines.append("|---|---:|---:|---|")
    for source_id, rows in sorted(by_source.items()):
        total_rows = sum(int(r["row_count"] or 0) for r in rows if str(r["row_count"]).isdigit())
        notes = "; ".join(sorted(set(str(r["mapping_readiness"]) for r in rows)))
        lines.append(f"| {md_escape(source_id)} | {len(rows)} | {total_rows} | {md_escape(notes)} |")
    lines.append("")

    lines.append("## Highest-priority mapping candidates")
    lines.append("")
    lines.append("| Source | File | Sheet/table | Rows | Editor class | Split | Mapped fields |")
    lines.append("|---|---|---|---:|---|---|---|")
    for row in sorted(good, key=lambda r: (r["source_id"], str(r["file_path"]), str(r["sheet_or_table"])))[:40]:
        lines.append(
            "| "
            + " | ".join(
                [
                    md_escape(row["source_id"]),
                    md_escape(Path(str(row["file_path"])).name),
                    md_escape(row["sheet_or_table"]),
                    md_escape(row["row_count"]),
                    md_escape(row["editor_class_guess"]),
                    md_escape(row["split_guess"]),
                    md_escape(row["mapped_fields_guess"]),
                ]
            )
            + " |"
        )
    if not good:
        lines.append("")
        lines.append("No table reached the automatic `good_candidate` threshold; use the CSV mapping file for manual review.")
    lines.append("")

    lines.append("## Recommended unified schema mapping")
    lines.append("")
    lines.append("| Schema field | Typical Tier A source columns detected | Notes |")
    lines.append("|---|---|---|")
    for field in SCHEMA_FIELDS:
        hits = [
            r
            for r in mapping
            if r["proposed_schema_field"] == field and float(r["confidence"] or 0) >= 0.5
        ]
        examples = []
        for hit in hits[:8]:
            examples.append(f"{Path(str(hit['file_path'])).name}:{hit['sheet_or_table']}:{hit['source_column']}")
        note = "manual review" if not examples else "candidate"
        lines.append(f"| {field} | {md_escape('; '.join(examples))} | {note} |")
    lines.append("")

    lines.append("## Immediate interpretation")
    lines.append("")
    lines.append("- Use this inventory to choose columns, not as a final biological truth table.")
    lines.append("- Tables derived from model-associated repositories should be used carefully to avoid training/test leakage.")
    lines.append("- Rows with `low_mapping_signal` can still be useful as source metadata or example outputs, but are not first-choice benchmark tables.")
    lines.append("- The next script should create a normalized Tier A panel only from tables whose sequence, editor/editor_class, and measured efficiency/outcome columns are explicit.")
    lines.append("")
    lines.append("## Generated files")
    lines.append("")
    lines.append(f"- `{REPORT_DIR / 'data_inventory_tier_a.csv'}`")
    lines.append(f"- `{REPORT_DIR / 'tier_a_schema_mapping.csv'}`")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    inventory_rows: list[dict[str, Any]] = []
    mapping_rows: list[dict[str, Any]] = []

    for source in TIER_A_SOURCES:
        files = discover_files(source)
        if not files:
            inventory_rows.append(
                {
                    "source_id": source.source_id,
                    "source_title": source.title,
                    "file_path": str(source.root_or_file),
                    "sheet_or_table": "",
                    "format": "",
                    "row_count": "",
                    "col_count": "",
                    "header_row": "",
                    "columns_preview": "",
                    "mapped_fields_guess": "",
                    "editor_class_guess": infer_editor_class_from_text(str(source.root_or_file)),
                    "split_guess": infer_split_from_text(str(source.root_or_file)),
                    "mapping_readiness": "missing",
                    "benchmark_role": source.benchmark_role,
                    "leakage_note": source.leakage_note,
                    "notes": "No matching files found",
                }
            )
            continue

        for file_path in files:
            suffixes = "".join(file_path.suffixes).lower()
            if suffixes.endswith(".xlsx") or suffixes.endswith(".xlsm"):
                inv, mp = inspect_xlsx_file(source, file_path)
            elif suffixes.endswith(".csv") or suffixes.endswith(".tsv") or suffixes.endswith(".tsv.gz"):
                inv, mp = inspect_text_file(source, file_path)
            else:
                continue
            inventory_rows.extend(inv)
            mapping_rows.extend(mp)

    inventory_csv = REPORT_DIR / "data_inventory_tier_a.csv"
    mapping_csv = REPORT_DIR / "tier_a_schema_mapping.csv"
    summary_md = REPORT_DIR / "TIER_A_SCHEMA_MAPPING.md"
    summary_json = REPORT_DIR / "data_inventory_tier_a.json"

    inventory_rows = postprocess_inventory_with_mapping(inventory_rows, mapping_rows)

    write_csv(inventory_csv, inventory_rows)
    write_csv(mapping_csv, mapping_rows)
    write_markdown_summary(inventory_rows, mapping_rows, summary_md)
    summary_json.write_text(
        json.dumps(
            {
                "generated": datetime.now().isoformat(timespec="seconds"),
                "inventory_count": len(inventory_rows),
                "mapping_count": len(mapping_rows),
                "inventory_csv": str(inventory_csv),
                "mapping_csv": str(mapping_csv),
                "summary_md": str(summary_md),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"Wrote {inventory_csv} ({len(inventory_rows)} rows)")
    print(f"Wrote {mapping_csv} ({len(mapping_rows)} rows)")
    print(f"Wrote {summary_md}")
    print(f"Wrote {summary_json}")


if __name__ == "__main__":
    main()
