"""
Build download-link manifests for candidate base-editor benchmark datasets.

This script fetches metadata only:
  - figshare article file lists
  - ENA read-run file reports
  - HEAD sizes for known Springer/Nature supplementary files

It does NOT download dataset payloads.
"""

from __future__ import annotations

import csv
import json
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

from runtime_paths import DATA_ROOT, MODELS_ROOT


OUT = DATA_ROOT / "download_manifests"
RAW_OUT = OUT / "raw_large_ena_run_files"


FIGSHARE_ARTICLES = [
    {
        "dataset_id": "BEHIVE_EFF_FIGSHARE",
        "article_id": "10673816",
        "title": "BE-Hive processed editing efficiency data",
        "target_folder": str(DATA_ROOT / Path("manual_downloads/be_hive_efficiency")),
        "recommended": "yes",
        "notes": "Processed CSV files; useful for efficiency benchmark/development.",
    },
    {
        "dataset_id": "BEHIVE_BYST_FIGSHARE",
        "article_id": "10678097",
        "title": "BE-Hive processed bystander editing data",
        "target_folder": str(DATA_ROOT / Path("manual_downloads/be_hive_bystander")),
        "recommended": "yes",
        "notes": "Processed pickle files; useful for outcome/bystander benchmark.",
    },
    {
        "dataset_id": "FORECASTBE_FIGSHARE",
        "article_id": "14460039",
        "title": "FORECasT-BE processed base editor screen data",
        "target_folder": str(DATA_ROOT / Path("manual_downloads/forecast_be")),
        "recommended": "yes",
        "notes": "Single processed TSV; raw ENA is much larger.",
    },
    {
        "dataset_id": "CGBE_SMART_PROCESSED",
        "article_id": "12275654",
        "title": "CGBE / BE-SMART processed editing efficiency data",
        "target_folder": str(DATA_ROOT / Path("manual_downloads/cgbe_smart_processed")),
        "recommended": "optional",
        "notes": "Processed CSVs for C-to-G/C-to-T extension; optional for ABE/CBE core benchmark.",
    },
]


SPRINGER_FILES = [
    # BEDICT2.0, already local but included for transfer/re-download.
    {
        "dataset_id": "BEDICT2_SUPP",
        "title": "BEDICT2.0 supplementary information PDF",
        "url": "https://static-content.springer.com/esm/art%3A10.1186%2Fs13059-025-03586-7/MediaObjects/13059_2025_3586_MOESM1_ESM.pdf",
        "target_folder": str(MODELS_ROOT / Path("BEDICT-V2/resources/paper_supplements")),
        "recommended": "already_local",
    },
    {
        "dataset_id": "BEDICT2_SUPP",
        "title": "BEDICT2.0 Additional file 2",
        "url": "https://static-content.springer.com/esm/art%3A10.1186%2Fs13059-025-03586-7/MediaObjects/13059_2025_3586_MOESM2_ESM.xlsx",
        "target_folder": str(MODELS_ROOT / Path("BEDICT-V2/resources/paper_supplements")),
        "recommended": "already_local",
    },
    {
        "dataset_id": "BEDICT2_SUPP",
        "title": "BEDICT2.0 Additional file 3",
        "url": "https://static-content.springer.com/esm/art%3A10.1186%2Fs13059-025-03586-7/MediaObjects/13059_2025_3586_MOESM3_ESM.xlsx",
        "target_folder": str(MODELS_ROOT / Path("BEDICT-V2/resources/paper_supplements")),
        "recommended": "already_local",
    },
    {
        "dataset_id": "BEDICT2_SUPP",
        "title": "BEDICT2.0 Additional file 4 / source data",
        "url": "https://static-content.springer.com/esm/art%3A10.1186%2Fs13059-025-03586-7/MediaObjects/13059_2025_3586_MOESM4_ESM.xlsx",
        "target_folder": str(MODELS_ROOT / Path("BEDICT-V2/resources/paper_supplements")),
        "recommended": "already_local",
    },
    # BE-DICT / Marquart 2021.
    {
        "dataset_id": "BEDICT_NATCOMM_SUPP",
        "title": "BE-DICT supplementary information",
        "url": "https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-021-25375-z/MediaObjects/41467_2021_25375_MOESM1_ESM.pdf",
        "target_folder": str(DATA_ROOT / Path("manual_downloads/be_dict_supplement")),
        "recommended": "supporting",
    },
    {
        "dataset_id": "BEDICT_NATCOMM_SUPP",
        "title": "BE-DICT Supplementary Data 1",
        "url": "https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-021-25375-z/MediaObjects/41467_2021_25375_MOESM2_ESM.xlsx",
        "target_folder": str(DATA_ROOT / Path("manual_downloads/be_dict_supplement")),
        "recommended": "yes",
    },
    {
        "dataset_id": "BEDICT_NATCOMM_SUPP",
        "title": "BE-DICT Supplementary Data 2",
        "url": "https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-021-25375-z/MediaObjects/41467_2021_25375_MOESM3_ESM.xlsx",
        "target_folder": str(DATA_ROOT / Path("manual_downloads/be_dict_supplement")),
        "recommended": "yes",
    },
    {
        "dataset_id": "BEDICT_NATCOMM_SUPP",
        "title": "BE-DICT Supplementary Data 3",
        "url": "https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-021-25375-z/MediaObjects/41467_2021_25375_MOESM4_ESM.xlsx",
        "target_folder": str(DATA_ROOT / Path("manual_downloads/be_dict_supplement")),
        "recommended": "yes",
    },
    {
        "dataset_id": "BEDICT_NATCOMM_SUPP",
        "title": "BE-DICT source data",
        "url": "https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-021-25375-z/MediaObjects/41467_2021_25375_MOESM8_ESM.xlsx",
        "target_folder": str(DATA_ROOT / Path("manual_downloads/be_dict_supplement")),
        "recommended": "yes",
    },
    # CRISPRon-BE paper supplements.
    {
        "dataset_id": "CRISPRONBE_SUPP",
        "title": "CRISPRon-BE Supplementary Data 1",
        "url": "https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-025-65200-5/MediaObjects/41467_2025_65200_MOESM3_ESM.xlsx",
        "target_folder": str(MODELS_ROOT / Path("CRISPRon-BE/repo/data_for_model_development_V1.0")),
        "recommended": "already_local_equivalent",
    },
    {
        "dataset_id": "CRISPRONBE_SUPP",
        "title": "CRISPRon-BE Supplementary Data 2",
        "url": "https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-025-65200-5/MediaObjects/41467_2025_65200_MOESM4_ESM.xlsx",
        "target_folder": str(DATA_ROOT / Path("manual_downloads/crispron_be_supplement")),
        "recommended": "supporting",
    },
    {
        "dataset_id": "CRISPRONBE_SUPP",
        "title": "CRISPRon-BE Supplementary Data 3",
        "url": "https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-025-65200-5/MediaObjects/41467_2025_65200_MOESM5_ESM.xlsx",
        "target_folder": str(DATA_ROOT / Path("manual_downloads/crispron_be_supplement")),
        "recommended": "supporting",
    },
    {
        "dataset_id": "CRISPRONBE_SUPP",
        "title": "CRISPRon-BE source data",
        "url": "https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-025-65200-5/MediaObjects/41467_2025_65200_MOESM8_ESM.xlsx",
        "target_folder": str(DATA_ROOT / Path("manual_downloads/crispron_be_supplement")),
        "recommended": "supporting",
    },
    # BEguider.
    {
        "dataset_id": "BEGUIDER_NEAR_PAMLESS",
        "title": "BEguider Supplementary Data",
        "url": "https://static-content.springer.com/esm/art%3A10.1038%2Fs42003-024-07078-5/MediaObjects/42003_2024_7078_MOESM3_ESM.xlsx",
        "target_folder": str(DATA_ROOT / Path("manual_downloads/beguider")),
        "recommended": "yes",
    },
    # CGBE / BE-SMART supplements.
    {
        "dataset_id": "CGBE_SMART_SUPP",
        "title": "CGBE/BE-SMART Supplementary Data 1",
        "url": "https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-021-25217-y/MediaObjects/41467_2021_25217_MOESM5_ESM.xlsx",
        "target_folder": str(DATA_ROOT / Path("manual_downloads/cgbe_smart_supplement")),
        "recommended": "optional",
    },
    {
        "dataset_id": "CGBE_SMART_SUPP",
        "title": "CGBE/BE-SMART Supplementary Data 2",
        "url": "https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-021-25217-y/MediaObjects/41467_2021_25217_MOESM6_ESM.xlsx",
        "target_folder": str(DATA_ROOT / Path("manual_downloads/cgbe_smart_supplement")),
        "recommended": "optional",
    },
    # BE_Endo supplements.
    {
        "dataset_id": "BE_ENDO_SUPP",
        "title": "BE_Endo Supplementary Data",
        "url": "https://static-content.springer.com/esm/art%3A10.1038%2Fs41421-023-00624-1/MediaObjects/41421_2023_624_MOESM2_ESM.xlsx",
        "target_folder": str(DATA_ROOT / Path("manual_downloads/be_endo_supplement")),
        "recommended": "yes",
    },
    {
        "dataset_id": "BE_ENDO_SUPP",
        "title": "BE_Endo source data",
        "url": "https://static-content.springer.com/esm/art%3A10.1038%2Fs41421-023-00624-1/MediaObjects/41421_2023_624_MOESM3_ESM.xlsx",
        "target_folder": str(DATA_ROOT / Path("manual_downloads/be_endo_supplement")),
        "recommended": "yes",
    },
]


GITHUB_FILES = [
    {
        "dataset_id": "BEGUIDER_GITHUB_DATA",
        "title": "BEguider GitHub data directory",
        "url": "https://github.com/Wangxiaoyue-lab/BEguider/tree/main/data",
        "target_folder": str(DATA_ROOT / Path("manual_downloads/beguider/github_data")),
        "size_bytes": "",
        "recommended": "yes",
    },
    {
        "dataset_id": "BEGUIDER_GITHUB_DATA",
        "title": "BEguider a_alt_anno.csv",
        "url": "https://raw.githubusercontent.com/Wangxiaoyue-lab/BEguider/main/data/a_alt_anno.csv",
        "target_folder": str(DATA_ROOT / Path("manual_downloads/beguider/github_data")),
        "size_bytes": "1860493",
        "recommended": "yes",
    },
    {
        "dataset_id": "BEGUIDER_GITHUB_DATA",
        "title": "BEguider a_ref_anno.csv",
        "url": "https://raw.githubusercontent.com/Wangxiaoyue-lab/BEguider/main/data/a_ref_anno.csv",
        "target_folder": str(DATA_ROOT / Path("manual_downloads/beguider/github_data")),
        "size_bytes": "5759092",
        "recommended": "yes",
    },
    {
        "dataset_id": "BEGUIDER_GITHUB_DATA",
        "title": "BEguider c_alt_anno.csv",
        "url": "https://raw.githubusercontent.com/Wangxiaoyue-lab/BEguider/main/data/c_alt_anno.csv",
        "target_folder": str(DATA_ROOT / Path("manual_downloads/beguider/github_data")),
        "size_bytes": "5679519",
        "recommended": "yes",
    },
    {
        "dataset_id": "BEGUIDER_GITHUB_DATA",
        "title": "BEguider c_ref_anno.csv",
        "url": "https://raw.githubusercontent.com/Wangxiaoyue-lab/BEguider/main/data/c_ref_anno.csv",
        "target_folder": str(DATA_ROOT / Path("manual_downloads/beguider/github_data")),
        "size_bytes": "1758812",
        "recommended": "yes",
    },
    {
        "dataset_id": "CRISPRONBE_GITHUB_DATA",
        "title": "CRISPRon-BE model development data directory",
        "url": "https://github.com/RTH-tools/crispron-BE/tree/main/data_for_model_development_V1.0",
        "target_folder": str(MODELS_ROOT / Path("CRISPRon-BE/repo/data_for_model_development_V1.0")),
        "size_bytes": "",
        "recommended": "already_local",
    },
    {
        "dataset_id": "BE_ENDO_GITHUB_MODELS",
        "title": "BE_Endo model_data_endo directory",
        "url": "https://github.com/zhaodalv/BE_Endo/tree/master/model_data_endo",
        "target_folder": str(MODELS_ROOT / Path("BE_Endo/repo/model_data_endo")),
        "size_bytes": "",
        "recommended": "model_resource",
    },
]


API_LINKS = [
    {
        "dataset_id": "BEDATAHIVE_API",
        "title": "BE-dataHIVE API documentation",
        "url": "https://be-datahive.com/documentation.html",
        "target_folder": str(DATA_ROOT / Path("manual_downloads/be_datahive")),
        "recommended": "yes",
        "notes": "API docs; use for studies/efficiency/bystander endpoint definitions.",
    },
    {
        "dataset_id": "BEDATAHIVE_API",
        "title": "BE-dataHIVE studies endpoint",
        "url": "https://be-server.herokuapp.com/studies",
        "target_folder": str(DATA_ROOT / Path("manual_downloads/be_datahive")),
        "recommended": "yes",
        "notes": "Study-level metadata and original data links.",
    },
    {
        "dataset_id": "BEDATAHIVE_API",
        "title": "BE-dataHIVE efficiency endpoint",
        "url": "https://be-server.herokuapp.com/efficiency",
        "target_folder": str(DATA_ROOT / Path("manual_downloads/be_datahive")),
        "recommended": "yes",
        "notes": "Batch with limit/offset; do not request entire database blindly.",
    },
    {
        "dataset_id": "BEDATAHIVE_API",
        "title": "BE-dataHIVE bystander endpoint",
        "url": "https://be-server.herokuapp.com/bystander",
        "target_folder": str(DATA_ROOT / Path("manual_downloads/be_datahive")),
        "recommended": "yes",
        "notes": "Batch with limit/offset; outcome/bystander features.",
    },
]


RAW_ACCESSIONS = [
    {
        "dataset_id": "BEDICT2_RAW_PRJNA1107439",
        "accession": "PRJNA1107439",
        "title": "BEDICT2.0 raw sequencing",
        "target_folder": str(DATA_ROOT / Path("manual_downloads/raw_sra_large/bedict2_prjna1107439")),
        "priority": "raw_audit_later",
    },
    {
        "dataset_id": "CRISPRONBE_RAW_PRJNA1321836",
        "accession": "PRJNA1321836",
        "title": "CRISPRon-BE raw sequencing",
        "target_folder": str(DATA_ROOT / Path("manual_downloads/raw_sra_large/crispron_be_prjna1321836")),
        "priority": "raw_audit_later",
    },
    {
        "dataset_id": "BE_ENDO_RAW_PRJNA885970",
        "accession": "PRJNA885970",
        "title": "BE_Endo raw sequencing",
        "target_folder": str(DATA_ROOT / Path("manual_downloads/raw_sra_large/be_endo_prjna885970")),
        "priority": "raw_audit_later",
    },
    {
        "dataset_id": "BEGUIDER_RAW_PRJNA1046210",
        "accession": "PRJNA1046210",
        "title": "BEguider raw sequencing",
        "target_folder": str(DATA_ROOT / Path("manual_downloads/raw_sra_large/beguider_prjna1046210")),
        "priority": "raw_audit_later",
    },
    {
        "dataset_id": "BEDICT_RAW_PRJNA735610",
        "accession": "PRJNA735610",
        "title": "BE-DICT 2021 raw sequencing",
        "target_folder": str(DATA_ROOT / Path("manual_downloads/raw_sra_large/be_dict_prjna735610")),
        "priority": "raw_audit_later",
    },
    {
        "dataset_id": "FORECASTBE_RAW_PRJEB12405",
        "accession": "PRJEB12405",
        "title": "FORECasT-BE raw sequencing",
        "target_folder": str(DATA_ROOT / Path("manual_downloads/raw_sra_large/forecast_be_prjeb12405")),
        "priority": "avoid_initial_tb_scale",
    },
    {
        "dataset_id": "CGBE_SMART_RAW_PRJNA749814",
        "accession": "PRJNA749814",
        "title": "CGBE/BE-SMART raw sequencing",
        "target_folder": str(DATA_ROOT / Path("manual_downloads/raw_sra_large/cgbe_smart_prjna749814")),
        "priority": "avoid_initial_tb_scale",
    },
]


def fetch_json(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=60) as handle:
        return json.load(handle)


def head_size(url: str) -> tuple[str, str]:
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=60) as handle:
            return handle.headers.get("Content-Length", ""), handle.headers.get("Content-Type", "")
    except Exception as exc:
        return "", f"HEAD_ERROR: {exc}"


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_figshare_manifest() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    article_rows: list[dict[str, Any]] = []
    file_rows: list[dict[str, Any]] = []
    for item in FIGSHARE_ARTICLES:
        api_url = f"https://api.figshare.com/v2/articles/{item['article_id']}"
        data = fetch_json(api_url)
        files = data.get("files") or []
        total_size = sum(int(f.get("size") or 0) for f in files)
        article_rows.append(
            {
                **item,
                "article_url": data.get("url_public_html") or f"https://figshare.com/articles/dataset/{item['article_id']}",
                "api_url": api_url,
                "file_count": len(files),
                "total_size_bytes": total_size,
                "total_size_gb_decimal": round(total_size / 1e9, 3),
            }
        )
        for f in files:
            file_rows.append(
                {
                    "dataset_id": item["dataset_id"],
                    "article_id": item["article_id"],
                    "article_title": data.get("title") or item["title"],
                    "file_name": f.get("name"),
                    "size_bytes": f.get("size"),
                    "download_url": f.get("download_url"),
                    "target_folder": item["target_folder"],
                    "recommended": item["recommended"],
                }
            )
    return article_rows, file_rows


def build_springer_manifest() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in SPRINGER_FILES:
        size, ctype = head_size(item["url"])
        rows.append({**item, "size_bytes": size, "content_type": ctype})
    return rows


def build_raw_ena_manifests() -> list[dict[str, Any]]:
    RAW_OUT.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []
    fields = (
        "run_accession,experiment_accession,study_accession,scientific_name,"
        "instrument_platform,fastq_ftp,fastq_bytes,submitted_ftp,submitted_bytes"
    )
    for item in RAW_ACCESSIONS:
        acc = item["accession"]
        query = urllib.parse.urlencode(
            {
                "accession": acc,
                "result": "read_run",
                "fields": fields,
                "format": "tsv",
                "download": "false",
                "limit": "0",
            }
        )
        api_url = f"https://www.ebi.ac.uk/ena/portal/api/filereport?{query}"
        out_tsv = RAW_OUT / f"ena_{acc}_read_run.tsv"
        rows = 0
        fastq_total = 0
        submitted_total = 0
        status = "ok"
        try:
            with urllib.request.urlopen(api_url, timeout=180) as handle:
                text = handle.read().decode("utf-8", errors="replace")
            out_tsv.write_text(text, encoding="utf-8")
            lines = text.strip().splitlines()
            if len(lines) > 1:
                for row in csv.DictReader(lines, delimiter="\t"):
                    rows += 1
                    for part in (row.get("fastq_bytes") or "").split(";"):
                        try:
                            fastq_total += int(part)
                        except ValueError:
                            pass
                    for part in (row.get("submitted_bytes") or "").split(";"):
                        try:
                            submitted_total += int(part)
                        except ValueError:
                            pass
        except Exception as exc:
            status = f"error: {exc}"
            out_tsv.write_text("", encoding="utf-8")

        summary_rows.append(
            {
                **item,
                "ena_browser_url": f"https://www.ebi.ac.uk/ena/browser/view/{acc}",
                "ncbi_bioproject_url": f"https://www.ncbi.nlm.nih.gov/bioproject/{acc}",
                "ena_portal_api_url": api_url,
                "local_run_manifest": str(out_tsv),
                "run_count": rows,
                "fastq_total_bytes": fastq_total,
                "fastq_total_gb_decimal": round(fastq_total / 1e9, 2),
                "submitted_total_bytes": submitted_total,
                "submitted_total_gb_decimal": round(submitted_total / 1e9, 2),
                "status": status,
            }
        )
    return summary_rows


def build_master_manifest(
    fig_articles: list[dict[str, Any]],
    springer_rows: list[dict[str, Any]],
    raw_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for r in fig_articles:
        rows.append(
            {
                "dataset_id": r["dataset_id"],
                "category": "processed_figshare",
                "title": r["title"],
                "priority": r["recommended"],
                "size_bytes": r["total_size_bytes"],
                "size_gb_decimal": r["total_size_gb_decimal"],
                "download_page_or_api": r["article_url"],
                "direct_file_manifest": str(OUT / "figshare_file_manifest.csv"),
                "target_folder": r["target_folder"],
                "notes": r["notes"],
            }
        )
    for r in springer_rows:
        rows.append(
            {
                "dataset_id": r["dataset_id"],
                "category": "supplementary_file",
                "title": r["title"],
                "priority": r["recommended"],
                "size_bytes": r["size_bytes"],
                "size_gb_decimal": round(int(r["size_bytes"] or 0) / 1e9, 3) if str(r["size_bytes"]).isdigit() else "",
                "download_page_or_api": r["url"],
                "direct_file_manifest": str(OUT / "springer_supplementary_files.csv"),
                "target_folder": r["target_folder"],
                "notes": r.get("content_type", ""),
            }
        )
    for r in GITHUB_FILES:
        rows.append(
            {
                "dataset_id": r["dataset_id"],
                "category": "github_file_or_directory",
                "title": r["title"],
                "priority": r["recommended"],
                "size_bytes": r["size_bytes"],
                "size_gb_decimal": round(int(r["size_bytes"] or 0) / 1e9, 3) if str(r["size_bytes"]).isdigit() else "",
                "download_page_or_api": r["url"],
                "direct_file_manifest": str(OUT / "github_data_links.csv"),
                "target_folder": r["target_folder"],
                "notes": "",
            }
        )
    for r in API_LINKS:
        rows.append(
            {
                "dataset_id": r["dataset_id"],
                "category": "api_endpoint",
                "title": r["title"],
                "priority": r["recommended"],
                "size_bytes": "",
                "size_gb_decimal": "",
                "download_page_or_api": r["url"],
                "direct_file_manifest": str(OUT / "api_data_links.csv"),
                "target_folder": r["target_folder"],
                "notes": r["notes"],
            }
        )
    for r in raw_rows:
        rows.append(
            {
                "dataset_id": r["dataset_id"],
                "category": "raw_large_sequencing",
                "title": r["title"],
                "priority": r["priority"],
                "size_bytes": r["fastq_total_bytes"],
                "size_gb_decimal": r["fastq_total_gb_decimal"],
                "download_page_or_api": r["ena_browser_url"],
                "direct_file_manifest": r["local_run_manifest"],
                "target_folder": r["target_folder"],
                "notes": f"NCBI: {r['ncbi_bioproject_url']}; ENA API: {r['ena_portal_api_url']}",
            }
        )
    return rows


def write_markdown(
    master: list[dict[str, Any]],
    fig_articles: list[dict[str, Any]],
    raw_rows: list[dict[str, Any]],
) -> None:
    lines: list[str] = []
    lines.append("# Data download links for base-editor benchmark")
    lines.append("")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    lines.append("This file lists download links only. No dataset payloads were downloaded by the script.")
    lines.append("")
    lines.append("## Where to put downloaded files")
    lines.append("")
    lines.append("Use the `target_folder` column in:")
    lines.append("")
    lines.append("```text")
    lines.append(str(DATA_ROOT / Path("download_manifests/data_download_master_manifest.csv")))
    lines.append("```")
    lines.append("")
    lines.append("Manual-download root:")
    lines.append("")
    lines.append("```text")
    lines.append(str(DATA_ROOT / Path("manual_downloads")))
    lines.append("```")
    lines.append("")
    lines.append("## Recommended first downloads")
    lines.append("")
    lines.append("| Priority | Dataset | Approx size | Link/manifest | Target folder |")
    lines.append("|---|---|---:|---|---|")
    for row in master:
        if row["category"] in {"processed_figshare", "supplementary_file", "github_file_or_directory", "api_endpoint"} and row["priority"] in {"yes", "optional"}:
            size = row["size_gb_decimal"]
            if size == "":
                size_txt = ""
            else:
                size_txt = f"{float(size):.3f} GB"
            lines.append(
                f"| {row['priority']} | {row['title']} | {size_txt} | {row['download_page_or_api']} | `{row['target_folder']}` |"
            )
    lines.append("")
    lines.append("## Figshare processed datasets")
    lines.append("")
    lines.append("Direct file-level URLs are in:")
    lines.append("")
    lines.append("```text")
    lines.append(str(OUT / "figshare_file_manifest.csv"))
    lines.append("```")
    lines.append("")
    lines.append("| Dataset | Files | Total size | Article page |")
    lines.append("|---|---:|---:|---|")
    for row in fig_articles:
        lines.append(
            f"| {row['title']} | {row['file_count']} | {row['total_size_gb_decimal']:.3f} GB | {row['article_url']} |"
        )
    lines.append("")
    lines.append("## Raw / large sequencing")
    lines.append("")
    lines.append("These are not recommended as first downloads. The run-level ENA manifests are small TSV files created under:")
    lines.append("")
    lines.append("```text")
    lines.append(str(RAW_OUT))
    lines.append("```")
    lines.append("")
    lines.append("| Dataset | Accession | Runs | ENA FASTQ size | ENA browser | Run manifest |")
    lines.append("|---|---|---:|---:|---|---|")
    for row in raw_rows:
        lines.append(
            f"| {row['title']} | {row['accession']} | {row['run_count']} | {row['fastq_total_gb_decimal']:.2f} GB | {row['ena_browser_url']} | `{row['local_run_manifest']}` |"
        )
    lines.append("")
    lines.append("## Manifest files")
    lines.append("")
    for name in [
        "data_download_master_manifest.csv",
        "figshare_article_manifest.csv",
        "figshare_file_manifest.csv",
        "springer_supplementary_files.csv",
        "github_data_links.csv",
        "api_data_links.csv",
        "raw_large_ena_summary.csv",
    ]:
        lines.append(f"- `{OUT / name}`")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- For figshare, use `figshare_file_manifest.csv` if you want direct per-file URLs.")
    lines.append("- For raw sequencing, use ENA browser/API manifests first; avoid downloading TB-scale datasets before the benchmark panels are fixed.")
    lines.append("- BEDICT2.0 and CRISPRon-BE processed files are already present locally, but links are retained for transfer/recovery.")
    (OUT / "DATA_DOWNLOAD_LINKS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig_articles, fig_files = build_figshare_manifest()
    springer_rows = build_springer_manifest()
    raw_rows = build_raw_ena_manifests()
    master = build_master_manifest(fig_articles, springer_rows, raw_rows)

    write_csv(OUT / "figshare_article_manifest.csv", fig_articles)
    write_csv(OUT / "figshare_file_manifest.csv", fig_files)
    write_csv(OUT / "springer_supplementary_files.csv", springer_rows)
    write_csv(OUT / "github_data_links.csv", GITHUB_FILES)
    write_csv(OUT / "api_data_links.csv", API_LINKS)
    write_csv(OUT / "raw_large_ena_summary.csv", raw_rows)
    write_csv(OUT / "data_download_master_manifest.csv", master)
    write_markdown(master, fig_articles, raw_rows)

    print(f"Wrote manifests under {OUT}")
    print(f"Master rows: {len(master)}")
    print(f"Figshare file rows: {len(fig_files)}")
    print(f"Raw accession rows: {len(raw_rows)}")


if __name__ == "__main__":
    main()
