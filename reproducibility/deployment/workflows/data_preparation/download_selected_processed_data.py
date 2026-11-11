"""
Download selected processed/supplementary data for the base-editor benchmark.

Scope:
  - BE-Hive processed efficiency
  - BE-Hive processed bystander
  - FORECasT-BE processed
  - BE-DICT supplementary/source data
  - BEguider supplementary + GitHub processed data
  - BE_Endo supplementary/source data
  - Optional CGBE processed figshare data

This script intentionally excludes raw FASTQ/SRA/TB-scale sequencing datasets.
It reads download manifests created by make_data_download_manifests.py and writes
a local download status manifest.
"""

from __future__ import annotations

import csv
import hashlib
import os
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

from runtime_paths import DATA_ROOT, REPORT_ROOT


MANIFEST_DIR = DATA_ROOT / "download_manifests"
REPORT_DIR = REPORT_ROOT

FIGSHARE_DATASETS = {
    "BEHIVE_EFF_FIGSHARE",
    "BEHIVE_BYST_FIGSHARE",
    "FORECASTBE_FIGSHARE",
    "CGBE_SMART_PROCESSED",
}

SPRINGER_DATASETS = {
    "BEDICT_NATCOMM_SUPP",
    "BEGUIDER_NEAR_PAMLESS",
    "BE_ENDO_SUPP",
}

GITHUB_DATASETS = {
    "BEGUIDER_GITHUB_DATA",
}


@dataclass
class DownloadItem:
    dataset_id: str
    title: str
    url: str
    target_folder: Path
    file_name: str
    expected_size: int | None
    source_manifest: str

    @property
    def target_path(self) -> Path:
        return self.target_folder / self.file_name


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "timestamp",
        "dataset_id",
        "title",
        "file_name",
        "url",
        "target_path",
        "expected_size",
        "actual_size",
        "sha256",
        "status",
        "message",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def safe_file_name_from_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    name = Path(urllib.parse.unquote(parsed.path)).name
    if not name:
        name = "downloaded_file"
    return name


def collect_download_items() -> list[DownloadItem]:
    items: list[DownloadItem] = []

    fig_path = MANIFEST_DIR / "figshare_file_manifest.csv"
    for row in read_csv(fig_path):
        if row["dataset_id"] not in FIGSHARE_DATASETS:
            continue
        items.append(
            DownloadItem(
                dataset_id=row["dataset_id"],
                title=row.get("article_title") or row["dataset_id"],
                url=row["download_url"],
                target_folder=Path(row["target_folder"]),
                file_name=row["file_name"],
                expected_size=int(row["size_bytes"]) if row.get("size_bytes") else None,
                source_manifest=str(fig_path),
            )
        )

    springer_path = MANIFEST_DIR / "springer_supplementary_files.csv"
    for row in read_csv(springer_path):
        if row["dataset_id"] not in SPRINGER_DATASETS:
            continue
        items.append(
            DownloadItem(
                dataset_id=row["dataset_id"],
                title=row["title"],
                url=row["url"],
                target_folder=Path(row["target_folder"]),
                file_name=safe_file_name_from_url(row["url"]),
                expected_size=int(row["size_bytes"]) if row.get("size_bytes", "").isdigit() else None,
                source_manifest=str(springer_path),
            )
        )

    github_path = MANIFEST_DIR / "github_data_links.csv"
    for row in read_csv(github_path):
        if row["dataset_id"] not in GITHUB_DATASETS:
            continue
        url = row["url"]
        if not url.startswith("https://raw.githubusercontent.com/"):
            continue
        items.append(
            DownloadItem(
                dataset_id=row["dataset_id"],
                title=row["title"],
                url=url,
                target_folder=Path(row["target_folder"]),
                file_name=safe_file_name_from_url(url),
                expected_size=int(row["size_bytes"]) if row.get("size_bytes", "").isdigit() else None,
                source_manifest=str(github_path),
            )
        )

    # Deduplicate exact URLs while keeping order.
    seen: set[str] = set()
    out: list[DownloadItem] = []
    for item in items:
        if item.url in seen:
            continue
        seen.add(item.url)
        out.append(item)
    return out


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download_one(item: DownloadItem, retries: int = 3) -> dict[str, object]:
    item.target_folder.mkdir(parents=True, exist_ok=True)
    target = item.target_path
    part = target.with_name(target.name + ".part")
    ts = datetime.now().isoformat(timespec="seconds")

    if target.exists():
        actual = target.stat().st_size
        if item.expected_size is None or actual == item.expected_size:
            return {
                "timestamp": ts,
                "dataset_id": item.dataset_id,
                "title": item.title,
                "file_name": item.file_name,
                "url": item.url,
                "target_path": str(target),
                "expected_size": item.expected_size or "",
                "actual_size": actual,
                "sha256": sha256_file(target),
                "status": "skipped_existing_complete",
                "message": "",
            }

    last_message = ""
    for attempt in range(1, retries + 1):
        try:
            resume_from = part.stat().st_size if part.exists() else 0
            headers = {"User-Agent": "base-editor-benchmark-downloader/0.1"}
            mode = "ab"
            if resume_from:
                headers["Range"] = f"bytes={resume_from}-"

            req = urllib.request.Request(item.url, headers=headers)
            with urllib.request.urlopen(req, timeout=120) as response:
                # If the server ignores Range, restart the partial file.
                if resume_from and response.status == 200:
                    resume_from = 0
                    mode = "wb"
                with part.open(mode + "") as handle:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        handle.write(chunk)

            actual_part = part.stat().st_size
            if item.expected_size is not None and actual_part != item.expected_size:
                last_message = f"size mismatch after attempt {attempt}: expected {item.expected_size}, got {actual_part}"
                time.sleep(min(5 * attempt, 20))
                continue

            if target.exists():
                # Keep an unexpected existing file by renaming it once.
                backup = target.with_name(target.name + f".old_{int(time.time())}")
                target.rename(backup)
            part.rename(target)
            actual = target.stat().st_size
            return {
                "timestamp": ts,
                "dataset_id": item.dataset_id,
                "title": item.title,
                "file_name": item.file_name,
                "url": item.url,
                "target_path": str(target),
                "expected_size": item.expected_size or "",
                "actual_size": actual,
                "sha256": sha256_file(target),
                "status": "downloaded",
                "message": "",
            }
        except Exception as exc:
            last_message = f"attempt {attempt} failed: {exc}"
            time.sleep(min(5 * attempt, 20))

    actual = part.stat().st_size if part.exists() else (target.stat().st_size if target.exists() else 0)
    return {
        "timestamp": ts,
        "dataset_id": item.dataset_id,
        "title": item.title,
        "file_name": item.file_name,
        "url": item.url,
        "target_path": str(target),
        "expected_size": item.expected_size or "",
        "actual_size": actual,
        "sha256": sha256_file(target) if target.exists() else "",
        "status": "failed",
        "message": last_message,
    }


def write_summary_markdown(rows: list[dict[str, object]], path: Path) -> None:
    by_status: dict[str, int] = {}
    by_dataset: dict[str, dict[str, int]] = {}
    for row in rows:
        status = str(row["status"])
        dataset = str(row["dataset_id"])
        by_status[status] = by_status.get(status, 0) + 1
        by_dataset.setdefault(dataset, {})
        by_dataset[dataset][status] = by_dataset[dataset].get(status, 0) + 1

    lines: list[str] = []
    lines.append("# Selected processed/supplementary data download report")
    lines.append("")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    lines.append("Raw FASTQ/SRA/TB-scale sequencing data were intentionally excluded.")
    lines.append("")
    lines.append("## Status counts")
    lines.append("")
    lines.append("| Status | Count |")
    lines.append("|---|---:|")
    for status, count in sorted(by_status.items()):
        lines.append(f"| {status} | {count} |")
    lines.append("")
    lines.append("## Dataset counts")
    lines.append("")
    lines.append("| Dataset | Downloaded | Existing complete | Failed |")
    lines.append("|---|---:|---:|---:|")
    for dataset, counts in sorted(by_dataset.items()):
        lines.append(
            f"| {dataset} | {counts.get('downloaded', 0)} | "
            f"{counts.get('skipped_existing_complete', 0)} | {counts.get('failed', 0)} |"
        )
    lines.append("")
    failed = [r for r in rows if r["status"] == "failed"]
    if failed:
        lines.append("## Failed files")
        lines.append("")
        lines.append("| Dataset | File | URL | Message |")
        lines.append("|---|---|---|---|")
        for row in failed:
            msg = str(row["message"]).replace("|", "\\|")
            lines.append(f"| {row['dataset_id']} | {row['file_name']} | {row['url']} | {msg} |")
        lines.append("")
    lines.append("## Detailed CSV")
    lines.append("")
    lines.append("```text")
    lines.append(str(REPORT_DIR / "selected_processed_data_download_status.csv"))
    lines.append("```")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    items = collect_download_items()
    print(f"Selected files: {len(items)}")
    expected_total = sum(item.expected_size or 0 for item in items)
    print(f"Expected total bytes: {expected_total} ({expected_total / 1e9:.3f} GB)")

    rows: list[dict[str, object]] = []
    for idx, item in enumerate(items, 1):
        print(f"[{idx}/{len(items)}] {item.dataset_id} :: {item.file_name}")
        row = download_one(item)
        print(f"  -> {row['status']} {row['actual_size']} bytes")
        if row["status"] == "failed":
            print(f"     {row['message']}")
        rows.append(row)
        write_csv(REPORT_DIR / "selected_processed_data_download_status.csv", rows)

    write_summary_markdown(rows, REPORT_DIR / "SELECTED_PROCESSED_DATA_DOWNLOAD_REPORT.md")
    failures = sum(1 for row in rows if row["status"] == "failed")
    if failures:
        raise SystemExit(f"{failures} download(s) failed. See report.")


if __name__ == "__main__":
    main()
