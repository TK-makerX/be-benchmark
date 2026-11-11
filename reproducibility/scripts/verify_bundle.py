"""Verify the deposited file lengths and SHA-256 values."""
import argparse
import csv
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--git-only", action="store_true", help="Verify the tracked files in a fresh checkout")
    args = parser.parse_args()
    count = size = 0
    errors = []
    with (ROOT / "payload_manifest.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if args.git_only and row["storage"] != "git":
                continue
            path = ROOT / row["path"]
            if not path.is_file() or path.stat().st_size != int(row["bytes"]) or sha256(path) != row["sha256"]:
                errors.append(row["path"])
                continue
            count += 1
            size += int(row["bytes"])
    if errors:
        raise SystemExit("Missing files or checksum mismatches:\n" + "\n".join(errors))
    print(f"PASS: {count:,} files, {size:,} bytes")


if __name__ == "__main__":
    main()

