"""Download, verify and extract one or more GitHub release assets."""
import argparse
import hashlib
import json
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from verify_bundle import ROOT, sha256


def extract(archive, spec):
    expected = {member["path"]: member for member in spec["members"]}
    with zipfile.ZipFile(archive) as zipped:
        names = zipped.namelist()
        if len(names) != len(expected) or set(names) != set(expected):
            raise ValueError(f"Unexpected archive members in {archive.name}")
        for item in zipped.infolist():
            meta = expected[item.filename]
            target = (ROOT / item.filename).resolve()
            if not target.is_relative_to(ROOT) or item.file_size != meta["bytes"]:
                raise ValueError(f"Invalid archive member: {item.filename}")
            if target.exists():
                if sha256(target) != meta["sha256"]:
                    raise ValueError(f"Existing file differs: {target}")
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            handle, temp = tempfile.mkstemp(dir=target.parent, prefix=".asset-")
            temp = Path(temp)
            try:
                import os
                with os.fdopen(handle, "wb") as out, zipped.open(item) as entry:
                    checksum = hashlib.sha256()
                    for block in iter(lambda: entry.read(4 * 1024 * 1024), b""):
                        checksum.update(block)
                        out.write(block)
                if checksum.hexdigest() != meta["sha256"]:
                    raise ValueError(f"Member checksum mismatch: {item.filename}")
                temp.replace(target)
            finally:
                temp.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", action="append", help="Select an asset by name. Defaults to all assets.")
    parser.add_argument("--from-dir", type=Path, help="Use already downloaded ZIPs from this directory")
    parser.add_argument("--download-dir", type=Path, help="Keep downloaded ZIPs in this directory")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "assets.json").read_text(encoding="utf-8"))
    assets = {row["name"]: row for row in manifest["assets"]}
    with tempfile.TemporaryDirectory(prefix="be-benchmark-") as temp:
        download_dir = args.download_dir or Path(temp)
        download_dir.mkdir(parents=True, exist_ok=True)
        for name in args.asset or assets:
            if name not in assets:
                parser.error(f"Unknown asset {name}")
            asset = assets[name]
            archive = (args.from_dir or download_dir) / name
            if not args.from_dir and not archive.is_file():
                incomplete = archive.with_suffix(".zip.part")
                request = urllib.request.Request(asset["url"], headers={"User-Agent":"be-benchmark-reproducibility"})
                try:
                    with urllib.request.urlopen(request, timeout=120) as response, incomplete.open("wb") as file:
                        shutil.copyfileobj(response, file)
                    if incomplete.stat().st_size != asset["bytes"] or sha256(incomplete) != asset["sha256"]:
                        raise ValueError(f"Downloaded asset failed verification: {name}")
                    incomplete.replace(archive)
                finally:
                    incomplete.unlink(missing_ok=True)
            if archive.stat().st_size != asset["bytes"] or sha256(archive) != asset["sha256"]:
                raise ValueError(f"Asset checksum mismatch: {name}")
            extract(archive, asset)
            print(f"PASS: {name} ({len(asset['members'])} files)", flush=True)


if __name__ == "__main__":
    main()

