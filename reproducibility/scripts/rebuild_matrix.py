"""Rebuild the V5 matrix from the archived upstream inputs in a separate directory."""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from verify_bundle import ROOT

def records(path):
    with path.open(encoding="utf-8-sig",newline="") as handle:
        yield from csv.DictReader(handle)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir",type=Path,required=True)
    args=parser.parse_args()
    project=args.work_dir.resolve()
    project.mkdir(parents=True,exist_ok=False)
    source_rows=list(records(ROOT/"data/matrix_source_files.csv"))
    if len(source_rows)!=61:
        raise ValueError("Unexpected count of upstream matrix source files")
    pairs=[(r["bundle_path"],r["project_alias"]) for r in source_rows]
    pairs.extend([
        ("data/matrix/source_candidate_inventory_v5.csv","benchmark_core/data/unified_matrix_v5/source_candidate_inventory_v5.csv"),
        ("data/matrix/source_candidate_summary_v5.csv","benchmark_core/data/unified_matrix_v5/source_candidate_summary_v5.csv"),
        ("data/config/model_eligibility_v3.csv","benchmark_core/config/model_eligibility_v3.csv"),
    ])
    for rel,alias in pairs:
        destination=project/alias
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/rel,destination)
    env=os.environ.copy()
    env.update(BE_REPLAY_PROJECT=str(project),PYTHONUTF8="1",PYTHONDONTWRITEBYTECODE="1")
    subprocess.run([sys.executable,str(ROOT/"data/code/build_unified_matrix_v5.py")],env=env,cwd=project,check=True)
    output=project/"benchmark_core/data/unified_matrix_v5"
    sizes={}
    for filename in ("observation_matrix_v5.csv","design_master_v5.csv","editor_context_master_v5.csv"):
        sizes[filename]=sum(1 for row in records(output/filename))
    if sizes["observation_matrix_v5.csv"]!=1_147_701:
        raise ValueError(f"Unexpected regenerated observation count: {sizes}")
    print(json.dumps(sizes,indent=2))


if __name__=="__main__":
    main()

