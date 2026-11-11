"""Verify empty native position output; do not change the historical run."""
import argparse, hashlib, json, sys
from pathlib import Path
import pandas as pd

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--project-root',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args(); a.output_dir.mkdir(parents=True,exist_ok=True)
    repo=a.project_root/'FORECasT-BE/repo'
    sys.path.insert(0,str(repo)); import forecast_be
    panel=pd.read_csv(a.project_root/'benchmark_core/data/processed/T3_position_profile_matrix/panels_v1/T3_v1_forecast_be.csv')
    ids=['T2T_9926595b5b07','T2T_3b7c0e3f35b1']
    records=[]
    for _,r in panel[panel.target_profile_id.isin(ids)].iterrows():
        result=forecast_be.predict(r.protospacer_20nt,editor=r.editor_class,mean=None,std=None)
        row=dict(target_profile_id=r.target_profile_id,protospacer=r.protospacer_20nt,editor_class=r.editor_class,
                 replay_completed=True,exception=None,native_return=[[int(pos),None if v is None else float(v)] for pos,v in result],
                 finite_output_count=sum(v is not None for _,v in result),
                 interpretation='Current replay verifies valid empty output at native supported positions; historical completion was not logged.')
        assert row['finite_output_count']==0
        records.append(row)
    assert len(records)==2
    assets=[repo/'forecast_be.py']+sorted((repo/'saved_models').rglob('*.joblib'))
    out=dict(records=records,source_hashes={str(f.relative_to(a.project_root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in assets},
             historical_run_modified=False)
    (a.output_dir/'FORECasT_empty_output_replay.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps(records,indent=2))

if __name__=='__main__': main()
