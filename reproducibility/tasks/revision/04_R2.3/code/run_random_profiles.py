"""Run frozen local profile adapters; all generated files stay in the revision."""
import argparse,hashlib,json,os,re,subprocess,sys,time
from pathlib import Path
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    sys.stdout.reconfigure(encoding='utf-8');p=argparse.ArgumentParser()
    for name in ['project-root','revision-root','torch-python','tf-python','forecast-python','yachie-python']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--models',nargs='+');a=p.parse_args();out=a.revision_root/'04_R2.3';code=out/'code/adapters';code.mkdir(exist_ok=True,parents=True)
    names=['common.py','t2_common.py','run_crispron_be_pilot.py','run_deepbe_pilot.py','run_t2_behive_bystander.py','run_t2_bedict_v2.py','run_t2_crispron_be.py','run_t2_deepbe.py','run_t2_be_endo_position.py','run_t2_forecast_position.py','run_t2_yachie_observed_products.py'];hashes={}
    for n in names:
        f=a.project_root/'benchmark_core/adapters'/n;hashes[str(f)]=sha(f);s=f.read_text(encoding='utf-8');s=re.sub(r'default=Path\(r"E:[^"]+"\)','required=True',s);s=re.sub(r"(?m)^DEEPBE_ROOT\s*=\s*Path\([^\r\n]+\)$", "DEEPBE_ROOT = None", s).replace('DEFAULT_WEIGHTS = DEEPBE_ROOT / "resources" / "extracted_models"','DEFAULT_WEIGHTS = None').replace('default=DEFAULT_WEIGHTS','required=True')
        if n=='run_t2_yachie_observed_products.py':
            s=s.replace('observed_outcome_candidate_score','sequence_enumerated_outcome_candidate_score').replace('observed_position_candidate_score','sequence_enumerated_position_candidate_score').replace('Yachie historical model scores only observed edited product candidates; not a complete product distribution.','Yachie native scores on all sequence-enumerated canonical 3–10 edited candidates; no measured frequency supplied; ranking scores, not probabilities.')
        (code/n).write_text(s,encoding='utf-8')
    specs=[('forecast_be','run_t2_forecast_position.py',a.forecast_python,['--repo',a.project_root/'FORECasT-BE/repo']),
        ('deepbe','run_t2_deepbe.py',a.tf_python,['--weights',a.project_root/'DeepBE/resources/extracted_models']),
        ('crispron_be','run_t2_crispron_be.py',a.tf_python,['--repo',a.project_root/'CRISPRon-BE/repo','--tools-bin',a.project_root/'tools/bin']),
        ('be_endo','run_t2_be_endo_position.py',a.tf_python,['--repo',a.project_root/'BE_Endo/repo']),
        ('bedict_v2','run_t2_bedict_v2.py',a.torch_python,['--repo',a.project_root/'BEDICT-V2/repo']),
        ('behive_bystander','run_t2_behive_bystander.py',a.torch_python,['--repo',a.project_root/'BE-Hive-bystander/repo']),
        ('yachie_base_editing_prediction','run_t2_yachie_observed_products.py',a.yachie_python,['--repo',a.project_root/'Yachie-base-editing-prediction/repo','--outcome-profile',out/'data/sequence_enumerated_canonical_candidates.csv'])]
    env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONIOENCODING='utf-8',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    records=[]
    for mid,script,exe,extra in specs:
        if a.models and mid not in a.models:continue
        dest=out/'raw'/mid;dest.mkdir(exist_ok=True,parents=True);cmd=list(map(str,[exe,code/script,'--panel',out/'data/frozen_union_inference_panel.csv','--output-dir',dest,*extra]));start=time.time();print('start',mid,flush=True)
        with (out/'logs'/f'{mid}.log').open('w',encoding='utf-8') as log:result=subprocess.run(cmd,cwd=out,env=env,stdout=log,stderr=subprocess.STDOUT)
        record=dict(model_id=mid,returncode=result.returncode,seconds=time.time()-start,command=cmd);records.append(record);print(json.dumps(record),flush=True);(out/'logs'/f'{mid}_execution.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
        if result.returncode:raise RuntimeError(mid+' failed; inspect saved log')
    assert all(sha(Path(k))==v for k,v in hashes.items());(out/'logs/adapter_provenance.json').write_text(json.dumps(dict(original_adapter_sha256=hashes,original_adapters_unchanged=True,new_adapters={f.name:sha(f) for f in code.glob('*.py')},fitting_performed=False),indent=2),encoding='utf-8')
if __name__=='__main__':main()
