"""Run frozen scalar checkpoints on a separate, explicit window endpoint.

All generated outputs and the legacy writer's working copy remain in output-dir.
No fitting and no change of checkpoint/editor branch from the frozen T1 setup.
"""
import argparse,json,hashlib,os,re,shutil,subprocess,sys,time
from pathlib import Path

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    sys.stdout.reconfigure(encoding='utf-8')
    p=argparse.ArgumentParser()
    for n in ['project-root','panel','output-dir','tf-python','forecast-python']:p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();root=a.project_root;out=a.output_dir;out.mkdir(exist_ok=True,parents=True)
    code=out/'code';code.mkdir(exist_ok=True);(out/'logs').mkdir(exist_ok=True)
    files=['common.py','run_crispron_be_pilot.py','run_forecast_be_pilot.py','run_deepbe_pilot.py','run_deepbaseeditor_abe_efficiency_pilot.py']
    sources={}
    for f in files:
        src=root/'benchmark_core/adapters'/f;sources[str(src)]=digest(src);s=src.read_text(encoding='utf-8')
        # Defaults are removed so a copied adapter never relies on the original machine path.
        s=re.sub(r'default=Path\(r"E:[^"]+"\)', 'required=True',s)
        s=s.replace('DEEPBE_ROOT = Path(r"E:\\AIselect\\BEs\\DeepBE")','DEEPBE_ROOT = None')
        s=s.replace('DEFAULT_WEIGHTS = DEEPBE_ROOT / "resources" / "extracted_models"','DEFAULT_WEIGHTS = None')
        s=s.replace('default=DEFAULT_WEIGHTS','required=True')
        (code/f).write_text(s,encoding='utf-8')
    # The upstream DBE script writes a fixed relative output file; isolate that behavior.
    native=root/'DeepBaseEditor-Paired-Library/repo/DeepBaseEditor/ABE_Efficiency'
    working=out/'legacy_dbe_working_copy'
    shutil.copytree(native,working,dirs_exist_ok=True,ignore=shutil.ignore_patterns('outputs','__pycache__'))
    specs=[('deepbaseeditor','run_deepbaseeditor_abe_efficiency_pilot.py',a.tf_python,['--repo',str(working)]),
        ('forecast_be','run_forecast_be_pilot.py',a.forecast_python,['--repo',str(root/'FORECasT-BE/repo')]),
        ('deepbe','run_deepbe_pilot.py',a.tf_python,['--weights',str(root/'DeepBE/resources/extracted_models')]),
        ('crispron_be','run_crispron_be_pilot.py',a.tf_python,['--repo',str(root/'CRISPRon-BE/repo'),'--tools-bin',str(root/'tools/bin')])]
    env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1';env['PYTHONIOENCODING']='utf-8'
    records=[]
    for mid,script,exe,extra in specs:
        dest=out/'raw'/mid;dest.mkdir(exist_ok=True,parents=True)
        cmd=[str(exe),str(code/script),'--panel',str(a.panel),'--output-dir',str(dest),*extra]
        print(json.dumps(dict(event='start',model=mid)),flush=True);t=time.time()
        with (out/'logs'/f'{mid}.log').open('w',encoding='utf-8') as log:
            r=subprocess.run(cmd,cwd=out,env=env,stdout=log,stderr=subprocess.STDOUT)
        row=dict(model_id=mid,returncode=r.returncode,seconds=time.time()-t,command=cmd)
        records.append(row);print(json.dumps(row),flush=True)
        (out/'logs/execution.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
        if r.returncode: raise RuntimeError(f'{mid} failed; inspect log')
    assert all(digest(Path(k))==v for k,v in sources.items())
    (out/'logs/provenance.json').write_text(json.dumps(dict(source_adapter_sha256=sources,panel_sha256=digest(a.panel),
        generated_adapters={p.name:digest(p) for p in code.glob('*.py')},fitting_performed=False,original_adapter_files_unchanged=True),indent=2),encoding='utf-8')

if __name__=='__main__':main()
