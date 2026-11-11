"""Correct frozen Fig.4 eligibility, with paired target-cluster uncertainty.

Inputs are immutable historical predictions, execution evidence and the native
FORECasT empty-output replay. Missing supported outputs remain missing in the
returned-only analysis. Zero completion is an explicit sensitivity assumption.
"""
import argparse, hashlib, json, platform, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

MODELS=['behive_bystander','forecast_be','bedict_v2','deepbe','crispron_be','be_endo','yachie_base_editing_prediction','epcnnbe','deepbaseeditor']
RATE={'behive_bystander','bedict_v2','deepbe','crispron_be','epcnnbe','deepbaseeditor'}

def truth(s): return s.astype(str).str.lower().isin(['true','1','yes'])
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def corr(x,y):
    return float(spearmanr(x,y).statistic) if len(x)>1 and np.ptp(x)>0 and np.ptp(y)>0 else float('nan')

def weighted_spearman_setup(x,y):
    """Exact row-replication Spearman from integer bootstrap multiplicities."""
    _,ix=np.unique(x,return_inverse=True); _,iy=np.unique(y,return_inverse=True)
    def calc(w):
        n=w.sum()
        if n<2: return np.nan
        wx=np.bincount(ix,weights=w); wy=np.bincount(iy,weights=w)
        rx=(np.cumsum(wx)-wx/2)[ix]-n/2
        ry=(np.cumsum(wy)-wy/2)[iy]-n/2
        den=np.sqrt(np.sum(w*rx*rx)*np.sum(w*ry*ry))
        return float(np.sum(w*rx*ry)/den) if den>0 else np.nan
    return calc

def statistics(d,model):
    x=d['_pred_filled'].to_numpy(float); y=d['_obs'].to_numpy(float)
    mask=d['returned'].to_numpy(bool)
    out=dict(targets=d.target_profile_id.nunique(),positions=len(d),returned_positions=int(mask.sum()),
             coverage=float(mask.mean()),rho_zero=corr(x,y),rho_returned=corr(x[mask],y[mask]))
    out['delta_zero_minus_returned']=out['rho_zero']-out['rho_returned']
    if model in RATE:
        for label,m in [('zero',np.ones(len(x),bool)),('returned',mask)]:
            out['mae_'+label]=float(np.mean(np.abs(x[m]-y[m])))
            out['rmse_'+label]=float(np.sqrt(np.mean((x[m]-y[m])**2)))
    return out

def bootstrap(d,n,seed):
    d=d.sort_values(['target_profile_id','protospacer_position','reference_base'])
    x=d['_pred_filled'].to_numpy(float); y=d['_obs'].to_numpy(float); m=d.returned.to_numpy(bool)
    targets,idx=np.unique(d.target_profile_id,return_inverse=True)
    zfun=weighted_spearman_setup(x,y); rfun=weighted_spearman_setup(x[m],y[m])
    rng=np.random.default_rng(seed); a=np.empty((n,3)); check_error=0.
    for i in range(n):
        counts=rng.multinomial(len(targets),np.ones(len(targets))/len(targets)); weights=counts[idx]
        z=zfun(weights); r=rfun(weights[m]); a[i]=[z,r,z-r]
        if i<20:
            for xx,yy,ww,v in [(x,y,weights,z),(x[m],y[m],weights[m],r)]:
                direct=corr(np.repeat(xx,ww),np.repeat(yy,ww))
                if np.isfinite(direct): check_error=max(check_error,abs(direct-v))
    assert check_error<1e-12,check_error
    table=pd.DataFrame(a,columns=['rho_zero','rho_returned','delta_zero_minus_returned'])
    ci={}
    for col in table:
        lo,hi=np.nanquantile(table[col],[.025,.975]); ci.update({col+'_lo':float(lo),col+'_hi':float(hi),col+'_finite_bootstrap':int(table[col].notna().sum())})
    return table,ci,check_error

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap=argparse.ArgumentParser(); ap.add_argument('--project-root',type=Path,required=True)
    ap.add_argument('--revision-root',type=Path,required=True); ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--bootstrap',type=int,default=5000);ap.add_argument('--seed',type=int,default=20261025)
    a=ap.parse_args(); out=a.output_dir
    for f in ['data','results','qa','figures']: (out/f).mkdir(parents=True,exist_ok=True)
    base=a.project_root/'SCI-data/Fig.4/source_data/T3_position_profile_v1'
    paths=[base/'T3_position_joined_universe_predictions.csv',base/'T3_position_metrics.csv',
        a.revision_root/'02_R2.1/execution_eligibility/T3_execution_eligibility_by_target.csv',
        a.revision_root/'02_R2.1/overlap_audit/CRISPRon_actual_backend_by_target.csv',
        out/'results/FORECasT_empty_output_replay.json']
    hashes={str(p):sha(p) for p in paths}
    joined=pd.read_csv(paths[0],low_memory=False); joined['returned']=truth(joined.has_model_prediction)
    oldmetrics=pd.read_csv(paths[1]); elig=pd.read_csv(paths[2]); cr=pd.read_csv(paths[3]); replay=json.loads(paths[4].read_text())
    replayids={r['target_profile_id'] for r in replay['records'] if r['replay_completed'] and r['finite_output_count']==0}
    manifest=[]
    for _,r in elig.iterrows():
        skip=truth(pd.Series([r.explicit_skip_record])).iloc[0]; pos=truth(pd.Series([r.backend_success_confirmed_by_finite_output])).iloc[0]
        empty=r.model_id=='forecast_be' and r.target_profile_id in replayids
        state='explicit_skip' if skip else ('finite_native_output' if pos else ('valid_empty_current_replay' if empty else 'unresolved_execution'))
        manifest.append(dict(model_id=r.model_id,target_profile_id=r.target_profile_id,eligible=bool((pos or empty) and not skip),execution_state=state))
    for _,r in cr[cr.task=='T3'].iterrows():
        ok=truth(pd.Series([r.backend_executed])).iloc[0]
        manifest.append(dict(model_id='crispron_be',target_profile_id=r.target_profile_id,eligible=bool(ok),execution_state='finite_native_output' if ok else 'explicit_skip'))
    manifest=pd.DataFrame(manifest)
    assert not manifest.duplicated(['model_id','target_profile_id']).any()
    assert not (manifest.execution_state=='unresolved_execution').any()
    joined=joined.merge(manifest,on=['model_id','target_profile_id'],how='left',validate='many_to_one')
    assert joined.eligible.notna().all()
    assert np.isfinite(joined['_obs']).all() and np.isfinite(joined['_pred_filled']).all()
    assert not joined.duplicated(['model_id','target_profile_id','protospacer_position','reference_base']).any()
    assert (joined.loc[~joined.returned,'_pred_filled']==0).all()
    fixed=joined[joined.eligible].copy(); excluded=joined[~joined.eligible].copy()
    assert not excluded.returned.any()
    manifest.to_csv(out/'data/T3_target_execution_manifest.csv',index=False)
    fixed.to_csv(out/'data/T3_corrected_joined.csv.gz',index=False,compression='gzip')
    excluded[['model_id','target_profile_id','protospacer_position','reference_base','_obs','execution_state']].to_csv(out/'data/T3_excluded_positions.csv',index=False)
    rows=[]; baselines=[]; draws=[]; checks=[]; pert=[]
    for mi,model in enumerate(MODELS):
        orig=joined[joined.model_id==model]; d=fixed[fixed.model_id==model]; mm=manifest[manifest.model_id==model]
        baseline=statistics(orig,model); formal=oldmetrics[oldmetrics.model_id==model].iloc[0]
        errors=[abs(baseline['rho_zero']-formal.pooled_spearman),abs(baseline['coverage']-formal.prediction_coverage_fraction)]
        if model in RATE: errors += [abs(baseline['mae_zero']-formal.mae),abs(baseline['rmse_zero']-formal.rmse)]
        assert max(errors)<1e-9
        baseline.update(model_id=model,max_original_replication_error=max(errors)); baselines.append(baseline)
        row=statistics(d,model); bt,ci,err=bootstrap(d,a.bootstrap,a.seed+mi)
        row.update(ci); row.update(model_id=model,selected_targets=len(mm),explicit_skip_targets=int((mm.execution_state=='explicit_skip').sum()),
                                  valid_empty_replay_targets=int((mm.execution_state=='valid_empty_current_replay').sum()),execution_coverage=float(mm.eligible.mean()),
                                  bootstrap_seed=a.seed+mi,bootstrap_replicates=a.bootstrap)
        rows.append(row);bt.insert(0,'model_id',model);bt.insert(1,'replicate',np.arange(1,len(bt)+1));draws.append(bt)
        checks.append(dict(model_id=model,weighted_bootstrap_vs_explicit_scipy_max_error=err))
        for tid,g in d.groupby('target_profile_id',sort=True): pert.append(dict(model_id=model,target_profile_id=tid,**statistics(g,model)))
        print(json.dumps({k:row[k] for k in ['model_id','targets','positions','returned_positions','rho_zero','rho_returned','delta_zero_minus_returned','delta_zero_minus_returned_lo','delta_zero_minus_returned_hi']}),flush=True)
    res=pd.DataFrame(rows);res.to_csv(out/'results/T3_corrected_metrics.csv',index=False)
    pd.DataFrame(baselines).to_csv(out/'results/T3_original_replication_metrics.csv',index=False)
    pd.concat(draws).to_csv(out/'results/T3_paired_target_bootstrap.csv.gz',index=False,compression='gzip')
    pd.DataFrame(pert).to_csv(out/'results/T3_corrected_per_target_metrics.csv',index=False)
    # Optional exclusion of the two valid empty targets is not the primary policy.
    fd=fixed[(fixed.model_id=='forecast_be')&(~fixed.target_profile_id.isin(replayids))]
    pd.DataFrame([dict(model_id='forecast_be',policy='exclude_valid_empty_targets_sensitivity',**statistics(fd,'forecast_be'))]).to_csv(out/'results/FORECasT_empty_target_sensitivity.csv',index=False)
    assert dict(res.set_index('model_id').targets)["crispron_be"]==149
    assert dict(res.set_index('model_id').targets)["be_endo"]==162
    assert len(excluded[excluded.model_id=='crispron_be'])==1629
    assert len(excluded[excluded.model_id=='be_endo'])==1555
    assert hashes=={str(p):sha(p) for p in paths}
    qa=dict(status='PASS',input_hashes_before_and_after=hashes,original_data_modified=False,
        weighted_rank_verification=checks,bootstrap_unit='target_profile_id',
        policies='returned positions versus zero completion of unreturned observed positions within the same execution-eligible target sample',
        caveat='Zero completion is sensitivity, not a biological zero assertion. Panels are model-specific and outcome-selected; weights may have development overlap.',
        python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__)
    (out/'qa/validation.json').write_text(json.dumps(qa,indent=2),encoding='utf-8')

if __name__=='__main__': main()
