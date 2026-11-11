"""Score the separate Marquart canonical-window transfer panel without fitting."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

IDS=['crispron_be','deepbe','forecast_be','deepbaseeditor']
def rho(x,y):return float(spearmanr(x,y).correlation) if len(x)>1 and np.ptp(x)>0 and np.ptp(y)>0 else np.nan
def main():
    sys.stdout.reconfigure(encoding='utf-8');p=argparse.ArgumentParser();p.add_argument('--revision-root',type=Path,required=True);a=p.parse_args();base=a.revision_root/'02_R2.1';out=base/'external_validation'
    (out/'results').mkdir(exist_ok=True);meta=pd.read_csv(base/'endpoint_correction/Marquart_ABE_compatible_inference_panel.csv')
    neighbors=pd.read_csv(base/'overlap_audit/CRISPRon_guide_neighbor_membership.csv');neighbors=neighbors[neighbors.task=='Marquart127']
    wide=meta.merge(neighbors[['record_id','development_min_hamming']],left_on='benchmark_id',right_on='record_id',validate='one_to_one')
    run_counts=[]
    for mid in IDS:
        files=list((out/'raw'/mid).glob('predictions*.csv'));assert len(files)==1
        d=pd.read_csv(files[0]);d=d[d.task_type=='efficiency'].copy()
        assert not d.benchmark_id.duplicated().any()
        d[mid]=pd.to_numeric(d.predicted_efficiency,errors='raise');assert np.isfinite(d[mid]).all()
        wide=wide.merge(d[['benchmark_id',mid]],on='benchmark_id',validate='one_to_one',how='left')
        run_counts.append(dict(model_id=mid,input_targets=len(meta),finite_efficiency_predictions=len(d),missing_predictions=int(wide[mid].isna().sum())))
    assert not wide[IDS].isna().any().any();wide.to_csv(out/'results/Marquart127_measured_and_predicted.csv',index=False)
    pd.DataFrame(run_counts).to_csv(out/'results/execution_coverage.csv',index=False)
    rows=[];draws=[];pairrows=[]
    for radius in [0,1,2]:
        d=wide[wide.development_min_hamming>radius].sort_values('benchmark_id').copy();guides,ix=np.unique(d.guide_20nt,return_inverse=True)
        clusters=[np.flatnonzero(ix==i) for i in range(len(guides))];y=d.canonical_3_10_rate.to_numpy();pred=d[IDS].to_numpy()
        boot=np.empty((5000,4));rng=np.random.default_rng(20261025+radius)
        for b in range(5000):
            ids=np.concatenate([clusters[i] for i in rng.integers(0,len(guides),len(guides))])
            for j in range(4):boot[b,j]=rho(pred[ids,j],y[ids])
        bt=pd.DataFrame(boot,columns=IDS);bt.insert(0,'excluded_hamming_radius',radius);bt.insert(1,'replicate',np.arange(1,5001));draws.append(bt)
        for j,mid in enumerate(IDS):
            x=pred[:,j];lo,hi=np.nanquantile(boot[:,j],[.025,.975])
            row=dict(model_id=mid,excluded_hamming_radius=radius,n_targets=len(d),n_guide_clusters=len(guides),
                spearman=rho(x,y),ci_low=float(lo),ci_high=float(hi),valid_bootstrap=int(np.isfinite(boot[:,j]).sum()),
                full_guide_endpoint_spearman=rho(x,d.total_efficiency.to_numpy()),
                endpoint='any canonical A>G in positions 3-10 / author-retained reads including WT',
                independence='CRISPRon reference excluded; other model training membership unknown',
                interpretation='source-transfer association; editor/assay shift; DeepBE ABE8e proxy, FORECasT z-score, DBE head window not independently verified')
            if mid=='crispron_be':
                row.update(mae=float(np.abs(x-y).mean()),rmse=float(np.sqrt(np.mean((x-y)**2))),
                    r2=float(1-np.sum((x-y)**2)/np.sum((y-y.mean())**2)),mean_prediction=float(x.mean()),mean_observation=float(y.mean()))
            rows.append(row)
        for j in range(4):
            for k in range(j+1,4):
                delta=boot[:,j]-boot[:,k];lo,hi=np.nanquantile(delta,[.025,.975]);pairrows.append(dict(radius=radius,model_a=IDS[j],model_b=IDS[k],delta_rho=rho(pred[:,j],y)-rho(pred[:,k],y),lo=float(lo),hi=float(hi),interpretation='same-target association difference only; no universal model ranking'))
    pd.DataFrame(rows).to_csv(out/'results/external_association_metrics.csv',index=False)
    pd.DataFrame(pairrows).to_csv(out/'results/external_paired_association_differences.csv',index=False)
    pd.concat(draws).to_csv(out/'results/external_target_bootstrap.csv.gz',index=False,compression='gzip')
    checks=dict(status='PASS',n_input=127,all_four_models_complete=True,bootstrap_replicates=5000,
        resampling_unit='identical 20nt guide cluster',resampling_paired_across_models=True,fitting=False,
        observation_min=float(wide.canonical_3_10_rate.min()),observation_max=float(wide.canonical_3_10_rate.max()),
        source_filtered_panel=True,strict_four_model_independence=False,not_pooled_with_original_T1=True)
    (out/'results/validation.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    print(pd.DataFrame(rows)[['model_id','excluded_hamming_radius','n_targets','spearman','ci_low','ci_high','mae','rmse','r2']].to_string(index=False))

if __name__=='__main__':main()
