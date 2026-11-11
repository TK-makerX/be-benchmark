import argparse,itertools,json
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import spearmanr
IDS=['crispron_be','deepbe','forecast_be','deepbaseeditor']
def main():
    p=argparse.ArgumentParser();p.add_argument('--project-root',type=Path,required=True);p.add_argument('--revision-root',type=Path,required=True);a=p.parse_args();out=a.revision_root/'17_R3.3';d=pd.read_csv(out/'data/frozen_substitution_panel.csv');coverage=[]
    for mid in IDS:
        files=list((out/'inference/raw'/mid).glob('predictions*.csv'));assert len(files)==1
        z=pd.read_csv(files[0]);z=z[z.task_type=='efficiency'];assert not z.benchmark_id.duplicated().any();d=d.merge(z[['benchmark_id','predicted_efficiency']].rename(columns={'predicted_efficiency':mid}),on='benchmark_id',how='left',validate='one_to_one');coverage.append(dict(model_id=mid,input_rows=len(d),finite_scores=int(d[mid].notna().sum()),missing_scores=int(d[mid].isna().sum())))
    base=d[d.replicate==0].set_index('parent_id');assert len(base)==200 and base[IDS].notna().all().all();mut=d[d.replicate>0].copy();complete=mut[IDS].notna().all(axis=1);mut['all_four_complete']=complete
    original=pd.read_csv(a.project_root/'benchmark_core/formal_results/T1_v8_label_blind_known_crispron_clean/data/T1_V8_COMMON_VALID_MODEL_OBSERVATIONS.csv.gz',usecols=['observation_id','model_id','prediction_for_association']);replay=[];rows=[]
    for mid in IDS:
        old=original[original.model_id==mid].set_index('observation_id').loc[base.original_observation_id,'prediction_for_association'].to_numpy();err=float(np.max(np.abs(base[mid].to_numpy()-old)));replay.append(dict(model_id=mid,n=200,max_abs_error=err));assert err<1e-5,(mid,err)
        iqr=float(np.subtract(*np.quantile(base[mid],[.75,.25])));assert iqr>0;b=mut.parent_id.map(base[mid]).to_numpy();delta=mut[mid].to_numpy()-b;mut[mid+'__delta']=delta;mut[mid+'__abs_iqr']=np.abs(delta)/iqr
        s=np.sort(base[mid].to_numpy());ecdf=lambda v:(np.searchsorted(s,v,'left')+np.searchsorted(s,v,'right'))/(2*len(s));mut[mid+'__abs_ecdf_shift']=np.abs(ecdf(mut[mid].to_numpy())-ecdf(b))
        for scope,mask in [('per_model_finite',mut[mid].notna()),('all_four_common',complete)]:
            q=mut.loc[mask];groups=q.groupby('parent_id')[[mid+'__abs_iqr',mid+'__abs_ecdf_shift']].agg(['sum','count']);rng=np.random.default_rng(20261027);ix=rng.integers(len(groups),size=(5000,len(groups)));a0=groups[(mid+'__abs_iqr','sum')].to_numpy();a1=groups[(mid+'__abs_iqr','count')].to_numpy();boot=a0[ix].sum(axis=1)/a1[ix].sum(axis=1);lo,hi=np.quantile(boot,[.025,.975]);rhos=[spearmanr(g[mid],g.parent_id.map(base[mid]))[0] for _,g in q.groupby('replicate')]
            rows.append(dict(model_id=mid,scope=scope,n_pairs=len(q),n_parent_guides=q.parent_id.nunique(),baseline_iqr=iqr,mean_abs_native_delta=float(q[mid+'__delta'].abs().mean()),median_abs_native_delta=float(q[mid+'__delta'].abs().median()),mean_abs_delta_over_iqr=float(q[mid+'__abs_iqr'].mean()),ci_low=lo,ci_high=hi,mean_abs_ecdf_shift=float(q[mid+'__abs_ecdf_shift'].mean()),min_replicate_rank_stability=float(min(rhos)),max_replicate_rank_stability=float(max(rhos))))
    d.to_csv(out/'data/all_native_predictions.csv.gz',index=False);mut.to_csv(out/'data/paired_substitution_scores.csv.gz',index=False);pd.DataFrame(rows).to_csv(out/'results/substitution_sensitivity.csv',index=False);pd.DataFrame(coverage).to_csv(out/'results/execution_coverage.csv',index=False);pd.DataFrame(replay).to_csv(out/'results/original_score_replay.csv',index=False)
    common=mut[complete];parents=sorted(common.parent_id.unique());rng=np.random.default_rng(20261027);ix=rng.integers(len(parents),size=(5000,len(parents)));pairs=[]
    for aa,bb in itertools.combinations(IDS,2):
        q=common.assign(delta=common[aa+'__abs_iqr']-common[bb+'__abs_iqr']).groupby('parent_id').delta.agg(['sum','count']).loc[parents];v=q['sum'].to_numpy()[ix].sum(axis=1)/q['count'].to_numpy()[ix].sum(axis=1);lo,hi=np.quantile(v,[.025,.975]);pairs.append(dict(model_1=aa,model_2=bb,normalized_mean_abs_change_difference=float(q['sum'].sum()/q['count'].sum()),ci_low=lo,ci_high=hi))
    pd.DataFrame(pairs).to_csv(out/'results/paired_sensitivity_differences.csv',index=False)
    report=dict(status='PASS',n_parent_guides=200,n_generated_mutations=600,n_common_pairs=int(complete.sum()),missing_pairs_not_imputed=True,native_checkpoint_fitting=False,baseline_scores_reproduced=True,bootstrap_replicates=5000,bootstrap_unit='parent guide, keeping all its sampled mutations together',standardization='Absolute change divided by that model baseline-score IQR, fixed for resampling; empirical CDF shift supplied separately.',interpretation='In silico sensitivity of frozen predictors; does not infer experimental mutation effects, predictive correctness or a universal robustness ranking.')
    (out/'results/validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(pd.DataFrame(rows).round(5).to_string(index=False));print(pd.DataFrame(coverage).to_string(index=False));print(pd.DataFrame(replay).to_string(index=False))
if __name__=='__main__':main()
