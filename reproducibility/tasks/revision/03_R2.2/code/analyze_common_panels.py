"""Compare frozen profile predictions on identical targets and position keys."""
import argparse,itertools,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

IDS=['behive_bystander','forecast_be','bedict_v2','deepbe','crispron_be','be_endo','yachie_base_editing_prediction']
RATE={'behive_bystander','bedict_v2','deepbe','crispron_be'}
KEY=['target_profile_id','protospacer_position','reference_base']

def rho(x,y):return float(spearmanr(x,y).correlation) if len(x)>1 and np.ptp(x)>0 and np.ptp(y)>0 else np.nan
def wfun(x,y):
    _,ix=np.unique(x,return_inverse=True);_,iy=np.unique(y,return_inverse=True)
    def f(w):
        n=w.sum();wx=np.bincount(ix,weights=w);wy=np.bincount(iy,weights=w)
        rx=(np.cumsum(wx)-wx/2)[ix]-n/2;ry=(np.cumsum(wy)-wy/2)[iy]-n/2
        den=np.sqrt(np.sum(w*rx*rx)*np.sum(w*ry*ry));return np.sum(w*rx*ry)/den if den>0 else np.nan
    return f

def main():
    sys.stdout.reconfigure(encoding='utf-8');p=argparse.ArgumentParser();p.add_argument('--project-root',type=Path,required=True);p.add_argument('--revision-root',type=Path,required=True);a=p.parse_args()
    out=a.revision_root/'03_R2.2';[(out/x).mkdir(exist_ok=True,parents=True) for x in ['data','results','figures','response']]
    core=a.project_root/'benchmark_core';t2=pd.read_csv(core/'formal_results/T2_expand_v0/T2_expanded_position_joined_predictions.csv')
    t3=pd.read_csv(a.revision_root/'02_R2.1/denominator_correction/data/T3_corrected_joined.csv.gz',low_memory=False);t3=t3[t3.model_id.isin(IDS)].copy()
    t2['returned']=True;t2['_pred_filled']=t2['_pred'];t3['returned']=t3.returned.astype(str).str.lower()=='true'
    products=pd.read_csv(core/'formal_results/T2_expand_v0/T2_expanded_outcome_per_target_metrics.csv')
    metrics=[];deltas=[];membership=[];inventory=[];dataframes=[];rng=np.random.default_rng(20261025)
    panels=[('all_seven',IDS),('four_conditional',[m for m in IDS if m in RATE])]+[(f'pair_{i+1:02}',list(pair)) for i,pair in enumerate(itertools.combinations(IDS,2))]
    for task,d in [('T2',t2),('T3',t3)]:
        assert not d.duplicated(['model_id']+KEY).any()
        sets={m:set(d[d.model_id==m].target_profile_id) for m in IDS}
        for panel,models in panels:
            tids=set.intersection(*[sets[m] for m in models])
            for tid in sorted(tids):membership.append(dict(task=task,panel=panel,models=';'.join(models),target_profile_id=tid))
            # Shared positions: both biological observations and prediction availability identical.
            policies=['shared_returned_positions']+(['shared_all_observed_zero_completion'] if task=='T3' else [])
            for policy in policies:
                s=d[d.target_profile_id.isin(tids)&d.model_id.isin(models)].copy()
                if policy=='shared_returned_positions':s=s[s.returned]
                counts=s.groupby(KEY).model_id.nunique();keys=counts[counts==len(models)].reset_index()[KEY]
                s=s.merge(keys,on=KEY,validate='many_to_one');w=s.pivot(index=KEY,columns='model_id',values='_pred_filled').reset_index()
                ob=s.groupby(KEY)['_obs'].agg(['first','nunique']).reset_index();assert (ob['nunique']==1).all()
                w=w.merge(ob[KEY+['first']],on=KEY);w=w.rename(columns={'first':'observed'})
                inventory.append(dict(task=task,panel=panel,policy=policy,models=';'.join(models),common_target_ids=len(tids),scored_targets=w.target_profile_id.nunique(),identical_position_rows=len(w)))
                if len(w)<2:continue
                w['task']=task;w['panel']=panel;w['policy']=policy;dataframes.append(w)
                clusters,ix=np.unique(w.target_profile_id,return_inverse=True);obs=w.observed.to_numpy();pred=w[models].to_numpy();fns=[wfun(pred[:,j],obs) for j in range(len(models))]
                boots=np.empty((5000,len(models)))
                for b in range(5000):
                    weights=rng.multinomial(len(clusters),np.ones(len(clusters))/len(clusters))[ix]
                    for j,f in enumerate(fns):boots[b,j]=f(weights)
                points=[]
                for j,m in enumerate(models):
                    point=rho(pred[:,j],obs);points.append(point);valid=boots[:,j][np.isfinite(boots[:,j])];ci=np.quantile(valid,[.025,.975]) if len(valid) else [np.nan,np.nan]
                    row=dict(task=task,panel=panel,policy=policy,model_id=m,common_target_ids=len(tids),scored_targets=len(clusters),n_positions=len(w),spearman=point,ci_low=ci[0],ci_high=ci[1],finite_bootstrap=len(valid))
                    if m in RATE:row.update(mae=float(np.abs(pred[:,j]-obs).mean()),rmse=float(np.sqrt(np.mean((pred[:,j]-obs)**2))))
                    metrics.append(row)
                for j,k in itertools.combinations(range(len(models)),2):
                    diff=boots[:,j]-boots[:,k];v=diff[np.isfinite(diff)];ci=np.quantile(v,[.025,.975]) if len(v) else [np.nan,np.nan]
                    deltas.append(dict(task=task,panel=panel,policy=policy,model_a=models[j],model_b=models[k],n_targets=len(clusters),n_positions=len(w),delta_rho=points[j]-points[k],ci_low=ci[0],ci_high=ci[1],finite_bootstrap=len(v)))
        print(task,'complete',flush=True)
    pd.DataFrame(inventory).to_csv(out/'results/common_panel_inventory.csv',index=False)
    pd.DataFrame(metrics).to_csv(out/'results/common_position_metrics.csv',index=False)
    pd.DataFrame(deltas).to_csv(out/'results/common_position_paired_differences.csv',index=False)
    pd.DataFrame(membership).to_csv(out/'data/common_target_membership.csv',index=False)
    pd.concat(dataframes,ignore_index=True).to_csv(out/'data/common_position_scoring_inputs.csv.gz',index=False,compression='gzip')
    prows=[];pdelta=[];pids=[]
    # Distributional comparisons only among normalized-product models; scores are separate.
    groups=[('all_four_distributions',sorted(RATE))]+[(f'pair_product_{i+1:02}',list(pair)) for i,pair in enumerate(itertools.combinations(sorted(RATE),2))]
    for panel,models in groups:
        tids=set.intersection(*[set(products[products.model_id==m].target_profile_id) for m in models])
        for tid in sorted(tids):pids.append(dict(panel=panel,models=';'.join(models),target_profile_id=tid))
        s=products[products.target_profile_id.isin(tids)&products.model_id.isin(models)].copy();s['top1_numeric']=s.top1_product_match.astype(str).str.lower().eq('true').astype(float)
        for metric in ['jsd','observed_product_spearman','top1_numeric']:
            w=s.pivot(index='target_profile_id',columns='model_id',values=metric).dropna(subset=models);v=w[models].to_numpy(float)
            if not len(w):continue
            bt=np.array([v[rng.integers(0,len(w),len(w))].mean(axis=0) for _ in range(5000)])
            for j,m in enumerate(models):
                lo,hi=np.quantile(bt[:,j],[.025,.975]);prows.append(dict(panel=panel,metric=metric,model_id=m,n_common_targets=len(tids),n_defined_paired_targets=len(w),mean=float(v[:,j].mean()),ci_low=lo,ci_high=hi))
            for j,k in itertools.combinations(range(len(models)),2):
                lo,hi=np.quantile(bt[:,j]-bt[:,k],[.025,.975]);pdelta.append(dict(panel=panel,metric=metric,model_a=models[j],model_b=models[k],n_targets=len(w),difference=float(v[:,j].mean()-v[:,k].mean()),ci_low=lo,ci_high=hi))
    pd.DataFrame(prows).to_csv(out/'results/common_product_metrics.csv',index=False);pd.DataFrame(pdelta).to_csv(out/'results/common_product_paired_differences.csv',index=False);pd.DataFrame(pids).to_csv(out/'data/common_product_target_membership.csv',index=False)
    report=dict(status='PASS',same_target_and_position_keys_asserted=True,identical_observed_values_asserted=True,execution_skips_excluded=True,
        bootstrap_replicates=5000,seed=20261025,selection='Intersection of existing frozen panels; no new outcome-independent sampling is claimed.',
        scope='Pairwise targets differ between pairs. Shared panel describes these compatible outputs, including declared proxy/rank roles; does not establish native-editor or all-model training independence.',
        products='JSD, within-observed-product rho and native top1 recovery compared on common targets among four distribution models; Yachie observed-candidate scores not treated as probability distributions.',
        uncertainty='Small intersections and unadjusted intervals are descriptive. Undefined statistic replicates counted; no universal ranking inferred.')
    (out/'results/validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(pd.DataFrame(inventory).query('panel=="all_seven" or panel=="four_conditional"').to_string(index=False))

if __name__=='__main__':main()
