import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
def main():
    p=argparse.ArgumentParser();p.add_argument('--project-root',type=Path,required=True);p.add_argument('--revision-root',type=Path,required=True);a=p.parse_args();out=a.revision_root/'09_R2.8';d=pd.read_csv(a.project_root/'benchmark_core/formal_results/T7_unique_design_recommendation_v2/T7_unique_design_model_group_scores.csv');ids=sorted(d.recommendation_group_id.unique());rng=np.random.default_rng(20261025);ix=rng.integers(0,len(ids),(10000,len(ids)));signs=rng.choice([-1,1],(100000,len(ids)));rows=[]
    for metric in ['ndcg_at_k','precision_at_k','recall_at_k','enrichment_at_k','mrr','regret_top1','topk_best_hit','top1_best_hit']:
        w=d.pivot(index='recommendation_group_id',columns='model_id',values=metric).loc[ids].astype(float);x=(w.crispron_be-w.forecast_be).to_numpy();lo,hi=np.quantile(x[ix].mean(axis=1),[.025,.975]);null=signs@x/len(ids);pv=(1+(np.abs(null)>=abs(x.mean())-1e-14).sum())/(1+len(null));rows.append(dict(metric=metric,crispron_mean=w.crispron_be.mean(),forecast_mean=w.forecast_be.mean(),difference=x.mean(),ci_low=lo if np.ptp(x)>0 else np.nan,ci_high=hi if np.ptp(x)>0 else np.nan,p_signflip=pv,n_groups=len(ids),note='Both models recovered 0/39 at rank one; degenerate bootstrap omitted.' if metric=='top1_best_hit' else 'Conditional, exploratory group-level comparison.'))
    order=np.argsort([r['p_signflip'] for r in rows]);values=np.array([rows[j]['p_signflip'] for j in order])*(len(rows)-np.arange(len(rows)));adj=np.minimum(1,np.maximum.accumulate(values))
    for j,v in zip(order,adj):rows[j]['p_holm_eight_metrics']=v
    pd.DataFrame(rows).to_csv(out/'results/native_pairwise_complete_metrics.csv',index=False)
    s=pd.read_csv(out/'results/shared_guide_subsampling.csv');s.groupby('metric').difference.agg(mean='mean',central_025=lambda x:x.quantile(.025),central_975=lambda x:x.quantile(.975),fraction_positive=lambda x:(x>0).mean()).to_csv(out/'results/shared_guide_subsampling_summary.csv');print(pd.DataFrame(rows).round(5).to_string(index=False));print(s.groupby('metric').difference.quantile([.025,.975]).to_string())
if __name__=='__main__':main()
