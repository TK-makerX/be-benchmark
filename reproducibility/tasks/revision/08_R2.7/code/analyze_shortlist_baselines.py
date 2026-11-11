"""Reproduce frozen shortlist metrics; add untuned sequence-only heuristics.

Heuristic ties are averaged analytically, not broken using measured outcomes.
Groups are benchmark contexts, not independent experimental platforms.
"""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np,pandas as pd
MODELS=['crispron_be','forecast_be'];HEUR=['window_substrate_count','window_center_weight','gc_balance','random_expectation'];MET=['ndcg_at_k','precision_at_k','regret_top1','top1_best_hit']
def expected_metrics(y,h,score):
    order=np.argsort(-score,kind='stable');ys=y[order];hs=h[order];ss=score[order];ideal=np.sort(y)[-10:][::-1];discount=1/np.log2(np.arange(2,12));ideal_dcg=ideal@discount
    groups=np.r_[0,np.flatnonzero(np.diff(ss)!=0)+1,len(ss)];dcg=prec=0.;topmean=None;bestprob=None
    for start,end in zip(groups[:-1],groups[1:]):
        if start>=10:break
        stop=min(end,10);dcg+=ys[start:end].mean()*discount[start:stop].sum();prec+=hs[start:end].mean()*(stop-start)
        if start==0:topmean=ys[start:end].mean();bestprob=float(np.mean(ys[start:end]==y.max()))
    return dict(ndcg_at_k=float(dcg/ideal_dcg),precision_at_k=float(prec/10),regret_top1=float(y.max()-topmean),top1_best_hit=bestprob)
def actual_metrics(y,h,score,design):
    order=np.lexsort((design,-score));ys=y[order];top=order[:10];ideal=np.sort(y)[-10:][::-1];discount=1/np.log2(np.arange(2,12));return dict(ndcg_at_k=float(y[top]@discount/(ideal@discount)),precision_at_k=float(h[top].mean()),regret_top1=float(y.max()-ys[0]),top1_best_hit=float(ys[0]==y.max()))
def holm(p):
    p=np.asarray(p);order=np.argsort(p);adj=np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p))));out=np.empty_like(p);out[order]=np.minimum(adj,1);return out
def main():
    sys.stdout.reconfigure(encoding='utf-8');p=argparse.ArgumentParser();p.add_argument('--project-root',type=Path,required=True);p.add_argument('--revision-root',type=Path,required=True);a=p.parse_args();root=a.revision_root;out=root/'08_R2.7';out28=root/'09_R2.8'
    for base in [out,out28]:
        for x in ['data','results','figures','response','manuscript']:(base/x).mkdir(exist_ok=True,parents=True)
    base=a.project_root/'benchmark_core/formal_results/T7_unique_design_recommendation_v2';memberfile=base/'T7_unique_design_membership.csv';scorefile=base/'T7_unique_design_model_scores.csv';designfile=a.project_root/'benchmark_core/data/unified_matrix_v5/design_master_v5.csv'
    cols=['recommendation_group_id','design_id','design_group_instance_id','measured_efficiency','high_efficiency_label','source_family','editor','editor_class','cell_context'];d=pd.read_csv(memberfile,usecols=cols);assert len(d)==438768;assert d.recommendation_group_id.nunique()==39;assert not d.design_group_instance_id.duplicated().any();d['high_efficiency_label']=d.high_efficiency_label.astype(str).str.lower().eq('true')
    scores=pd.read_csv(scorefile,usecols=['design_group_instance_id','model_id','predicted_recommendation_score']);scores=scores[scores.model_id.isin(MODELS)];wide=scores.pivot(index='design_group_instance_id',columns='model_id',values='predicted_recommendation_score');d=d.merge(wide,on='design_group_instance_id',validate='one_to_one');assert len(d)==438768 and d[MODELS].notna().all().all()
    designs=pd.read_csv(designfile,usecols=['design_id','protospacer_20nt','sequence_30nt'],dtype=str);d=d.merge(designs,on='design_id',validate='many_to_one');assert d.protospacer_20nt.str.len().eq(20).all();assert (d.protospacer_20nt==d.sequence_30nt.str.slice(4,24)).all();assert set(d.editor_class)=={'ABE','CBE'}
    d['gc_balance']=-d.protospacer_20nt.map(lambda s:abs((s.count('G')+s.count('C'))/20-.5));weights=np.exp(-.5*((np.arange(3,11)-6.5)/2)**2)
    sub=np.array([[b==('A' if ed=='ABE' else 'C') for b in seq[2:10]] for seq,ed in zip(d.protospacer_20nt,d.editor_class)],dtype=float);d['window_substrate_count']=sub.sum(axis=1);d['window_center_weight']=sub@weights;d['random_expectation']=0.
    d.to_csv(out/'data/complete_candidate_scores_and_heuristics.csv.gz',index=False,compression='gzip');records=[];cache=[];guides,guide_ix=np.unique(d.protospacer_20nt,return_inverse=True);d['_guide_index']=guide_ix
    for gid,g in d.groupby('recommendation_group_id'):
        g=g.sort_values('design_id');y=g.measured_efficiency.to_numpy();h=g.high_efficiency_label.to_numpy(float);ids=g.design_id.to_numpy();threshold=np.sort(y)[-int(np.ceil(.1*len(g)))];assert np.array_equal(h,y>=threshold)
        for method in MODELS+HEUR:
            x=g[method].to_numpy();m=actual_metrics(y,h,x,ids) if method in MODELS else expected_metrics(y,h,x);records.append(dict(recommendation_group_id=gid,source_family=g.source_family.iloc[0],editor=g.editor.iloc[0],editor_class=g.editor_class.iloc[0],cell_context=g.cell_context.iloc[0],method=method,n_candidates=len(g),**m))
        cache.append(dict(gid=gid,source=g.source_family.iloc[0],guide=g._guide_index.to_numpy(),y=y,observed_order=np.argsort(-y,kind='stable'),orders={m:np.argsort(-g[m].to_numpy(),kind='stable') for m in MODELS}))
    results=pd.DataFrame(records);results.to_csv(out/'results/shortlist_per_group.csv',index=False)
    original=pd.read_csv(base/'T7_unique_design_model_group_scores.csv');checks=[]
    for model in MODELS:
        ours=results[results.method==model].set_index('recommendation_group_id');old=original[original.model_id==model].set_index('recommendation_group_id').loc[ours.index]
        for metric in MET:
            v=old[metric].astype(float).to_numpy();err=float(np.abs(ours[metric].to_numpy()-v).max());checks.append(dict(model=model,metric=metric,max_abs_difference=err));assert err<1e-10
    rng=np.random.default_rng(20261025);groups=sorted(results.recommendation_group_id.unique());B=10000;index=rng.integers(0,39,(B,39));signs=rng.choice([-1,1],size=(100000,39));summaries=[];contrasts=[];modelpairs=[]
    for metric in MET:
        w=results.pivot(index='recommendation_group_id',columns='method',values=metric).loc[groups];mean=w.mean()
        for method in MODELS+HEUR:
            value=w[method].to_numpy();lo,hi=np.quantile(value[index].mean(axis=1),[.025,.975]);summaries.append(dict(method=method,metric=metric,n_groups=39,mean=mean[method],ci_low=lo,ci_high=hi))
        for model in MODELS:
            for heuristic in HEUR:
                diff=(w[model]-w[heuristic]).to_numpy();lo,hi=np.quantile(diff[index].mean(axis=1),[.025,.975]);null=(signs@diff)/39;pval=(1+np.sum(np.abs(null)>=abs(diff.mean())-1e-14))/(len(null)+1);contrasts.append(dict(model=model,baseline=heuristic,metric=metric,difference=diff.mean(),ci_low=lo,ci_high=hi,groups_model_better=int((diff<0 if metric=='regret_top1' else diff>0).sum()),groups_tied=int((diff==0).sum()),p_signflip=pval))
        diff=(w[MODELS[0]]-w[MODELS[1]]).to_numpy();lo,hi=np.quantile(diff[index].mean(axis=1),[.025,.975]);null=(signs@diff)/39;pval=(1+np.sum(np.abs(null)>=abs(diff.mean())-1e-14))/(len(null)+1);modelpairs.append(dict(metric=metric,difference=diff.mean(),ci_low=lo,ci_high=hi,p_signflip=pval,leave_one_group_out_min=float(min((diff.sum()-diff)/38)),leave_one_group_out_max=float(max((diff.sum()-diff)/38))))
    for records in [contrasts,modelpairs]:
        adjusted=holm([r['p_signflip'] for r in records])
        for row,pa in zip(records,adjusted):row['p_holm']=float(pa)
    pd.DataFrame(summaries).to_csv(out/'results/shortlist_summary.csv',index=False);pd.DataFrame(contrasts).to_csv(out/'results/shortlist_paired_contrasts.csv',index=False);pd.DataFrame(checks).to_csv(out/'results/original_metric_replication.csv',index=False);pd.DataFrame(modelpairs).to_csv(out28/'results/model_paired_comparison.csv',index=False)
    results.groupby(['source_family','method'])[MET].mean().reset_index().to_csv(out28/'results/source_stratified_metrics.csv',index=False)
    results.groupby(['editor_class','method'])[MET].mean().reset_index().to_csv(out28/'results/editor_class_stratified_metrics.csv',index=False)
    results.groupby(['source_family','method']).size().rename('groups').reset_index().to_csv(out28/'results/source_group_counts.csv',index=False)
    # Keep the same 80% guide subset across every context. These are sensitivity
    # distributions, not confidence intervals or new biological experiments.
    sensitivity=[]
    for rep in range(200):
        keep=np.zeros(len(guides),bool);keep[rng.choice(len(guides),int(.8*len(guides)),replace=False)]=True;rows=[]
        for g in cache:
            mask=keep[g['guide']];y=g['y'];n=int(mask.sum());oi=g['observed_order'][mask[g['observed_order']]];highcut=y[oi[int(np.ceil(.1*n))-1]];ideal=y[oi[:10]];disc=1/np.log2(np.arange(2,12));best=ideal[0]
            for model in MODELS:
                order=g['orders'][model];top=order[mask[order]][:10];rows.append(dict(model=model,ndcg_at_k=float(y[top]@disc/(ideal@disc)),precision_at_k=float(np.mean(y[top]>=highcut)),regret_top1=float(best-y[top[0]]),top1_best_hit=float(y[top[0]]==best)))
        v=pd.DataFrame(rows).groupby('model')[MET].mean()
        for metric in MET:sensitivity.append(dict(replicate=rep,metric=metric,crispron=float(v.loc[MODELS[0],metric]),forecast=float(v.loc[MODELS[1],metric]),difference=float(v.loc[MODELS[0],metric]-v.loc[MODELS[1],metric])))
    pd.DataFrame(sensitivity).to_csv(out28/'results/shared_guide_subsampling.csv',index=False)
    report=dict(status='PASS',native_results_reproduced=True,n_candidates=438768,n_groups=39,source_groups={'BE-Hive':37,'CRISPRon-BE':2},heuristics='Untuned window substrate count (3–10), substrate-weighted fixed central Gaussian (center 6.5, SD 2), closeness to 50% guide GC.',heuristic_ties='Exact expectation over uniform order within score ties; no measured-label tie breaking.',random='Analytical random-order expectation; original Fig8 Monte Carlo baseline retained separately.',bootstrap='10000 paired group resamples, conditional descriptive intervals for these benchmark groups.',permutation='100000 paired group sign flips; Holm correction over 32 model–baseline metric contrasts and separately four model–model contrasts. Group exchangeability is an exploratory assumption, not proof of platform-level independence.',dependence='Groups share designs and comprise only two source families; source/editor stratification, leave-one-group-out means, and 200 shared 80% guide subsamples supplement the group analysis.',subsampling='Observed high-efficiency labels and oracle quantities recomputed within each retained candidate group. Sampling masks shared across models and across all contexts for a guide; central ranges are sensitivities, not confidence intervals.',source_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [memberfile,scorefile,designfile]})
    for dest in [out,out28]:(dest/'results/validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(pd.DataFrame(summaries).pivot(index='method',columns='metric',values='mean').round(5).to_string());print(pd.DataFrame(modelpairs).round(5).to_string(index=False))
if __name__=='__main__':main()
