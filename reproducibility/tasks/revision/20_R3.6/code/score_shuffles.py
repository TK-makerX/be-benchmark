import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score
IDS=['crispron_be','deepbe','forecast_be','deepbaseeditor'];NAMES=['CRISPRon-BE','DeepBE','FORECasT-BE','DeepBaseEditor']
def weighted_auc_setup(y,x):
    _,score_ix=np.unique(x,return_inverse=True);positive=y==1
    def calc(w):
        pos=np.bincount(score_ix,weights=w*positive);neg=np.bincount(score_ix,weights=w*(~positive));den=pos.sum()*neg.sum()
        return float(np.sum(pos*(np.cumsum(neg)-.5*neg))/den) if den else np.nan
    return calc
def main():
    p=argparse.ArgumentParser();p.add_argument('--project-root',type=Path,required=True);p.add_argument('--revision-root',type=Path,required=True);a=p.parse_args();out=a.revision_root/'20_R3.6';panel=pd.read_csv(out/'data/frozen_shuffle_inference_panel.csv');units=pd.read_csv(out/'data/frozen_selected_units.csv');coverage=[]
    for mid in IDS:
        files=list((out/'inference/raw'/mid).glob('predictions*.csv'));assert len(files)==1;z=pd.read_csv(files[0]);z=z[z.task_type=='efficiency'];assert not z.benchmark_id.duplicated().any();panel=panel.merge(z[['benchmark_id','predicted_efficiency']].rename(columns={'predicted_efficiency':mid}),on='benchmark_id',how='left',validate='one_to_one');coverage.append(dict(model_id=mid,input_rows=len(panel),finite_scores=int(panel[mid].notna().sum()),missing_scores=int(panel[mid].isna().sum())))
    base=panel[panel.replicate==0].set_index('parent_id');assert base[IDS].notna().all().all();original_source=a.project_root/'benchmark_core/formal_results/T4_high_low_classification_v2_1_context_unit_ci_revision/source_data';all_units=pd.read_csv(original_source/'T4_V2_1_CONTEXT_UNITS_JOINED.csv.gz');all_units=all_units[all_units.v2_1_clean_label.notna()];replay=[];metrics=[];draws=[];pairrows=[];per_model=[];max_error=0.;original_auc_ref=pd.read_csv(original_source/'T4_auroc_by_model.csv').set_index('model')
    for mid,name in zip(IDS,NAMES):
        err=float(np.max(np.abs(units.parent_id.map(base[mid]).to_numpy()-units[name].to_numpy())));assert err<1e-5;auc=roc_auc_score(all_units.v2_1_clean_label,all_units[name]);assert abs(auc-original_auc_ref.loc[name,'auroc'])<1e-12;replay.append(dict(model_id=mid,n_units=500,max_abs_error=err,original_full_5108_auc=auc,sampled_500_original_auc=roc_auc_score(units.v2_1_clean_label,units.parent_id.map(base[mid])),sampled_500_original_ap=average_precision_score(units.v2_1_clean_label,units.parent_id.map(base[mid]))))
    for rep in range(1,4):
        z=panel[panel.replicate==rep].set_index('parent_id');u=units.copy()
        for mid in IDS:u[mid+'__original']=u.parent_id.map(base[mid]);u[mid+'__shuffled']=u.parent_id.map(z[mid])
        for mid in IDS:
            h=u[u[mid+'__shuffled'].notna()];yy=h.v2_1_clean_label;xx=h[mid+'__original'];zz=h[mid+'__shuffled'];per_model.append(dict(model_id=mid,replicate=rep,n_units=len(h),original_auc=roc_auc_score(yy,xx),shuffled_auc=roc_auc_score(yy,zz),delta_auc=roc_auc_score(yy,zz)-roc_auc_score(yy,xx),scope='per-model available pairs; descriptive sensitivity, not a common-model comparison'))
        u['replicate']=rep;u['all_four_complete']=u[[m+'__shuffled' for m in IDS]].notna().all(axis=1);pairrows.append(u);g=u[u.all_four_complete].copy();y=g.v2_1_clean_label.to_numpy(int);guide,ix=np.unique(g.protospacer_20nt,return_inverse=True);rng=np.random.default_rng(20261028);counts=rng.multinomial(len(guide),np.ones(len(guide))/len(guide),size=2000)
        for mid in IDS:
            x=g[mid+'__original'].to_numpy();z=g[mid+'__shuffled'].to_numpy();fx=weighted_auc_setup(y,x);fz=weighted_auc_setup(y,z);v=[]
            for i,c in enumerate(counts):
                weight=c[ix];a0=fx(weight);a1=fz(weight);v.append([a0,a1,a1-a0])
                if i<5:max_error=max(max_error,abs(a0-roc_auc_score(y,x,sample_weight=weight)),abs(a1-roc_auc_score(y,z,sample_weight=weight)))
            v=np.array(v);values=[roc_auc_score(y,x),roc_auc_score(y,z),roc_auc_score(y,z)-roc_auc_score(y,x)]
            row=dict(model_id=mid,replicate=rep,n_units=len(g),n_guides=len(guide),n_high=int(y.sum()),n_low=int((y==0).sum()),original_ap=average_precision_score(y,x),shuffled_ap=average_precision_score(y,z))
            for j,name in enumerate(['original_auc','shuffled_auc','delta_auc']):lo,hi=np.nanquantile(v[:,j],[.025,.975]);row[name]=float(values[j]);row[name+'_lo']=lo;row[name+'_hi']=hi
            metrics.append(row);b=pd.DataFrame(v,columns=['original_auc','shuffled_auc','delta_auc']);b['model_id']=mid;b['replicate']=rep;draws.append(b)
    assert max_error<1e-12
    panel.to_csv(out/'data/all_native_predictions.csv.gz',index=False);pd.concat(pairrows).to_csv(out/'data/paired_unit_scores.csv.gz',index=False);pd.DataFrame(metrics).to_csv(out/'results/shuffle_discrimination.csv',index=False);pd.DataFrame(replay).to_csv(out/'results/original_score_replay.csv',index=False);pd.DataFrame(coverage).to_csv(out/'results/execution_coverage.csv',index=False);pd.concat(draws).to_csv(out/'results/paired_guide_bootstrap.csv.gz',index=False);pd.DataFrame(per_model).to_csv(out/'results/per_model_available_pair_sensitivity.csv',index=False)
    report=dict(status='PASS',sampled_units=500,class_balance={'low':250,'high':250},original_full_panel_units=5108,shuffle_replicates=3,bootstrap_replicates=2000,weighted_auc_validation_error=max_error,native_checkpoints_unchanged=True,all_original_scores_reproduced=True,primary_comparison='Within replicate, identical complete original/shuffled units across all four models; no missing-score imputation.',labels='Original frozen experimental high/low labels retained solely as a stress-test reference; no claim they are experimentally observed outcomes of shuffled guides.',scope='Shuffling disrupts position and motif order jointly while retaining total guide composition. Residual discrimination can reflect composition, retained flanks and sample context; this is not a causal feature attribution.')
    (out/'results/validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(pd.DataFrame(metrics)[['model_id','replicate','n_units','original_auc','shuffled_auc','delta_auc','delta_auc_lo','delta_auc_hi']].round(4).to_string(index=False));print(pd.DataFrame(replay).round(8).to_string(index=False));print(pd.DataFrame(coverage).to_string(index=False))
if __name__=='__main__':main()
