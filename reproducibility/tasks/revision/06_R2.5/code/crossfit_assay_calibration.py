"""Guide-disjoint five-fold calibration of the existing BE-Hive assay readout.

No model fitting, refitting of native checkpoints, external-label pooling or
test-fold model selection. Linear and isotonic procedures are both reported.
"""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression
IDS=['crispron_be','deepbe','forecast_be','deepbaseeditor']
def evaluate(y,x):
    return dict(rmse=float(np.sqrt(np.mean((y-x)**2))),mae=float(np.mean(np.abs(y-x))),r2=float(1-np.sum((y-x)**2)/np.sum((y-y.mean())**2)),rho=float(spearmanr(y,x).correlation))
def main():
    sys.stdout.reconfigure(encoding='utf-8');p=argparse.ArgumentParser();p.add_argument('--project-root',type=Path,required=True);p.add_argument('--revision-root',type=Path,required=True);a=p.parse_args();out=a.revision_root/'06_R2.5'
    for x in ['data','results','figures','response','manuscript']:(out/x).mkdir(exist_ok=True,parents=True)
    f=a.project_root/'benchmark_core/formal_results/T1_v8_label_blind_known_crispron_clean/data/T1_V8_COMMON_VALID_MODEL_OBSERVATIONS.csv.gz'
    d=pd.read_csv(f,low_memory=False);assert len(d)==40000;assert not d.duplicated(['observation_id','model_id']).any();obs=d[d.model_id==IDS[0]].copy().sort_values('observation_id');seqcol='sequence_30nt' if 'sequence_30nt' in obs else 'sequence_context_30nt';keys=obs[seqcol].astype(str);assert keys.str.len().eq(30).all()
    # Cluster on the 20nt guide, including alternate flanks in the same fold.
    obs['calibration_cluster']=keys.str.slice(4,24);guides=sorted(obs.calibration_cluster.unique());rng=np.random.default_rng(20261025);perm=rng.permutation(len(guides));foldmap={guides[int(idx)]:int(j%5) for j,idx in enumerate(perm)};obs['fold']=obs.calibration_cluster.map(foldmap)
    keep=['observation_id','design_id','calibration_cluster','fold','measured_total_efficiency']+[c for c in ['editor','cell_context','source_dataset','source_family'] if c in obs];w=obs[keep].copy().reset_index(drop=True);y=w.measured_total_efficiency.to_numpy(float);assert np.isfinite(y).all();weights=[];params=[];predcols=[]
    w[['observation_id','design_id','calibration_cluster','fold']].to_csv(out/'data/fixed_guide_disjoint_folds.csv',index=False)
    for model in IDS:
        s=d[d.model_id==model].set_index('observation_id').loc[w.observation_id];x=s.prediction_for_association.to_numpy(float);assert np.isfinite(x).all();w[model+'__native_score']=x
        for method in ['linear_clipped','isotonic','training_mean']:
            z=np.full(len(w),np.nan)
            for fold in range(5):
                train=w.fold.ne(fold).to_numpy();test=~train;assert not set(w.loc[train,'calibration_cluster'])&set(w.loc[test,'calibration_cluster'])
                if method=='linear_clipped':
                    fit=LinearRegression().fit(x[train,None],y[train]);z[test]=np.clip(fit.predict(x[test,None]),0,1);extra=dict(intercept=float(fit.intercept_),slope=float(fit.coef_[0]))
                elif method=='isotonic':
                    fit=IsotonicRegression(y_min=0,y_max=1,increasing=True,out_of_bounds='clip').fit(x[train],y[train]);z[test]=fit.predict(x[test]);extra=dict(n_thresholds=len(fit.X_thresholds_));pd.DataFrame(dict(score=fit.X_thresholds_,assay_fraction=fit.y_thresholds_)).to_csv(out/'data'/f'{model}_isotonic_fold_{fold}.csv',index=False)
                else:z[test]=y[train].mean();extra=dict(training_mean=float(y[train].mean()))
                params.append(dict(model_id=model,method=method,fold=fold,n_train=int(train.sum()),n_test=int(test.sum()),**extra))
            w[model+'__'+method]=z;predcols.append((model,method,model+'__'+method))
        if model=='crispron_be':predcols.append((model,'native_unadapted',model+'__native_score'))
    cl,ix=np.unique(w.calibration_cluster,return_inverse=True);bootweights=[rng.multinomial(len(cl),np.ones(len(cl))/len(cl))[ix] for _ in range(2000)];metrics=[];bootstrap={};bins=[]
    for model,method,col in predcols:
        x=w[col].to_numpy();m=evaluate(y,x);bm=[]
        for weight in bootweights:
            mean=np.average(y,weights=weight);mse=np.average((y-x)**2,weights=weight);mae=np.average(np.abs(y-x),weights=weight);var=np.average((y-mean)**2,weights=weight);bm.append([np.sqrt(mse),mae,1-mse/var])
        bm=np.asarray(bm);bootstrap[(model,method)]=bm;r=dict(model_id=model,method=method,n_observations=len(w),n_guide_clusters=len(cl),**m)
        for k,name in enumerate(['rmse','mae','r2']):lo,hi=np.quantile(bm[:,k],[.025,.975]);r[name+'_ci_low']=lo;r[name+'_ci_high']=hi
        metrics.append(r)
        for b,g in pd.DataFrame(dict(predicted=x,observed=y,bin=pd.qcut(pd.Series(x).rank(method='first'),10,labels=False))).groupby('bin'):
            bins.append(dict(model_id=model,method=method,decile=int(b)+1,n=len(g),mean_predicted=g.predicted.mean(),mean_observed=g.observed.mean()))
    diffs=[]
    for method in ['linear_clipped','isotonic','training_mean']:
        change=bootstrap[('crispron_be',method)]-bootstrap[('crispron_be','native_unadapted')]
        for k,name in enumerate(['rmse','mae','r2']):lo,hi=np.quantile(change[:,k],[.025,.975]);v1=next(x for x in metrics if x['model_id']=='crispron_be' and x['method']==method)[name];v0=next(x for x in metrics if x['model_id']=='crispron_be' and x['method']=='native_unadapted')[name];diffs.append(dict(model_id='crispron_be',method=method,metric=name,difference=v1-v0,ci_low=lo,ci_high=hi,definition='out_of_fold calibrated minus native unadapted'))
    w.to_csv(out/'data/out_of_fold_assay_predictions.csv.gz',index=False,compression='gzip');pd.DataFrame(params).to_csv(out/'results/fold_fit_parameters.csv',index=False);pd.DataFrame(metrics).to_csv(out/'results/calibration_metrics.csv',index=False);pd.DataFrame(diffs).to_csv(out/'results/crispron_paired_calibration_changes.csv',index=False);pd.DataFrame(bins).to_csv(out/'results/calibration_bins.csv',index=False)
    native=next(m for m in metrics if m['model_id']=='crispron_be' and m['method']=='native_unadapted');assert abs(native['r2']-(-.2089))<.0001;assert abs(native['rho']-.472201414486)<1e-9
    report=dict(status='PASS',folds=5,seed=20261025,guide_disjoint=True,fold_assignment_outcome_independent=True,all_observations_predicted_once_out_of_fold=True,n_observations=len(w),n_guide_clusters=len(cl),bootstrap_replicates=2000,native_checkpoints_unchanged=True,source_sha256=hashlib.sha256(f.read_bytes()).hexdigest(),scope='Empirical calibration to the original broad BE-Hive assay endpoint; does not establish native canonical-window accuracy or transfer to another assay.',calibrators='Clipped ordinary least squares and nondecreasing isotonic; both fixed before evaluation, neither selected using held-out metrics.',native_score_note='CRISPRon native values are used for reproducing original errors. Other native contracts are not recast as assay probabilities; all four receive explicit empirical calibration.',uncertainty='Percentile guide-cluster bootstrap conditional on fitted cross-validation models; does not refit calibrators in each resample.')
    (out/'results/validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(pd.DataFrame(metrics)[['model_id','method','rmse','mae','r2','rho']].round(5).to_string(index=False))
if __name__=='__main__':main()
