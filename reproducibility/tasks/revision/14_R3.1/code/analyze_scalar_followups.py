"""All-model calibration display, positive subsets and model agreement.
The frozen T1 panel and already out-of-fold calibrators are reused unchanged.
"""
import argparse,hashlib,json,itertools
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import spearmanr
IDS=['crispron_be','deepbe','forecast_be','deepbaseeditor']
def weighted_rho(x,y):
    _,ix=np.unique(x,return_inverse=True);_,iy=np.unique(y,return_inverse=True)
    def calc(w):
        n=w.sum();wx=np.bincount(ix,weights=w);wy=np.bincount(iy,weights=w)
        rx=(np.cumsum(wx)-wx/2)[ix]-n/2;ry=(np.cumsum(wy)-wy/2)[iy]-n/2
        den=np.sqrt(np.sum(w*rx*rx)*np.sum(w*ry*ry));return float(np.sum(w*rx*ry)/den) if den>0 else np.nan
    return calc
def ci(v):return np.nanquantile(v,[.025,.975]).tolist()
def main():
    p=argparse.ArgumentParser();p.add_argument('--revision-root',type=Path,required=True);a=p.parse_args();root=a.revision_root
    for d in ['14_R3.1','15_R3.2','16_R3.2.1']:
        for q in ['data','results','figures','code']:(root/d/q).mkdir(exist_ok=True,parents=True)
    src=root/'06_R2.5/data/out_of_fold_assay_predictions.csv.gz';w=pd.read_csv(src);y=w.measured_total_efficiency.to_numpy();guides,ix=np.unique(w.calibration_cluster,return_inverse=True);rng=np.random.default_rng(20261026)
    nboot=2000;counts=rng.multinomial(len(guides),np.ones(len(guides))/len(guides),size=nboot);bins=[];subsets=[];near=[];agreement=[];max_error=0.
    for mid in IDS:
        x=w[mid+'__native_score'].to_numpy();z=w[mid+'__linear_clipped'].to_numpy();b=pd.qcut(z,10,labels=False,duplicates='drop')
        for bn in sorted(np.unique(b)):
            mask=b==bn;sumy=np.bincount(ix[mask],weights=y[mask],minlength=len(guides));n=np.bincount(ix[mask],minlength=len(guides));v=(counts@sumy)/(counts@n);lo,hi=ci(v)
            bins.append(dict(model_id=mid,method='guide_disjoint_out_of_fold_linear_clipped',bin=int(bn)+1,n=int(mask.sum()),mean_predicted=z[mask].mean(),mean_observed=y[mask].mean(),ci_low=lo,ci_high=hi))
        for label,mask in [('all',np.ones(len(y),bool)),('measured_gt_0.01',y>.01),('measured_gt_0.10',y>.10)]:
            f=weighted_rho(x[mask],y[mask]);v=[]
            for rep,ct in enumerate(counts):
                weight=ct[ix[mask]];r=f(weight);v.append(r)
                if rep<3:
                    direct=spearmanr(np.repeat(x[mask],weight),np.repeat(y[mask],weight))[0];max_error=max(max_error,abs(direct-r))
            lo,hi=ci(v);subsets.append(dict(model_id=mid,subset=label,n_observations=int(mask.sum()),n_guide_clusters=len(np.unique(ix[mask])),rho=float(spearmanr(x[mask],y[mask])[0]),ci_low=lo,ci_high=hi,finite_bootstrap=int(np.isfinite(v).sum())))
        mask=y<=.005
        for scale,pred in [('out_of_fold_linear_assay_fraction',z)]+([('native_score_against_broad_assay',x)] if mid=='crispron_be' else []):
            for cutoff in [.01,.1]:
                positive=pred>cutoff;n=np.bincount(ix[mask],minlength=len(guides));ns=np.bincount(ix[mask],weights=positive[mask].astype(float),minlength=len(guides));v=(counts@ns)/(counts@n);lo,hi=ci(v)
                if positive[mask].all() or not positive[mask].any():lo=hi=np.nan
                near.append(dict(model_id=mid,prediction_scale=scale,measured_max=.005,prediction_cutoff_strict_gt=cutoff,denominator=int(mask.sum()),n_guides=len(np.unique(ix[mask])),positive_count=int(positive[mask].sum()),positive_fraction=float(positive[mask].mean()),ci_low=lo,ci_high=hi,meaning='Positive call among measured <=0.5%; endpoint-specific empirical assay false-positive rate, not validated clinical specificity'))
    for aa,bb in itertools.combinations(IDS,2):
        x=w[aa+'__native_score'].to_numpy();z=w[bb+'__native_score'].to_numpy();f=weighted_rho(x,z);v=[f(c[ix]) for c in counts];lo,hi=ci(v);agreement.append(dict(model_1=aa,model_2=bb,n_observations=len(w),n_guides=len(guides),rho=float(spearmanr(x,z)[0]),ci_low=lo,ci_high=hi))
    assert max_error<1e-12
    pd.DataFrame(bins).to_csv(root/'14_R3.1/results/all_model_oof_calibration_bins.csv',index=False)
    pd.DataFrame(subsets).to_csv(root/'15_R3.2/results/positive_subset_correlations.csv',index=False)
    pd.DataFrame(near).to_csv(root/'15_R3.2/results/near_zero_positive_calls.csv',index=False)
    pd.DataFrame(agreement).to_csv(root/'16_R3.2.1/results/pairwise_model_agreement.csv',index=False)
    membership=w[['observation_id','design_id','calibration_cluster','fold','measured_total_efficiency']].copy();membership['measured_gt_0.01']=y>.01;membership['measured_gt_0.10']=y>.10;membership['measured_le_0.005']=y<=.005;membership.to_csv(root/'15_R3.2/data/subset_membership.csv.gz',index=False)
    v=dict(status='PASS',n_observations=len(w),n_guide_clusters=len(guides),bootstrap_replicates=nboot,seed=20261026,weighted_spearman_validation_max_error=max_error,source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),native_predictions_unchanged=True,models_refitted=False,calibration='Existing R2.5 guide-disjoint out-of-fold clipped-linear predictions; fixed procedure, no new tuning.',uncertainty='Paired guide-cluster resamples; subset and bin membership fixed; conditional on existing calibration fits.',subset_selection='Measured-efficiency restrictions explicitly requested by reviewer; sensitivity estimands with restricted outcome range, not replacement primary panels.',near_zero='Raw native CRISPRon calls retained separately; other uncalibrated scores are not compared to probability cutoffs. The common four-model comparison uses held-out assay-calibrated predictions.')
    for d in ['14_R3.1','15_R3.2','16_R3.2.1']:(root/d/'results/validation.json').write_text(json.dumps(v,indent=2),encoding='utf-8')
    print(pd.DataFrame(subsets).round(4).to_string(index=False));print(pd.DataFrame(near).drop(columns='meaning').round(4).to_string(index=False));print(pd.DataFrame(agreement).round(4).to_string(index=False))
if __name__=='__main__':main()
