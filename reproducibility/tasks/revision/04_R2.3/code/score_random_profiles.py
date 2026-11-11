"""Score the same canonical-window endpoint on three input-only random panels."""
import argparse,itertools,json,re,sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
IDS=['behive_bystander','forecast_be','bedict_v2','deepbe','crispron_be','be_endo','yachie_base_editing_prediction']
RATE=['behive_bystander','bedict_v2','deepbe','crispron_be'];YACHIE=IDS[-1]
EVENT=re.compile(r'^A(\d+)>G$')
def project(event):
    positions=[]
    for part in str(event).split(';'):
        m=EVENT.match(part)
        if m and 3<=int(m.group(1))<=10:positions.append(int(m.group(1)))
    return ';'.join('A%d>G'%j for j in sorted(set(positions)))
def rho(x,y):return float(spearmanr(x,y).correlation) if len(x)>1 and np.ptp(x)>0 and np.ptp(y)>0 else np.nan
def wf(x,y):
    _,ix=np.unique(x,return_inverse=True);_,iy=np.unique(y,return_inverse=True)
    def run(w):
        n=w.sum();wx=np.bincount(ix,weights=w);wy=np.bincount(iy,weights=w);rx=(wx.cumsum()-wx/2)[ix]-n/2;ry=(wy.cumsum()-wy/2)[iy]-n/2;den=np.sqrt((w*rx*rx).sum()*(w*ry*ry).sum());return (w*rx*ry).sum()/den if den>0 else np.nan
    return run
def interval(v):
    v=np.asarray(v);v=v[np.isfinite(v)];return (*np.quantile(v,[.025,.975]),len(v)) if len(v) else (np.nan,np.nan,0)
def jsd(p,q):
    m=(p+q)/2;s=0.
    for x in [p,q]:
        ok=x>0;s+=np.sum(x[ok]*np.log(x[ok]/m[ok]))/2
    return float(s)
def main():
    sys.stdout.reconfigure(encoding='utf-8');p=argparse.ArgumentParser();p.add_argument('--revision-root',type=Path,required=True);a=p.parse_args();out=a.revision_root/'04_R2.3';sys.path.insert(0,str(a.revision_root/'03_R2.2/code'))
    panel=pd.read_csv(out/'data/frozen_union_inference_panel.csv');members=pd.read_csv(out/'data/random_seed_membership.csv');audit=pd.read_csv(out/'data/post_selection_outcome_audit.csv');observed=pd.read_csv(out/'data/observed_canonical_window_products.csv');candidates=pd.read_csv(out/'data/sequence_enumerated_canonical_candidates.csv')
    ids=set(panel.target_profile_id);assert len(ids)==len(panel);source_hashes=json.loads((out/'data/sampling_freeze.json').read_text(encoding='utf-8'))
    products={};availability=[];all_projected=[]
    for model in RATE+[YACHIE]:
        files=list((out/'raw'/model).glob('predictions*_outcomes.csv'));assert len(files)==1,(model,files);d=pd.read_csv(files[0]);d['event']=d.predicted_outcome_events.map(project);col='predicted_outcome_score' if model==YACHIE else 'predicted_outcome_frequency';d['value']=pd.to_numeric(d[col],errors='coerce')
        for tid in sorted(ids):
            s=d[d.target_profile_id==tid];g=s[s.event.ne('')].groupby('event').value.sum(min_count=1);mass=float(g.sum());good=bool(len(s) and len(g) and np.isfinite(g).all() and (g>=0).all() and mass>0)
            availability.append(dict(target_profile_id=tid,model_id=model,native_rows=len(s),projected_positive_mass=mass,status='evaluable' if good else 'unavailable_conditional_output'))
            if good:
                products[(tid,model)]=g/mass if model in RATE else g
                for event,value in products[(tid,model)].items():all_projected.append(dict(target_profile_id=tid,model_id=model,event=event,value=value,scale='conditional_window_product_probability' if model in RATE else 'native_candidate_score'))
    direct={}
    for model in ['forecast_be','be_endo']:
        files=list((out/'raw'/model).glob('predictions*position*.csv'));assert len(files)==1;d=pd.read_csv(files[0]);col='predicted_position_score';d=d[(d.reference_base=='A')&d.protospacer_position.between(3,10)]
        assert not d.duplicated(['target_profile_id','protospacer_position']).any();direct[model]=d.set_index(['target_profile_id','protospacer_position'])[col].astype(float)
    obs={tid:g.set_index('outcome_events').observed_outcome_frequency for tid,g in observed.groupby('target_profile_id')}
    lookup=panel.set_index('target_profile_id');posrows=[];prows=[]
    for tid in sorted(ids):
        if tid not in obs:continue
        positions=[j+1 for j,c in enumerate(lookup.loc[tid].protospacer_20nt) if c=='A' and 3<=j+1<=10]
        for j in positions:
            event=f'A{j}>G';v=obs[tid];row=dict(target_profile_id=tid,sequence_context_30nt=lookup.loc[tid].sequence_context_30nt,protospacer_position=j,observed=float(sum(x for e,x in v.items() if event in e.split(';'))))
            for model in IDS:
                if model in direct:row[model]=direct[model].get((tid,j),np.nan)
                else:
                    s=products.get((tid,model));row[model]=sum(x for e,x in s.items() if event in e.split(';')) if s is not None else np.nan
            posrows.append(row)
        universe=sorted(candidates[candidates.target_profile_id==tid].outcome_events);ov=obs[tid].reindex(universe,fill_value=0).to_numpy();assert abs(ov.sum()-1)<1e-8
        for model in RATE+[YACHIE]:
            if (tid,model) not in products:continue
            s=products[(tid,model)];pv=s.reindex(universe,fill_value=0).to_numpy();assert np.isfinite(pv).all()
            top=int(np.argmax(pv));osupport=ov>0
            row=dict(target_profile_id=tid,sequence_context_30nt=lookup.loc[tid].sequence_context_30nt,model_id=model,top1=float(ov[top]==ov.max()),observed_product_rho=rho(pv[osupport],ov[osupport]),n_observed_products=int(osupport.sum()),n_sequence_candidates=len(universe))
            if model in RATE:assert abs(pv.sum()-1)<1e-8;row['jsd']=jsd(pv,ov)
            prows.append(row)
    positions=pd.DataFrame(posrows);product_metrics=pd.DataFrame(prows);positions.to_csv(out/'data/common_window_position_scoring.csv.gz',index=False,compression='gzip');product_metrics.to_csv(out/'data/common_window_product_per_target.csv',index=False);pd.DataFrame(all_projected).to_csv(out/'data/projected_model_products.csv.gz',index=False,compression='gzip');pd.DataFrame(availability).to_csv(out/'results/output_availability.csv',index=False)
    rng=np.random.default_rng(20261025);metrics=[];deltas=[];pmetrics=[];pdeltas=[];counts=[];strata=[]
    audit=audit.set_index('target_profile_id');known=audit.known_crispron_development_guide.astype(str).str.lower().eq('true')
    for seed,mm in members.groupby('sampling_seed'):
        for scope in ['all_selected','known_crispron_exact_excluded']:
            tids=set(mm.target_profile_id)
            if scope!='all_selected':tids-=set(known[known].index)
            w=positions[positions.target_profile_id.isin(tids)].dropna(subset=IDS).copy();complete=set(w.target_profile_id)
            expected={t:sum(c=='A' for c in lookup.loc[t].protospacer_20nt[2:10]) for t in complete}
            complete={t for t in complete if len(w[w.target_profile_id==t])==expected[t]};w=w[w.target_profile_id.isin(complete)]
            counts.append(dict(seed=seed,scope=scope,selected_targets=len(tids),observed_window_evaluable=sum(t in obs for t in tids),complete_seven_model_targets=len(complete),identical_position_rows=len(w)))
            if len(w)>1:
                clusters,ix=np.unique(w.sequence_context_30nt,return_inverse=True);fns=[wf(w[model].to_numpy(),w.observed.to_numpy()) for model in IDS];boots=np.empty((2000,len(IDS)))
                for b in range(2000):
                    weight=rng.multinomial(len(clusters),np.ones(len(clusters))/len(clusters))[ix]
                    for j,f in enumerate(fns):boots[b,j]=f(weight)
                points=[]
                for j,model in enumerate(IDS):
                    point=rho(w[model].to_numpy(),w.observed.to_numpy());points.append(point);lo,hi,n=interval(boots[:,j]);r=dict(seed=seed,scope=scope,model_id=model,n_targets=len(complete),n_design_clusters=len(clusters),n_positions=len(w),rho=point,ci_low=lo,ci_high=hi,finite_bootstrap=n)
                    if model in RATE:r.update(mae=float(np.abs(w[model]-w.observed).mean()),rmse=float(np.sqrt(np.mean((w[model]-w.observed)**2))))
                    metrics.append(r)
                for j,k in itertools.combinations(range(len(IDS)),2):
                    lo,hi,n=interval(boots[:,j]-boots[:,k]);deltas.append(dict(seed=seed,scope=scope,model_a=IDS[j],model_b=IDS[k],delta_rho=points[j]-points[k],ci_low=lo,ci_high=hi,finite_bootstrap=n))
            for metric,mods in [('jsd',RATE),('observed_product_rho',RATE+[YACHIE]),('top1',RATE+[YACHIE])]:
                w=product_metrics[product_metrics.target_profile_id.isin(tids)].pivot(index='target_profile_id',columns='model_id',values=metric).reindex(columns=mods).dropna();arr=w.to_numpy();design=lookup.loc[w.index].sequence_context_30nt.to_numpy();cl,ix=np.unique(design,return_inverse=True)
                if not len(w):continue
                bt=[]
                for b in range(2000):
                    weight=rng.multinomial(len(cl),np.ones(len(cl))/len(cl))[ix];bt.append(np.sum(arr*weight[:,None],axis=0)/weight.sum())
                bt=np.asarray(bt)
                for j,model in enumerate(mods):
                    lo,hi,n=interval(bt[:,j]);pmetrics.append(dict(seed=seed,scope=scope,metric=metric,model_id=model,n_common_targets=len(w),n_design_clusters=len(cl),mean=float(arr[:,j].mean()),ci_low=lo,ci_high=hi))
                for j,k in itertools.combinations(range(len(mods)),2):
                    lo,hi,n=interval(bt[:,j]-bt[:,k]);pdeltas.append(dict(seed=seed,scope=scope,metric=metric,model_a=mods[j],model_b=mods[k],difference=float((arr[:,j]-arr[:,k]).mean()),ci_low=lo,ci_high=hi,n_targets=len(w)))
        s=lookup.loc[mm.target_profile_id].join(audit);strata.append(dict(seed=seed,selected=len(s),source_outcome_rows_min=int(s.full_source_outcome_rows.min()),source_outcome_rows_median=float(s.full_source_outcome_rows.median()),source_outcome_rows_max=int(s.full_source_outcome_rows.max()),targets_above_previous_50_product_cap=int((s.full_source_outcome_rows>50).sum()),no_observed_window_event=int((s.observed_status=='no_observed_window_edit').sum()),known_crispron_matches=int(s.known_crispron_development_guide.astype(str).str.lower().eq('true').sum())))
        print('scored',seed,flush=True)
    for name,rows in [('random_position_metrics',metrics),('random_position_paired_differences',deltas),('random_product_metrics',pmetrics),('random_product_paired_differences',pdeltas),('random_panel_denominators',counts),('random_panel_characteristics',strata)]:pd.DataFrame(rows).to_csv(out/'results'/(name+'.csv'),index=False)
    report=dict(status='PASS',sample_fixed_before_outcome_values_used=True,n_sampling_pool=source_hashes['pool_targets'],seeds=source_hashes['seeds'],n_per_seed=source_hashes['n_per_seed'],no_outcome_row_truncation=True,endpoint='Canonical A>G product/position distribution at guide 3–10 conditional on at least one canonical-window event among available processed source products.',all_models_identical_position_keys=True,bootstrap='2000 paired 30nt-design-cluster resamples per seed and scope',proxies='DeepBE ABE8e and BEDICT ABEmax used on BE-Hive ABE/ABE-CP1040 data; no native-editor equivalence claimed.',source_scope='Input-eligible BE-Hive 12kChar sources; random within available processed pools. Author preprocessing and source association remain.',yachie='All 2^k-1 canonical 3–10 products enumerated from sequence before outcomes; native scores used for ranking only.',selection_missingness='Selected targets with no observed canonical-window edit or unavailable model conditioning remain in audit; no replacement sampling.',intervals='Unadjusted descriptive percentile intervals; repeated seeds test sampling sensitivity, not independent biological replication.')
    (out/'results/validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(pd.DataFrame(counts).to_string(index=False))
if __name__=='__main__':main()
