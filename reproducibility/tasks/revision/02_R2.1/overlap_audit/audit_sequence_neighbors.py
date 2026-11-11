"""Forward, ungapped guide Hamming sensitivity against released CRISPRon development rows.

Distances 0, <=1, <=2 are reported separately. Similarity is neither proof of
leakage nor a proof of independence. Target/sequence filters are fixed before
computing performance; model weights are not changed.
"""
import argparse,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr

def enc(s):return np.array([list(x.encode('ascii')) for x in s],dtype=np.uint8)
def main():
    sys.stdout.reconfigure(encoding='utf-8');p=argparse.ArgumentParser();p.add_argument('--project-root',type=Path,required=True);p.add_argument('--revision-root',type=Path,required=True);a=p.parse_args()
    out=a.revision_root/'02_R2.1/overlap_audit';base=a.project_root/'benchmark_core/formal_results/T1_v8_label_blind_known_crispron_clean/data'
    t1=pd.read_csv(base/'T1_V8_LABEL_BLIND_SELECTED_METADATA.csv.gz');t1['task']='T1';t1['record_id']=t1.observation_id;t1['guide']=t1.protospacer_20nt
    t23=pd.read_csv(out/'CRISPRon_sensitivity_target_membership.csv');t23=t23[t23.backend_executed.astype(str).str.lower()=='true'].copy();t23['guide']=t23.guide_20nt_clean
    ex=pd.read_csv(a.revision_root/'02_R2.1/endpoint_correction/Marquart_ABE_compatible_inference_panel.csv');ex['task']='Marquart127';ex['record_id']=ex.benchmark_id;ex['guide']=ex.guide_20nt
    allq=pd.concat([t1[['task','record_id','editor_class','guide']],t23[['task','record_id','editor_class','guide']],ex[['task','record_id','editor_class','guide']]],ignore_index=True)
    rows=[]
    for ed in ['ABE','CBE']:
        q=allq[allq.editor_class==ed];unique=sorted(set(q.guide));qa=enc(unique);dist={};nearest={}
        for split in ['train','test']:
            f=a.project_root/f'CRISPRon-BE/repo/data_for_model_development_V1.0/{ed}_{split}.tsv.gz'
            ref=sorted(set(pd.read_csv(f,sep='\t',usecols=['refs']).refs));ra=enc(ref);ds=[];ns=[]
            for i in range(0,len(unique),128):
                block=np.rint(cdist(qa[i:i+128],ra,metric='hamming')*20).astype(np.uint8);ix=block.argmin(axis=1)
                ds.extend(block[np.arange(len(block)),ix]);ns.extend(ref[j] for j in ix)
            dist[split]=dict(zip(unique,map(int,ds)));nearest[split]=dict(zip(unique,ns))
        for _,r in q.iterrows():
            rows.append(dict(task=r.task,record_id=r.record_id,editor_class=ed,guide=r.guide,
                 train_min_hamming=dist['train'][r.guide],test_min_hamming=dist['test'][r.guide],
                 development_min_hamming=min(dist['train'][r.guide],dist['test'][r.guide]),
                 nearest_train_guide=nearest['train'][r.guide],nearest_test_guide=nearest['test'][r.guide]))
        print(ed,'queries',len(q),'unique',len(unique),flush=True)
    df=pd.DataFrame(rows);df.to_csv(out/'CRISPRon_guide_neighbor_membership.csv',index=False)
    summaries=[]
    for task,g in df.groupby('task'):
        for scope in ['train','test','development']:
            for radius in [0,1,2]:summaries.append(dict(task=task,reference_scope=scope,radius=radius,total_records=len(g),excluded_records=int((g[scope+'_min_hamming']<=radius).sum()),retained_records=int((g[scope+'_min_hamming']>radius).sum())))
    pd.DataFrame(summaries).to_csv(out/'CRISPRon_guide_neighbor_summary.csv',index=False)
    preds=pd.read_csv(base/'T1_V8_COMMON_VALID_MODEL_OBSERVATIONS.csv.gz',usecols=['observation_id','design_id','model_id','prediction_for_association','measured_total_efficiency'])
    pred=preds.merge(df[df.task=='T1'][['record_id','development_min_hamming']],left_on='observation_id',right_on='record_id',validate='many_to_one')
    metrics=[]
    for radius in [0,1,2]:
        z=pred[pred.development_min_hamming>radius]
        for mid,g in z.groupby('model_id'):
            metrics.append(dict(radius_excluded=radius,model_id=mid,n_observations=len(g),n_designs=g.design_id.nunique(),spearman=float(spearmanr(g.prediction_for_association,g.measured_total_efficiency).correlation),
                interpretation='Ranking association to BE-Hive broad event; CRISPRon reference only, no inference about other model training membership.'))
    pd.DataFrame(metrics).to_csv(out/'T1_neighborhood_exclusion_associations.csv',index=False)
    print(pd.DataFrame(summaries).query('reference_scope=="development"').to_string(index=False));print(pd.DataFrame(metrics).drop(columns='interpretation').to_string(index=False))
    (out/'neighbor_analysis_definition.json').write_text(json.dumps(dict(distance='Hamming on forward 20nt guide, no gaps; ABE/CBE backend matched',radii=[0,1,2],
        warning='Post hoc sensitivity of fixed panel, not an independent random test. Exact full context matches are covered by the earlier audit. No local-alignment or reverse-complement equivalence is asserted.'),indent=2),encoding='utf-8')

if __name__=='__main__':main()
