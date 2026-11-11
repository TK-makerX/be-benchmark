"""Composition-preserving stress test with frozen original high/low labels."""
import argparse,hashlib,json,shutil
from pathlib import Path
import numpy as np,pandas as pd
def main():
    p=argparse.ArgumentParser();p.add_argument('--project-root',type=Path,required=True);p.add_argument('--revision-root',type=Path,required=True);a=p.parse_args();out=a.revision_root/'20_R3.6'
    for name in ['data','results','code','figures']:(out/name).mkdir(exist_ok=True,parents=True)
    src=a.project_root/'benchmark_core/formal_results/T4_high_low_classification_v2_1_context_unit_ci_revision/source_data/T4_V2_1_CONTEXT_UNITS_JOINED.csv.gz';d=pd.read_csv(src);pool=d[d.v2_1_clean_label.notna()].sort_values('unit_id');assert len(pool)==5108
    rng=np.random.default_rng(20261028);selected=pd.concat([g.iloc[rng.choice(len(g),250,replace=False)] for _,g in pool.groupby('v2_1_clean_label')]).sort_values('unit_id').reset_index(drop=True);assert len(selected)==500 and selected.v2_1_clean_label.sum()==250
    seqfile=a.project_root/'benchmark_core/formal_results/T1_v8_label_blind_known_crispron_clean/data/T1_V8_COMMON_VALID_MODEL_OBSERVATIONS.csv.gz';seqs=pd.read_csv(seqfile,usecols=['design_id','sequence_30nt','protospacer_20nt']).drop_duplicates();assert not seqs.design_id.duplicated().any();selected=selected.merge(seqs,on='design_id',validate='many_to_one');assert selected.sequence_30nt.notna().all()
    designs=selected.sort_values(['sequence_30nt','unit_id']).drop_duplicates('sequence_30nt').reset_index(drop=True);seqid={s:f'R36_{i+1:04}' for i,s in enumerate(designs.sequence_30nt)};selected['parent_id']=selected.sequence_30nt.map(seqid);selected.to_csv(out/'data/frozen_selected_units.csv',index=False);rows=[]
    for r in designs.itertuples():
        parent=seqid[r.sequence_30nt]
        for rep in range(4):
            seq=r.sequence_30nt if not rep else r.sequence_30nt[:4]+''.join(rng.permutation(list(r.protospacer_20nt)))+r.sequence_30nt[24:]
            assert sorted(seq[4:24])==sorted(r.protospacer_20nt) and seq[:4]==r.sequence_30nt[:4] and seq[24:]==r.sequence_30nt[24:]
            rows.append(dict(benchmark_id=f'{parent}_{rep}',parent_id=parent,replicate=rep,kind='original' if not rep else 'composition_preserving_shuffle',sequence_30nt=seq,sequence_context_30nt=seq,sequence_context=seq,protospacer_20nt=seq[4:24],pam=seq[24:27],editor=r.editor,editor_class=r.editor_class,cell_context=r.cell_context,original_guide=r.protospacer_20nt,has_window_A='A' in seq[6:14],unchanged_after_shuffle=seq==r.sequence_30nt))
    f=out/'data/frozen_shuffle_inference_panel.csv';panel=pd.DataFrame(rows);panel.to_csv(f,index=False)
    freeze=dict(status='FROZEN',seed=20261028,pool_clean_units=5108,selected_units=500,low=250,high=250,unique_native_contexts=len(designs),shuffle_replicates=3,predictions_used_in_sampling=False,label_definition='Original frozen design-disjoint reference quartile labels; never recomputed for shuffled sequences.',shuffle='Uniform permutation of the 20nt protospacer; exact A/C/G/T counts, PAM and flanks preserved; no resampling for model compatibility.',nonbaseline_rows_without_window_A=int(((panel.replicate>0)&~panel.has_window_A).sum()),source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),panel_sha256=hashlib.sha256(f.read_bytes()).hexdigest())
    (out/'data/sampling_freeze.json').write_text(json.dumps(freeze,indent=2),encoding='utf-8');shutil.copy2(a.revision_root/'02_R2.1/endpoint_correction/run_external_models.py',out/'code/run_frozen_scalar_models.py');print(json.dumps(freeze))
if __name__=='__main__':main()
