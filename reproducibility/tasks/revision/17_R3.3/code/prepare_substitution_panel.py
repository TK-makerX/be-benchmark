"""Freeze input-only random guides and substitutions before inference."""
import argparse,hashlib,json,shutil
from pathlib import Path
import numpy as np,pandas as pd
def main():
    p=argparse.ArgumentParser();p.add_argument('--project-root',type=Path,required=True);p.add_argument('--revision-root',type=Path,required=True);a=p.parse_args();out=a.revision_root/'17_R3.3'
    for sub in ['data','results','figures','code']:(out/sub).mkdir(exist_ok=True,parents=True)
    src=a.project_root/'benchmark_core/formal_results/T1_v8_label_blind_known_crispron_clean/data/T1_V8_COMMON_VALID_MODEL_OBSERVATIONS.csv.gz'
    cols=['observation_id','design_id','sequence_30nt','protospacer_20nt','editor','editor_class','cell_context','pam'];d=pd.read_csv(src,usecols=cols).drop_duplicates().sort_values(['protospacer_20nt','sequence_30nt','observation_id']).drop_duplicates('protospacer_20nt')
    assert len(d)==7429 and (d.sequence_30nt.str.slice(4,24)==d.protospacer_20nt).all();rng=np.random.default_rng(20261027);selected=d.iloc[rng.choice(len(d),200,replace=False)].sort_values('protospacer_20nt').reset_index(drop=True);selected.to_csv(out/'data/selected_parent_metadata.csv',index=False);records=[]
    for i,r in selected.iterrows():
        parent=f'R33_{i+1:04}';base=r.sequence_30nt
        for rep in range(4):
            seq=base;pos=-1;before=after=''
            if rep:
                pos=int(rng.integers(20));before=base[4+pos];after=str(rng.choice([b for b in 'ACGT' if b!=before]));seq=base[:4+pos]+after+base[5+pos:]
            records.append(dict(benchmark_id=f'{parent}_{rep}',parent_id=parent,replicate=rep,kind='original' if rep==0 else 'single_substitution',mutation_position_1based=pos+1 if rep else '',reference_base=before,alternate_base=after,original_observation_id=r.observation_id,original_design_id=r.design_id,sequence_30nt=seq,sequence_context_30nt=seq,sequence_context=seq,protospacer_20nt=seq[4:24],pam=seq[24:27],editor=r.editor,editor_class=r.editor_class,cell_context=r.cell_context,has_window_A='A' in seq[6:14]))
    panel=pd.DataFrame(records)
    for _,g in panel.groupby('parent_id'):
        base=g.iloc[0].sequence_30nt
        for r in g.iloc[1:].itertuples():assert sum(x!=y for x,y in zip(base,r.sequence_30nt))==1 and base[:4]==r.sequence_30nt[:4] and base[24:]==r.sequence_30nt[24:]
    target=out/'data/frozen_substitution_panel.csv';panel.to_csv(target,index=False)
    freeze=dict(status='FROZEN',seed=20261027,parent_guides=200,substitutions=600,replicates_per_guide=3,eligible_unique_guides=len(d),metadata_columns=cols,selection_uses_measured_outcomes=False,substitution='Uniform guide position 1–20 and uniform alternative among the other three bases, independently with replacement; PAM/flanks preserved.',source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),panel_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),targets_losing_window_A=int((~panel.has_window_A).sum()),resampling_invalid_mutations=False)
    (out/'data/sampling_freeze.json').write_text(json.dumps(freeze,indent=2),encoding='utf-8');shutil.copy2(a.revision_root/'02_R2.1/endpoint_correction/run_external_models.py',out/'code/run_frozen_scalar_models.py');print(json.dumps(freeze))
if __name__=='__main__':main()
