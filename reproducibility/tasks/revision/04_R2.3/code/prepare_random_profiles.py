"""Freeze input-only random samples from complete BE-Hive profile-source key pools.

Outcome values, outcome row counts and activity measurements are not selection
variables. All source products of selected targets are read after the freeze.
"""
import argparse,hashlib,itertools,json,re,sys
from pathlib import Path
import numpy as np
import pandas as pd
DNA=set('ACGT');POS=re.compile(r'^([ACGT])(-?\d+)$')
KEYS=['Name (unique)','Name','Oligo Id'];GUIDES=['gRNA (20nt)','Designed gRNA (NGG orientation)_x']
CONTEXT=['Sequence context (56nt)','Sequence context (35 or 61 nt)','Sequence context prepended_x','Designed 61-bp target site (pathogenic)_x','Designed target_x']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def first(row,names):
    for n in names:
        if n in row and pd.notna(row[n]) and str(row[n]):return str(row[n]).upper()
    return ''
def context50(seq,g):
    if len(seq)==50 and set(seq)<=DNA:return seq
    j=seq.find(g)
    if j>=20 and len(g)==20 and j+30<=len(seq):return seq[j-20:j+30]
    return ''
def main():
    sys.stdout.reconfigure(encoding='utf-8');p=argparse.ArgumentParser();p.add_argument('--project-root',type=Path,required=True);p.add_argument('--revision-root',type=Path,required=True);p.add_argument('--per-seed',type=int,default=200);a=p.parse_args()
    out=a.revision_root/'04_R2.3';[(out/x).mkdir(exist_ok=True,parents=True) for x in ['data','results','logs','figures','response','manuscript']]
    manual=a.project_root/'benchmark_core/data/manual_downloads';models=pd.read_csv(a.project_root/'BE-Hive-bystander/repo/models.csv')
    supported=set(zip(models['Public base editor'],models.Celltype))|set(zip(models['Internal base editor'],models.Celltype))
    pool=[];inventory=[];hashes={};cache={}
    for f in sorted((manual/'be_hive_bystander').glob('*ABE*.pkl')):
        cell,library,editor=f.stem.split('_',2)
        if (editor,cell) not in supported:continue
        ef=manual/'be_hive_efficiency'/(f.stem+'.csv')
        if not ef.exists():continue
        meta=pd.read_csv(ef,dtype=str,usecols=lambda x:x in KEYS+GUIDES+CONTEXT).fillna('');kcol=next(k for k in KEYS if k in meta)
        compare=[c for c in GUIDES+CONTEXT if c in meta]
        nunique=meta.groupby(kcol)[compare].nunique();ambiguous=set(nunique.index[(nunique>1).any(axis=1)])
        original_meta_rows=len(meta);meta=meta[~meta[kcol].isin(ambiguous)].drop_duplicates(kcol)
        # Deserialization makes the dictionary keys available; product values are
        # not inspected or accessed while constructing/fixing the selection.
        profiles=pd.read_pickle(f);cache[f.name]=profiles;available=set(map(str,profiles.keys()))
        key_lookup={str(k):k for k in profiles.keys()};cache[f.name+'_keys']=key_lookup
        hashes[str(f)]=sha(f);hashes[str(ef)]=sha(ef);count=0
        for _,r in meta.iterrows():
            key=str(r[kcol])
            if key not in available:continue
            g=first(r,GUIDES);s=context50(first(r,CONTEXT),g)
            if len(g)!=20 or set(g)-DNA or len(s)!=50 or set(s)-DNA or s[20:40]!=g or s[41:43]!='GG' or 'A' not in g[2:10]:continue
            tid='R23_'+hashlib.sha256((f.name+'|'+key).encode()).hexdigest()[:16]
            pool.append(dict(target_profile_id=tid,source_filename=f.name,source_row_key=key,source_dataset=library,source_family='BE-Hive',editor=editor,editor_class='ABE',cell_context=cell,sequence_50nt=s,protospacer_20nt=g,sequence_context_30nt=s[16:46],sequence_context_40nt=s[10:50],pam4=s[40:44],t2_model_editor='ABEmax-SpCas9',source_relation_to_model_for_t2='source_associated_or_proxy_as_declared'))
            count+=1
        inventory.append(dict(source=f.name,complete_source_keys=len(available),metadata_rows=original_meta_rows,ambiguous_input_keys_excluded=len(ambiguous),input_eligible=count))
        print(f.name,'input-only eligible',count,flush=True)
    pool=pd.DataFrame(pool).sort_values('target_profile_id').reset_index(drop=True);assert not pool.target_profile_id.duplicated().any();assert len(pool)>=a.per_seed
    pool.to_csv(out/'data/input_only_sampling_pool.csv',index=False)
    seeds=[20261025,20261026,20261027];members=[]
    for seed in seeds:
        rows=pool.iloc[np.random.default_rng(seed).choice(len(pool),a.per_seed,replace=False)].copy();rows['sampling_seed']=seed;rows.to_csv(out/'data'/f'frozen_panel_seed_{seed}.csv',index=False);members.append(rows[['target_profile_id','sampling_seed']])
    mem=pd.concat(members,ignore_index=True);mem.to_csv(out/'data/random_seed_membership.csv',index=False);selected=pool[pool.target_profile_id.isin(mem.target_profile_id)].copy();selected.to_csv(out/'data/frozen_union_inference_panel.csv',index=False)
    freeze=dict(seeds=seeds,n_per_seed=a.per_seed,pool_targets=len(pool),union_targets=len(selected),metadata_allowlist=KEYS+GUIDES+CONTEXT,
        gates=['Existing processed profile key with unambiguous input metadata','Native BE-Hive editor/cell supported','Concrete 50nt with exact guide at index 20','20nt guide; NGG PAM','At least one A at guide positions 3–10'],
        unused_selection_variables=['Measured editing efficiency','Product frequency','Product count','Outcome row count','Observed editing window','QC label flags'],
        source_scope='Complete keys in the eight compatible processed BE-Hive ABE profile sources, joined to input-sequence metadata. Author source-processing filters remain.',
        source_sha256=hashes,pool_sha256=sha(out/'data/input_only_sampling_pool.csv'),union_sha256=sha(out/'data/frozen_union_inference_panel.csv'),frozen_before_product_values_accessed=True)
    (out/'data/sampling_freeze.json').write_text(json.dumps(freeze,indent=2),encoding='utf-8');pd.DataFrame(inventory).to_csv(out/'results/full_source_inventory.csv',index=False)
    # Only now access outcomes. Project canonical A>G events at positions 3–10;
    # condition all observed/predicted products on >=1 such event. Other-window
    # outcomes contribute to source audit, not to the conditional endpoint.
    products=[];audit=[];candidates=[]
    known=set()
    for split in ['train','test']:
        known.update(pd.read_csv(a.project_root/f'CRISPRon-BE/repo/data_for_model_development_V1.0/ABE_{split}.tsv.gz',sep='\t',usecols=['refs']).refs)
    for _,r in selected.iterrows():
        d=cache[r.source_filename][cache[r.source_filename+'_keys'][r.source_row_key]];freq=pd.to_numeric(d['Frequency'],errors='coerce');cols=[c for c in d if POS.match(str(c))];agcols=[c for c in cols if c[0]=='A' and 3<=int(c[1:])<=10]
        canon=[]
        for _,v in d.iterrows():canon.append(';'.join(sorted([f'{c}>G' for c in agcols if str(v[c]).upper()=='G'],key=lambda x:int(x[1:x.index('>')]))))
        g=pd.DataFrame({'event':canon,'frequency':freq.to_numpy()});bad=not np.isfinite(freq).all() or (freq<0).any();window_mass=g.loc[g.event.ne(''),'frequency'].sum();status='invalid_source_frequency' if bad else ('no_observed_window_edit' if window_mass<=0 else 'evaluable')
        audit.append(dict(target_profile_id=r.target_profile_id,full_source_outcome_rows=len(d),source_frequency_sum=float(freq.sum()),canonical_window_mass=float(window_mass),observed_status=status,known_crispron_development_guide=r.protospacer_20nt in known))
        if status=='evaluable':
            for event,value in g[g.event.ne('')].groupby('event').frequency.sum().items():products.append(dict(target_profile_id=r.target_profile_id,outcome_events=event,observed_outcome_frequency=float(value/window_mass),source_window_mass=float(window_mass)))
        apos=[j+1 for j,base in enumerate(r.protospacer_20nt) if base=='A' and 3<=j+1<=10]
        for n in range(1,len(apos)+1):
            for positions in itertools.combinations(apos,n):candidates.append(dict(target_profile_id=r.target_profile_id,outcome_profile_id=f'{r.target_profile_id}_{len(candidates)}',outcome_events=';'.join(f'A{j}>G' for j in positions)))
    pd.DataFrame(products).to_csv(out/'data/observed_canonical_window_products.csv',index=False);pd.DataFrame(audit).to_csv(out/'data/post_selection_outcome_audit.csv',index=False);pd.DataFrame(candidates).to_csv(out/'data/sequence_enumerated_canonical_candidates.csv',index=False)
    assert all(sha(Path(k))==v for k,v in hashes.items());print(json.dumps({k:freeze[k] for k in ['pool_targets','union_targets','seeds','n_per_seed']}),flush=True)
if __name__=='__main__':main()
