"""Exact overlap of declared panels with locally visible candidate development pools.
No model calls, retraining, or performance calculations. All writes stay here.
"""
import os
from pathlib import Path
from collections import defaultdict, Counter
import csv, hashlib, json, re, sys

sys.stdout.reconfigure(encoding='utf-8')
ROOT=Path(os.environ.get("BE_REPLAY_PROJECT", "."))
OUT=Path(__file__).resolve().parent
MEMBERS=OUT/'available_sequence_memberships_non_CRISPRon.csv'
PREFIX='candidate_overlap_'

def rows(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def js(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def csvout(name,data,fields=None):
    with (OUT/(PREFIX+name+'.csv')).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields or list(data[0]));w.writeheader();w.writerows(data)
def jsonout(name,data):
    (OUT/(PREFIX+name+'.json')).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def dna(s,n):return len(s)==n and set(s)<=set('ACGT')

evidence=[]
def cite(eid,rel,pattern):
    p=ROOT/rel
    found=[]
    for i,s in enumerate(p.read_text(encoding='utf-8-sig',errors='replace').splitlines(),1):
        if re.search(pattern,s):
            evidence.append(dict(evidence_id=eid,path=str(p),line=i,excerpt=s));found.append(i)
    assert found,(eid,rel)

cite('BE40','BE_Endo/repo/README.md',r'40bp\(10bp upstream')
cite('BE20','BE_Endo/repo/BE_endo_Pack/Preprocess/Preprocessing.py',r'def sgRNA|return seq\[10:30\]|PAM=sequence\[30:33\]')
cite('BE_BRANCH','benchmark_core/adapters/run_t2_be_endo_position.py',r'model_data_seq|sequence_context_40nt|seq\[30:33\]')
cite('EP20','EPCNNBE/repo/EPCNNBE_independent_tests.py',r'def load_dataset\(type=.abe., st=4, en=24\)|test_seqs =')
cite('EP30','EPCNNBE/repo/EPCNNBE_web.py',r'onehot_encoding\(seqs_all, 4, 24\)|onehot_encoding\(seqs_all, 0, 30\)|input sequence length')
cite('EP_ADAPTER','benchmark_core/adapters/run_t3_epcnnbe.py',r'sequence_context_30nt|onehot_encoding\(seqs_all, 0, 30\)|events_from_sequences\(target, outcome, 4')
cite('T2_WINDOW','benchmark_core/scripts/prepare_t2_expanded_panels.py',r'seq30.append\(seq\[start - 4 : start \+ 26\]\)|seq40.append\(seq\[start - 10 : start \+ 30\]\)')
cite('T3_WINDOW','benchmark_core/scripts/prepare_t3_v1_expanded_panels.py',r'seq30.append\(seq\[start - 4 : start \+ 26\]|seq40.append\(seq\[start - 10 : start \+ 30\]')
cite('BE_MEMBERS','BE_Endo/repo/model_training/Eff_training_main.py',r'X_train,X_train_bio,Y_train,X_test,X_test_bio,Y_test=item|validation_split=0.2')
csvout('coordinate_evidence',evidence)

t2_manifest=ROOT/'benchmark_core/data/processed/T2_bystander_matrix/panels/T2_expand_panel_manifest.json'
t3_manifest=ROOT/'benchmark_core/data/processed/T3_position_profile_matrix/panels_v1/T3_v1_panel_manifest.json'
package_t2=ROOT/'SCI-data/Fig.3/source_data/benchmark_core_data_processed_T2_bystander_matrix/panels/T2_expand_panel_manifest.json'
package_t3=ROOT/'SCI-data/Fig.4/source_data/benchmark_core_data_processed_T3_position_profile_matrix/panels_v1/T3_v1_panel_manifest.json'
assert sha(t2_manifest)==sha(package_t2)
assert sha(t3_manifest)==sha(package_t3)
m2,m3=js(t2_manifest),js(t3_manifest)
panels=[
 dict(panel='T2_BE_Endo',model='BE_Endo',model_id='be_endo',manifest=t2_manifest,entry=m2['panels']['be_endo'],
      full_len=40,guide_start=10,full_field='sequence_context_40nt',role='T2 formal position-only rank panel',
      package=package_t2.parent/'T2_expand_be_endo.csv',
      prediction=ROOT/'benchmark_core/runs/T2_expand_v0/be_endo/predictions_t2_be_endo_positions.csv',
      raw=ROOT/'benchmark_core/runs/T2_expand_v0/be_endo/raw_t2_be_endo_seq.tsv',raw_field='seq',
      per_target=ROOT/'benchmark_core/formal_results/T2_expand_v0/T2_expanded_position_per_target_metrics.csv'),
 dict(panel='T3_BE_Endo',model='BE_Endo',model_id='be_endo',manifest=t3_manifest,entry=m3['panels']['be_endo'],
      full_len=40,guide_start=10,full_field='sequence_context_40nt',role='T3 formal rank-only panel',
      package=package_t3.parent/'T3_v1_be_endo.csv',
      prediction=ROOT/'benchmark_core/runs/T3_position_profile_v1/be_endo/predictions_t2_be_endo_positions.csv',
      raw=ROOT/'benchmark_core/runs/T3_position_profile_v1/be_endo/raw_t2_be_endo_seq.tsv',raw_field='seq',
      per_target=ROOT/'benchmark_core/formal_results/T3_position_profile_v1/T3_position_per_target_metrics.csv'),
 dict(panel='T3_EPCNNBE',model='EPCNNBE',model_id='epcnnbe',manifest=t3_manifest,entry=m3['panels']['epcnnbe'],
      full_len=30,guide_start=4,full_field='sequence_context_30nt',role='T3 exploratory supplementary proxy panel in formal manifest',
      package=package_t3.parent/'T3_v1_epcnnbe.csv',
      prediction=ROOT/'benchmark_core/runs/T3_position_profile_v1/epcnnbe/predictions_t3_epcnnbe_positions.csv',
      raw=ROOT/'benchmark_core/runs/T3_position_profile_v1/epcnnbe/raw_t3_epcnnbe_outcomes.csv',raw_field='target_sequence_30nt',
      per_target=ROOT/'benchmark_core/formal_results/T3_position_profile_v1/T3_position_per_target_metrics.csv')]

# Verify these current T3 panel files also match the recorded canonical V2 audit.
historical_manifest=ROOT/'SCI-data/Fig.4/source_data/T3_position_profile_coverage_sensitivity_v1_2026-08-25/metadata/T3_coverage_sensitivity_input_manifest.json'
recorded={str(Path(x['path'])):x['sha256'] for x in js(historical_manifest)['canonical_source_tables']}
input_manifest=[]
for p in [MEMBERS,t2_manifest,t3_manifest,package_t2,package_t3,historical_manifest]:
    input_manifest.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)))

members=rows(MEMBERS)
indexes=defaultdict(list);pool_counts=defaultdict(lambda:dict(raw_memberships=0,valid_full_memberships=0,valid_guide_memberships=0,sequences=set()))
for i,m in enumerate(members):
    m['_member_row']=i+2
    length=40 if m['model']=='BE_Endo' else 30
    start=10 if m['model']=='BE_Endo' else 4
    seq=m['sequence'].upper()
    key=(m['model'],m['editor'],m['role'],m['split'],m['source_path'])
    pool_counts[key]['raw_memberships']+=1
    if dna(seq,length):
        pool_counts[key]['valid_full_memberships']+=1
        pool_counts[key]['valid_guide_memberships']+=1
        pool_counts[key]['sequences'].add(seq)
        indexes[(m['model'],'full_context',seq)].append(m)
        indexes[(m['model'],'protospacer20',seq[start:start+20])].append(m)

pool_quality=[]
for key,v in pool_counts.items():
    pool_quality.append(dict(model=key[0],pool_editor=key[1],pool_role=key[2],pool_split=key[3],pool_source_path=key[4],
        raw_memberships=v['raw_memberships'],valid_full_memberships=v['valid_full_memberships'],
        valid_guide_memberships=v['valid_guide_memberships'],unique_full_contexts=len(v['sequences'])))
csvout('pool_sequence_quality',pool_quality)

target_results=[];pair_rows=[];panel_manifest=[];input_checks=[];all_summary=[];split_summary=[]
pair_fields=['panel','model','match_window','target_profile_id','query_editor','pool_editor','editor_class_matches',
             'pool_role','pool_split','query_sequence','matched_full_context','pool_source_path','pool_source_field',
             'pool_source_id','membership_catalog_csv_line','membership_status','interpretation']

for spec in panels:
    path=Path(spec['entry']['path'])
    assert sha(path)==sha(spec['package']),('package_original_mismatch',spec['panel'])
    if spec['panel'].startswith('T3'):
        assert str(path) in recorded,('not_in_canonical_manifest',str(path))
        assert sha(path)==recorded[str(path)],('historical_hash_mismatch',str(path))
    data=rows(path)
    assert len({r['target_profile_id'] for r in data})==len(data)
    expected=spec['entry'].get('rows',spec['entry'].get('selected_targets'))
    assert len(data)==expected
    returned={r['target_profile_id'] for r in rows(spec['prediction'])}
    scored={r['target_profile_id'] for r in rows(spec['per_target']) if r['model_id']==spec['model_id']}
    targetids={r['target_profile_id'] for r in data}
    assert returned<=targetids and scored<=targetids
    raw_map=defaultdict(set)
    with spec['raw'].open(encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f,delimiter='\t' if spec['raw'].suffix=='.tsv' else ','):
            raw_map[r['target_profile_id']].add(r[spec['raw_field']].upper())
    for r in data:
        tid=r['target_profile_id'];seq=r[spec['full_field']].upper()
        if tid in returned:
            assert raw_map[tid]=={seq},('raw_input_disagrees',spec['panel'],tid)
        input_checks.append(dict(panel=spec['panel'],target_profile_id=tid,returned_prediction=tid in returned,
            raw_model_input_found=tid in raw_map,raw_model_input_matches_panel=(raw_map[tid]=={seq}) if tid in raw_map else 'NA'))
    for p in [path,spec['package'],spec['prediction'],spec['raw'],spec['per_target']]:
        input_manifest.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)))
    panel_manifest.append(dict(panel=spec['panel'],model=spec['model'],role=spec['role'],panel_path=str(path),
        formal_panel_manifest=str(spec['manifest']),formal_manifest_entry=spec['entry'],package_copy=str(spec['package']),
        selected_target_n=len(data),returned_prediction_target_n=len(returned),per_target_table_target_n=len(scored),
        source_families=dict(Counter(x['source_family'] for x in data)),editor_classes=dict(Counter(x['editor_class'] for x in data)),
        coordinate_rule=f'full length {spec["full_len"]}; guide 0-based [{spec["guide_start"]}:{spec["guide_start"]+20}]',
        population_label='candidate development-pool exact overlap; weight-specific training membership unknown'))
    these=[]
    for r in data:
        tid=r['target_profile_id'];full=r[spec['full_field']].upper();guide=r['protospacer_20nt'].upper()
        original=r.get('sequence_context_original','').upper();native50=r.get('native_50nt_available','').lower()=='true'
        start=spec['guide_start'];length=spec['full_len'];s50=r.get('sequence_50nt','').upper()
        native_support=native50 and dna(original,len(original)) and full in original and s50 in original
        agreement=dna(full,length) and full[start:start+20]==guide
        adapter_contract=(full[30:33] in {'AGG','CGG','TGG','GGG'}) if spec['model']=='BE_Endo' and len(full)==40 else True
        for mode,query in [('full_context',full),('protospacer20',guide)]:
            reasons=[]
            if mode=='full_context':
                if not dna(full,length):reasons.append('missing_or_ambiguous_full_context')
                if not agreement:reasons.append('guide_coordinate_disagreement')
                if not native_support:reasons.append('original_sequence_does_not_verify_native_flanks')
                if not adapter_contract:reasons.append('model_PAM_input_contract_failed')
            else:
                if not dna(guide,20):reasons.append('missing_or_ambiguous_20nt_guide')
                if dna(full,length) and not agreement:reasons.append('guide_coordinate_disagreement')
                if guide not in original:reasons.append('original_sequence_does_not_verify_guide')
            comparable=not reasons
            matches=indexes[(spec['model'],mode,query)] if comparable else []
            same=[m for m in matches if m['editor']==r['editor_class']]
            for m in matches:
                pair_rows.append(dict(panel=spec['panel'],model=spec['model'],match_window=mode,target_profile_id=tid,
                    query_editor=r['editor_class'],pool_editor=m['editor'],editor_class_matches=m['editor']==r['editor_class'],
                    pool_role=m['role'],pool_split=m['split'],query_sequence=query,matched_full_context=m['sequence'],
                    pool_source_path=m['source_path'],pool_source_field=m['source_field'],pool_source_id=m['source_id'],
                    membership_catalog_csv_line=m['_member_row'],membership_status=m['membership_status'],
                    interpretation='候选开发池重叠；不能推断已证实权重训练泄漏'))
            item=dict(panel=spec['panel'],model=spec['model'],target_profile_id=tid,source_family=r['source_family'],
                source_file=r['source_file'],source_row_key=r['source_row_key'],editor=r['editor'],editor_class=r['editor_class'],
                cell_context=r['cell_context'],match_window=mode,query_sequence=query,
                comparable=comparable,not_comparable_reason=';'.join(reasons),full_context_verified_from_original=native_support,
                stored_guide_agrees_with_contract_window=agreement,returned_prediction=tid in returned,per_target_table=tid in scored,
                same_editor_hit=bool(same) if comparable else 'NA',any_editor_hit=bool(matches) if comparable else 'NA',
                same_editor_train_hit=any(m['role']=='train' for m in same) if comparable else 'NA',
                same_editor_test_hit=any(m['role']=='test' for m in same) if comparable else 'NA',
                same_editor_matching_roles_splits=';'.join(sorted({m['editor']+':'+m['role']+':split='+m['split'] for m in same})),
                same_editor_matching_member_records=len(same) if comparable else 'NA')
            target_results.append(item);these.append((item,same,matches))
    for subset in ('selected_panel','returned_prediction'):
        for mode in ('full_context','protospacer20'):
            items=[x for x in these if x[0]['match_window']==mode and (subset=='selected_panel' or x[0]['returned_prediction'])]
            comparable=[x for x in items if x[0]['comparable']]
            all_summary.append(dict(panel=spec['panel'],model=spec['model'],subset=subset,match_window=mode,
                panel_target_n=len(items),comparable_target_n=len(comparable),NA_target_n=len(items)-len(comparable),
                comparable_unique_query_sequence_n=len({x[0]['query_sequence'] for x in comparable}),
                same_editor_any_role_hit_target_n=sum(bool(x[1]) for x in comparable),
                same_editor_train_hit_target_n=sum(any(m['role']=='train' for m in x[1]) for x in comparable),
                same_editor_test_hit_target_n=sum(any(m['role']=='test' for m in x[1]) for x in comparable),
                any_editor_any_role_hit_target_n=sum(bool(x[2]) for x in comparable),
                interpretation='候选开发池重叠；零命中不证明权重训练独立'))
            for poolkey,pool in pool_counts.items():
                if poolkey[0]!=spec['model']:continue
                pool_editor,pool_role,pool_split,pool_path=poolkey[1:]
                relevant=[x for x in items if x[0]['editor_class']==pool_editor]
                comp=[x for x in relevant if x[0]['comparable']]
                hit=[x for x in comp if any((m['role'],m['split'],m['source_path'])==(pool_role,pool_split,pool_path) for m in x[1])]
                split_summary.append(dict(panel=spec['panel'],model=spec['model'],subset=subset,match_window=mode,
                    pool_editor=pool_editor,pool_role=pool_role,pool_split=pool_split,pool_source_path=pool_path,
                    pool_membership_n=pool['raw_memberships'],same_editor_panel_target_n=len(relevant),
                    comparable_target_n=len(comp),NA_target_n=len(relevant)-len(comp),hit_target_n=len(hit),
                    hit_target_ids=';'.join(x[0]['target_profile_id'] for x in hit),
                    interpretation='候选开发池重叠；角色分开保留，非权重泄漏结论'))

csvout('summary',all_summary)
csvout('by_original_role_split',split_summary)
csvout('target_results',target_results)
csvout('source_match_pairs',pair_rows,pair_fields)
csvout('raw_input_verification',input_checks)
jsonout('panel_manifest',panel_manifest)
jsonout('input_hash_manifest',input_manifest)
jsonout('summary',dict(definition='exact same-orientation A/C/G/T matching; no reverse complement/approximate matching',
    main_match_scope='same editor class; any-editor match counts supplied separately',
    result_unit='target_profile_id; not outcome rows or source split membership records',
    independent_status='UNKNOWN even when no known-pool exact hit',
    weight_link_limitations={'BE_Endo':'known endo development splits do not establish BE_Seq checkpoint training membership',
                             'EPCNNBE':'train filename and tests do not establish each released checkpoint training membership'},
    summaries=all_summary,source_match_pair_rows=len(pair_rows),target_result_rows=len(target_results),model_calls=0))
print(json.dumps(all_summary,ensure_ascii=False,indent=2))
