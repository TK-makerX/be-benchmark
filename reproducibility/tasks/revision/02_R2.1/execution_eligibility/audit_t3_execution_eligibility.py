"""Bounded read-only audit of T3 execution eligibility, excluding CRISPRon.
Reads saved artifacts and mirrors only input guards; never imports model code,
runs inference, or recomputes performance metrics. Writes stay in this folder.
"""
import os
from pathlib import Path
from collections import Counter, defaultdict
import csv,json,hashlib,math,re,sys

sys.stdout.reconfigure(encoding='utf-8')
R=Path(os.environ.get("BE_REPLAY_PROJECT", "."));B=R/'benchmark_core';O=Path(__file__).resolve().parent
RUN=B/'runs/T3_position_profile_v1'
MAN=B/'data/processed/T3_position_profile_matrix/panels_v1/T3_v1_panel_manifest.json'
JOIN=B/'formal_results/T3_position_profile_v1/T3_position_joined_universe_predictions.csv'
def read(p,delim=','):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f,delimiter=delim))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def savecsv(name,data,fields=None):
    with (O/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields or list(data[0]));w.writeheader();w.writerows(data)
def savejson(name,data):(O/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def valid(s,n):return len(s)==n and set(s.upper())<=set('ACGT')
def finite(v):
    try:return math.isfinite(float(v))
    except (TypeError,ValueError):return False

specs={
 'behive_bystander':dict(adapter='run_t2_behive_bystander.py',raw='raw_t2_behive_bystander_stats.json',idcol='target_profile_id',score='Total predicted probability'),
 'forecast_be':dict(adapter='run_t2_forecast_position.py',raw=None,idcol='target_profile_id',score='predicted_position_score'),
 'bedict_v2':dict(adapter='run_t2_bedict_v2.py',raw='raw_t2_bedict_v2_predictions_ABEmax-SpCas9.csv',idcol='seq_id',score='average_predicted_proportion'),
 'deepbe':dict(adapter='run_t2_deepbe.py',raw='raw_deepbe_pilot.csv',idcol='benchmark_id',score='prediction_score'),
 'be_endo':dict(adapter='run_t2_be_endo_position.py',raw='raw_t2_be_endo_seq.tsv',idcol='target_profile_id',score='base1'),
 'yachie_base_editing_prediction':dict(adapter='run_t2_yachie_observed_products.py',raw='raw_t2_yachie_observed_products.csv',idcol='target_profile_id',score='score'),
 'epcnnbe':dict(adapter='run_t3_epcnnbe.py',raw='raw_t3_epcnnbe_outcomes.csv',idcol='target_profile_id',score='outcome_proportion'),
 'deepbaseeditor':dict(adapter='run_t3_deepbaseeditor_proportion.py',raw='raw_t3_deepbaseeditor_proportion.csv',idcol='target_profile_id',score='predicted_outcome_proportion')}
man=json.loads(MAN.read_text(encoding='utf-8'))
known_hashes=json.loads((R/'SCI-data/Fig.4/source_data/T3_position_profile_coverage_sensitivity_v1_2026-08-25/metadata/T3_coverage_sensitivity_input_manifest.json').read_text(encoding='utf-8'))
known={str(Path(x['path'])):x['sha256'] for x in known_hashes['canonical_source_tables']}
assert sha(JOIN)==known[str(JOIN)]
assert sha(MAN)==known[str(MAN)]

assets=[]
def asset(p,role):
    p=Path(p);assets.append(dict(path=str(p),role=role,bytes=p.stat().st_size,sha256=sha(p)))
asset(MAN,'formal panel manifest');asset(JOIN,'canonical all-observed joined evaluation')

evidence=[]
def evidence_lines(key,p,pattern):
    p=Path(p);count=0
    for i,s in enumerate(p.read_text(encoding='utf-8-sig',errors='replace').splitlines(),1):
        if re.search(pattern,s):evidence.append(dict(evidence_id=key,path=str(p),line=i,excerpt=s));count+=1
    assert count,(key,p)
for model,s in specs.items():
    p=B/'adapters'/s['adapter'];asset(p,'adapter source only')
    evidence_lines(model,p,r'skips?\.append|skipped\.append|all_skips\.append|if len\(|if not |if .*editor|model_data_seq|\.predict\(|\.simulation\(|raw_rows.append|stats_rows.append|ensemble_test_request|predict_from_dataframe|run_deepbe_8e|score is None')
evidence_lines('scoring_gate',B/'scripts/score_t3_position_profile.py',r'panel_targets =|universe = observed|how="left"|has_model_prediction.*=|_pred_filled.*=|missing_prediction_filled_with_zero.*=')
evidence_lines('forecast_empty_window',R/'FORECasT-BE/repo/forecast_be.py',r'target_base = EDITOR_TARGET_BASE|ret = \[\(pos, pred\) if|pos_list = POS_MODELS')
asset(R/'FORECasT-BE/repo/forecast_be.py','static empty-window contract')

models=read(R/'BE-Hive-bystander/repo/models.csv')
behive_pairs={(x[k],x['Celltype']) for x in models for k in ('Public base editor','Internal base editor')}
bedict_editors={'ABEmax-NG','ABE8e-NG','ABE8e-SpCas9','ABE8e-SpRY','ABEmax-SpCas9','ABEmax-SpRY'}

def gate(model,r,seen):
    e=r.get('editor','').upper();ec=r.get('editor_class','').upper()
    s30=r.get('sequence_context_30nt','').upper();s40=r.get('sequence_context_40nt','').upper();g=r.get('protospacer_20nt','').upper()
    if model=='behive_bystander':
        if (r.get('editor',''),r.get('cell_context','')) not in behive_pairs:return False,'unsupported editor/cell_context'
        return valid(r.get('sequence_50nt',''),50),'50nt ACGT contract'
    if model=='forecast_be':return valid(g,20) and ec in ('ABE','CBE'),'20nt ACGT plus ABE/CBE class'
    if model=='bedict_v2':return (r.get('t2_model_editor','') or 'ABEmax-SpCas9') in bedict_editors and valid(g,20) and valid(r.get('pam4',''),4),'supported model editor;20nt guide+4nt PAM'
    if model=='deepbe':return ec=='ABE' and valid(s30,30) and 'A' in s30[6:14],'ABE,30nt ACGT,A in source indices6:14'
    if model=='be_endo':
        # This reproduces the adapter's use of editor OR class, not a corrected mapping.
        chosen=(r.get('editor') or r.get('editor_class') or '').upper()
        mapped='ABE' in chosen or 'CBE' in chosen or 'BE4' in chosen or 'C>' in r.get('intended_edit','').upper()
        if not mapped:return False,'no BE_Endo editor mapping'
        return valid(s40,40) and s40[30:33] in {'AGG','CGG','TGG','GGG'},'40nt/NGG indices30:33'
    if model=='yachie_base_editing_prediction':return valid(s30,30) and ('ABEMAX' in e or ec=='ABE' or 'ABE' in e or 'BE4' in e or 'AID' in e or ec=='CBE'),'target-level seq/mapping; observed-outcome guards separately evidenced by skip table'
    if model=='epcnnbe':return ec in ('ABE','CBE') and valid(s30,30) and ('A' if ec=='ABE' else 'C') in s30[6:14],'30nt ACGT and substrate in indices6:14'
    if model=='deepbaseeditor':
        if ec not in ('ABE','CBE') or not valid(s30,30):return False,'unsupported class or 30nt source invalid'
        start,length,w,base=(1,26,5,'A') if ec=='ABE' else (2,25,4,'C')
        seq=s30[start:start+length]
        if not valid(seq,length) or base not in seq[w:w+8]:return False,'model slice or substrate invalid'
        if (ec,seq) in seen:return False,'duplicate input'
        seen.add((ec,seq));return True,'derived ABE26/CBE25 input + substrate + no duplicate'
    raise ValueError(model)

joined=defaultdict(list)
for row in read(JOIN):
    if row['model_id'] in specs:joined[(row['model_id'],row['target_profile_id'])].append(row)

details=[];summaries=[];skipped_positions=[];log_inventory=[]
for model,spec in specs.items():
    panel_path=Path(man['panels'][model]['path']);assert sha(panel_path)==known[str(panel_path)]
    asset(panel_path,'model-specific selected panel');panel=read(panel_path);ids={x['target_profile_id'] for x in panel}
    assert len(ids)==len(panel)==man['panels'][model]['selected_targets']
    d=RUN/model;logs=list(d.rglob('*.log'))
    log_inventory.append(dict(model_id=model,run_dir=str(d),log_files=';'.join(str(p) for p in logs),
        current_v1_log_status='present' if logs else 'no .log file in current v1 model run directory; do not substitute v0 logs'))
    for p in logs:asset(p,'current v1 log')
    skipfiles=list(d.glob('skipped*.csv'));assert len(skipfiles)==1
    skipfile=skipfiles[0];asset(skipfile,'explicit per-row skipped record')
    skiprows=read(skipfile);skipby=defaultdict(list)
    for i,r in enumerate(skiprows,2):
        if r.get('target_profile_id'):skipby[r['target_profile_id']].append((i,r.get('reason','')))
    assert set(skipby)<=ids
    predfiles=sorted(set(d.glob('predictions*position*.csv')))
    predictions=defaultdict(list)
    for p in predfiles:
        asset(p,'saved position prediction output')
        for i,r in enumerate(read(p),2):
            if r.get('target_profile_id'):
                value=r.get('predicted_position_rate') or r.get('predicted_position_score')
                predictions[r['target_profile_id']].append((str(p),i,finite(value)))
    assert set(predictions)<=ids
    rawby=defaultdict(list)
    if spec['raw']:
        p=d/spec['raw'];asset(p,'raw backend output with target IDs')
        data=json.loads(p.read_text(encoding='utf-8')) if p.suffix=='.json' else read(p,'\t' if p.suffix=='.tsv' else ',')
        for i,r in enumerate(data):
            rawby[str(r[spec['idcol']])].append((str(p),('JSON item '+str(i)) if p.suffix=='.json' else ('line '+str(i+2)),finite(r.get(spec['score']))))
        assert set(rawby)<=ids
    seen=set();model_rows=[]
    for r in panel:
        tid=r['target_profile_id'];eligible,guard=gate(model,r,seen);skip=skipby.get(tid,[])
        pred=predictions.get(tid,[]);raw=rawby.get(tid,[])
        finite_raw=any(x[2] for x in raw);finite_pred=any(x[2] for x in pred)
        backend_success=finite_raw or finite_pred
        evalrows=joined.get((model,tid),[])
        fills=[x for x in evalrows if x['missing_prediction_filled_with_zero'].lower()=='true']
        matched=[x for x in evalrows if x['has_model_prediction'].lower()=='true']
        empty_window=False
        if model=='forecast_be':
            targetbase='A' if r['editor_class']=='ABE' else 'C'
            empty_window=all(r['protospacer_20nt'][i]!=targetbase for i in range(2,10))
        if skip and not backend_success:
            state='explicitly_skipped_no_backend_output'
        elif backend_success:
            state='backend_output_confirmed'
        elif empty_window and eligible and not skip:
            state='input_eligible_expected_empty_window_call_completion_unlogged'
        else:state='no_output_without_explicit_skip_unknown'
        row=dict(model_id=model,target_profile_id=tid,panel_path=str(panel_path),source_family=r.get('source_family',''),
            source_row_key=r.get('source_row_key',''),editor=r.get('editor',''),editor_class=r.get('editor_class',''),cell_context=r.get('cell_context',''),
            selected=True,input_eligible_static=eligible,input_gate=guard,backend_success_confirmed_by_finite_output=backend_success,
            backend_success_evidence=';'.join(f'{x[0]}:{x[1]}' for x in raw[:1]) if raw else (';'.join(f'{x[0]}:line {x[1]}' for x in pred[:1])),
            raw_output_rows=len(raw),position_output_rows=len(pred),explicit_skip_record=bool(skip),
            skip_evidence=';'.join(f'{skipfile}:line {i}' for i,_ in skip),skip_reasons=';'.join(sorted({reason for _,reason in skip})),
            no_output_without_explicit_skip=not(raw or pred or skip),forecast_empty_substrate_window=empty_window,
            backend_execution_state=state,all_observed_evaluation_positions=len(evalrows),matched_prediction_positions=len(matched),
            zero_filled_positions=len(fills),all_evaluation_positions_zero_filled=bool(evalrows) and len(fills)==len(evalrows),
            explicit_skip_included_in_accuracy_denominator=bool(skip) and bool(evalrows) and not backend_success,
            current_v1_log_available=bool(logs))
        details.append(row);model_rows.append(row)
        if row['explicit_skip_included_in_accuracy_denominator']:
            for x in evalrows:
                skipped_positions.append(dict(model_id=model,target_profile_id=tid,editor=r['editor'],editor_class=r['editor_class'],
                    protospacer_position=x['protospacer_position'],reference_base=x['reference_base'],
                    observed_position_rate=x['observed_position_rate'],predicted_position_value=x['predicted_position_value'],
                    missing_prediction_filled_with_zero=x['missing_prediction_filled_with_zero'],
                    skip_reason=row['skip_reasons'],skip_evidence=row['skip_evidence']))
    summaries.append(dict(model_id=model,selected_n=len(model_rows),input_eligible_static_n=sum(x['input_eligible_static'] for x in model_rows),
        backend_success_confirmed_n=sum(x['backend_success_confirmed_by_finite_output'] for x in model_rows),
        explicit_skipped_target_n=sum(x['explicit_skip_record'] for x in model_rows),
        no_output_without_explicit_skip_n=sum(x['no_output_without_explicit_skip'] for x in model_rows),
        expected_empty_window_no_call_log_n=sum(x['backend_execution_state']=='input_eligible_expected_empty_window_call_completion_unlogged' for x in model_rows),
        all_observed_evaluated_target_n=sum(x['all_observed_evaluation_positions']>0 for x in model_rows),
        skipped_targets_in_accuracy_denominator_n=sum(x['explicit_skip_included_in_accuracy_denominator'] for x in model_rows),
        skipped_zero_filled_position_n=sum(x['zero_filled_positions'] for x in model_rows if x['explicit_skip_included_in_accuracy_denominator']),
        successful_targets_with_all_observed_positions_zero_filled_n=sum(x['backend_success_confirmed_by_finite_output'] and x['all_evaluation_positions_zero_filled'] for x in model_rows),
        all_zero_filled_target_n=sum(x['all_evaluation_positions_zero_filled'] for x in model_rows),
        skip_reasons=dict(Counter(reason for vals in skipby.values() for _,reason in vals)),current_v1_log_files=len(logs)))

savecsv('T3_execution_eligibility_by_target.csv',details)
savecsv('T3_execution_eligibility_summary.csv',[dict(x,skip_reasons=json.dumps(x['skip_reasons'],ensure_ascii=False)) for x in summaries])
savejson('T3_execution_eligibility_summary.json',dict(scope='Eight non-CRISPRon T3 model roles; finite saved output establishes successful return, not biological validity.',
    no_model_calls=True,no_performance_recalculation=True,models=summaries))
savecsv('T3_explicitly_skipped_positions_in_zero_fill_denominator.csv',skipped_positions)
savecsv('T3_execution_source_evidence.csv',evidence)
savecsv('T3_current_run_log_inventory.csv',log_inventory)
savejson('T3_execution_input_hash_manifest.json',assets)
print(json.dumps(summaries,ensure_ascii=False,indent=2))
