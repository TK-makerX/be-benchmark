"""Test released CRISPRon label identities; construct an explicitly separate endpoint."""
import argparse,json,hashlib,sys
from pathlib import Path
import numpy as np
import pandas as pd

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap=argparse.ArgumentParser();ap.add_argument('--project-root',type=Path,required=True);ap.add_argument('--revision-root',type=Path,required=True)
    a=ap.parse_args();out=a.revision_root/'02_R2.1/endpoint_correction'
    out.mkdir(exist_ok=True); summary=[];details=[];hashes={}
    for ed,base,to in [('ABE','A','G'),('CBE','C','T')]:
        p=a.project_root/f'CRISPRon-BE/repo/data_for_model_development_V1.0/{ed}_train.tsv.gz'
        hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
        d=pd.read_csv(p,sep='\t');d['window_event']=[any(r[i]==base and o[i]==to for i in range(2,10)) for r,o in zip(d.refs,d.outcomes)]
        d['guide_event']=[any(r[i]==base and o[i]==to for i in range(20)) for r,o in zip(d.refs,d.outcomes)]
        d['window_reads']=d.newc*d.window_event;d['guide_reads']=d.newc*d.guide_event
        d['window_freq']=d.outcomefreq*d.window_event; d['all_freq']=d.outcomefreq
        g=d.groupby(['source','refs','surro_target'],sort=True).agg(rows=('newc','size'),reads=('newc','sum'),total=('total','first'),
            editingCounts=('editingCounts','first'),label_pct=('editingeff','first'),window_reads=('window_reads','sum'),guide_reads=('guide_reads','sum'),
            window_freq=('window_freq','sum'),all_freq=('all_freq','sum'),label_values=('editingeff','nunique')).reset_index()
        g['editor_class']=ed;g['window_label_abs_delta_pct']=(g.window_freq-g.label_pct).abs()
        g['window_count_abs_delta']=(g.window_reads-g.editingCounts).abs()
        g['represented_count_minus_total']=g.reads-g.total
        details.append(g)
        for source,s in g.groupby('source'):
            summary.append(dict(editor_class=ed,source=source,targets=len(s),
                window_sum_frequency_matches_efficiency_label_n=int((s.window_label_abs_delta_pct<1e-6).sum()),
                window_sum_frequency_max_abs_delta_pct=float(s.window_label_abs_delta_pct.max()),
                window_count_matches_editingCounts_n=int((s.window_count_abs_delta<1e-6).sum()),
                all_frequency_sums_to_100_n=int((abs(s.all_freq-100)<1e-6).sum()),
                represented_counts_equal_total_n=int((abs(s.represented_count_minus_total)<1e-6).sum()),
                labels_consistent_within_target=bool((s.label_values==1).all())))
    pd.DataFrame(summary).to_csv(out/'CRISPRon_training_head_endpoint_identity_by_source.csv',index=False)
    pd.concat(details).to_csv(out/'CRISPRon_training_head_endpoint_identity_per_target.csv.gz',index=False,compression='gzip')
    candidates=pd.read_csv(a.revision_root/'02_R2.1/external_feasibility/Marquart2021_ABE134_external_inference_candidate.csv')
    products=pd.read_csv(a.revision_root/'02_R2.1/external_feasibility/Marquart2021_candidate_outcome_audit.csv')
    products=products[products.editor=='ABEmax'].copy()
    products['canonical_3_10_event']=[any(r[i]=='A' and o[i]=='G' for i in range(2,10)) for r,o in zip(products.reference_20nt,products.outcome_20nt)]
    products['window_frequency']=products.full_proportion_including_wildtype*products.canonical_3_10_event
    rates=products.groupby('source_site_id').window_frequency.sum()
    candidates['canonical_3_10_rate']=candidates.source_site_id.map(rates)
    assert candidates.canonical_3_10_rate.notna().all()
    assert (candidates.canonical_3_10_rate<=candidates.total_efficiency+1e-12).all()
    candidates['window_substrate_present']=candidates.guide_20nt.str.slice(2,10).str.contains('A')
    candidates['endpoint_definition']='Probability of at least one A>G at guide positions 3-10 among author-retained reads including WT; other-window products stay in denominator; no renormalization.'
    candidates['benchmark_id']=candidates.candidate_id
    candidates['sequence_context_30nt']=candidates.sequence_30nt
    candidates['sequence_context']=candidates.sequence_30nt
    candidates['protospacer_20nt']=candidates.guide_20nt
    candidates['editor']='ABEmax';candidates['source_family']='Marquart2021';candidates['source_dataset']='ABEmax_integrated_library'
    candidates['label_value']=candidates.canonical_3_10_rate
    candidates['source_relation_to_model']='known_CRISPRon_exact_development_nonoverlap_other_training_membership_unknown'
    candidates.to_csv(out/'Marquart_ABE134_endpoint_panel.csv',index=False)
    valid=candidates[candidates.window_substrate_present].copy()
    valid.to_csv(out/'Marquart_ABE_compatible_inference_panel.csv',index=False)
    products.to_csv(out/'Marquart_ABE134_canonical_window_products.csv',index=False)
    info=dict(training_source_hashes=hashes,candidate_n=len(candidates),input_window_compatible_n=len(valid),
              exclusions=candidates.loc[~candidates.window_substrate_present,['candidate_id','guide_20nt']].to_dict('records'),
              changed_vs_full_guide_event_n=int((abs(candidates.canonical_3_10_rate-candidates.total_efficiency)>1e-10).sum()),
              source_selection='Prior author comparison subset; not an outcome-independent population sample.',
              independence='CRISPRon known exact training and test matches excluded; full independence of other models unknown.',
              endpoints='Separate canonical-window transfer challenge; cannot replace BE-Hive broad any-edit endpoint.')
    (out/'endpoint_identity_manifest.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    print(pd.DataFrame(summary).to_string(index=False));print(json.dumps(info,indent=2))

if __name__=='__main__':main()
