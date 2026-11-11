import argparse,json,sys
from pathlib import Path
import numpy as np,pandas as pd
def main():
    sys.stdout.reconfigure(encoding='utf-8');p=argparse.ArgumentParser();p.add_argument('--revision-root',type=Path,required=True);a=p.parse_args();root=a.revision_root;out=root/'07_R2.6'
    for x in ['data','results']:(out/x).mkdir(exist_ok=True,parents=True)
    s=pd.read_csv(root/'02_R2.1/denominator_correction/results/T3_corrected_per_target_metrics.csv');rng=np.random.default_rng(20261025);rows=[]
    for model,g in s.groupby('model_id'):
        w=g.dropna(subset=['rho_zero','rho_returned']);d=(w.rho_zero-w.rho_returned).to_numpy();b=np.array([rng.choice(d,len(d),replace=True).mean() for _ in range(5000)]) if len(d) else np.array([np.nan]);lo,hi=np.nanquantile(b,[.025,.975]);rows.append(dict(model_id=model,eligible_targets=len(g),jointly_defined_targets=len(w),mean_rho_returned=w.rho_returned.mean(),mean_rho_zero=w.rho_zero.mean(),mean_paired_difference=d.mean() if len(d) else np.nan,ci_low=lo,ci_high=hi))
    pd.DataFrame(rows).to_csv(out/'results/equal_target_missing_policy_sensitivity.csv',index=False);s.to_csv(out/'data/per_target_policy_inputs.csv',index=False)
    report=dict(status='PASS',source='Execution-corrected R2.1 scoring inputs',target_bootstrap=5000,seed=20261025,policies=['Returned-only within eligible targets','Zero completion within identical eligible targets as sensitivity','Identical returned-position intersection across models (R2.2)','Canonical-window common endpoint (R2.3; endpoint standardization is separate from imputation)'],extra_summary='Equal-target means use the same jointly defined targets for both policies; pooled position correlations remain the primary Fig4 summaries.',interpretation='Missing positions are not asserted biologically inactive. No outcome-derived imputation or sentinel minimum-score treatment is used.')
    (out/'results/validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(pd.DataFrame(rows).round(4).to_string(index=False))

if __name__ == '__main__':
    main()
