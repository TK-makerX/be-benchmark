# Analysis and asset index

Run the commands from the repository root with the environment in [requirements-analysis.txt](../requirements-analysis.txt). Each replay writes to a new directory. It uses the exact frozen observations and predictions for the analysis in the table.

```bash
python -B reproducibility/scripts/replay.py T3 --work-dir benchmark-replay/T3
```

| Replay | Inputs and frozen outputs | ZIP assets |
|---|---|---|
| [T1](T1/README.md) | 10,000 label-blind observations, four model outputs, observation and design association, clustered intervals and V8.1 calibration | tasks-T1-T2.zip, additional-task-inputs.zip |
| [T2](T2/README.md) | Expanded product and positional assays, source identifiers, model-specific input panels and native predictions | tasks-T1-T2.zip |
| [T3](T3/README.md) | Position matrices, exact returned/eligible position keys, per-target predictions and corrected denominators | tasks-T3-T4.zip, revision-overlap-external.zip, additional-task-inputs.zip |
| [T4](T4/README.md) | V2.1 context units, frozen labels, reference and split assignment, paired and top-k bootstrap | tasks-T3-T4.zip |
| [T5](T5/README.md) | Original T1 observations and scores, three joint matched-unit contrasts, prior sensitivity and current V1.2.1 tables | tasks-T1-T2.zip, tasks-T5-T6.zip |
| [T6](T6/README.md) | Exact pilot inputs, timing status, prediction/skip evidence and operational applicability tables | tasks-T5-T6.zip, revision-overlap-external.zip |
| [T7](T7/README.md) | 503,545 input observation records, 438,768 design-group instances, frozen scores and group metrics | tasks-T7.zip, T7-observation-inputs.zip |

| Revision replay | Frozen analysis | ZIP assets in addition to baseline inputs |
|---|---|---|
| `neighbors` | Forward 20-nt guide exclusion, exact or within one or two substitutions of released CRISPRon development sequences | revision-overlap-external.zip, matrix-source-inputs.zip |
| `external-endpoint`, `cross-source` | Marquart source tables and native forecasts for the canonical window, with exact target membership and paired association | revision-overlap-external.zip, additional-task-inputs.zip, matrix-source-inputs.zip |
| `common` | Identical T2 product and T3 returned-position keys across included models | revision-common-random.zip, revision-overlap-external.zip |
| `random` | Three input-only random panels, candidate products, native predictions and canonical-window scores | revision-common-random.zip, additional-task-inputs.zip |
| `calibration`, `scalar-followups` | Guide-disjoint cross-fitting, calibration bins, positive-only sensitivity, near-zero calls and pairwise model agreement | revision-calibration-robustness.zip |
| `missing` | Returned-only, zero-completion and equal-target sensitivities using the corrected eligibility manifest | revision-calibration-robustness.zip, revision-overlap-external.zip |
| `shortlist`, `paired` | Simple rules, matched group contrasts, source/editor strata and shared-guide subsampling | tasks-T7.zip, matrix-v5.zip, additional-task-inputs.zip |
| `source-audit` | Source, assay and scale inventory of the full matrix | matrix-v5.zip, revision-calibration-robustness.zip |
| `substitution` | Matched, single-base substitutions with frozen native predictions and replay of original scores | revision-perturbations.zip |
| `shuffle` | Matched original and shuffled sequences, held-out discrimination, execution coverage and original-score replay | revision-perturbations.zip |

The revision directories retain their source analysis names. Each `data/` or `raw/` directory contains selected keys, predictions, or observed values used by its numeric script. Each `results/` directory contains the frozen result tables. The source accession index covers datasets that remain at their original repositories. [Upstream catalog](../data/upstream_catalog.csv) · [File-level provenance](../payload_manifest.csv)
