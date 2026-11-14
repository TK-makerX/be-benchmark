# Analysis and asset index

Run the commands from the repository root with the environment in [requirements-analysis.txt](../requirements-analysis.txt). Each replay writes to a new directory. It uses the exact frozen observations and predictions for the analysis in the table.

```bash
python -B reproducibility/scripts/replay.py position-specific-editing --work-dir benchmark-replay/position-specific-editing
```

| Analysis | Manuscript ID | Inputs and frozen outputs | ZIP assets |
|---|---|---|---|
| [Scalar editing activity](scalar-editing-activity/README.md) | T1 | 10,000 label-blind observations, four model outputs, observation and design association, clustered intervals and V8.1 calibration | scalar-activity-and-bystander-outcomes.zip, additional-task-inputs.zip |
| [Bystander and product outcome distribution](bystander-outcome-distribution/README.md) | T2 | Expanded product and positional assays, source identifiers, model-specific input panels and native predictions | scalar-activity-and-bystander-outcomes.zip |
| [Position-specific editing](position-specific-editing/README.md) | T3 | Position matrices, exact returned/eligible position keys, per-target predictions and corrected denominators | position-specific-editing-and-activity-screening.zip, revision-overlap-external.zip, additional-task-inputs.zip |
| [High/low editing activity screening](high-low-activity-screening/README.md) | T4 | V2.1 context units, frozen labels, reference and split assignment, paired and top-k bootstrap | position-specific-editing-and-activity-screening.zip |
| [Ranking transferability](ranking-transferability/README.md) | T5 | Scalar activity observations and scores, three joint matched-unit contrasts, prior sensitivity and current V1.2.1 tables | scalar-activity-and-bystander-outcomes.zip, ranking-transfer-and-model-reproducibility.zip |
| [Model reproducibility and applicability](model-reproducibility-and-applicability/README.md) | T6 | Exact pilot inputs, timing status, prediction/skip evidence and operational applicability tables | ranking-transfer-and-model-reproducibility.zip, revision-overlap-external.zip |
| [Editor and guide recommendation](editor-and-guide-recommendation/README.md) | T7 | 503,545 input observation records, 438,768 design-group instances, frozen scores and group metrics | editor-and-guide-recommendation.zip, recommendation-observation-inputs.zip |

The replay command accepts the task directory names. The manuscript IDs also work as aliases.

| Revision replay | Frozen analysis | ZIP assets in addition to baseline inputs |
|---|---|---|
| `neighbors` | Forward 20-nt guide exclusion, exact or within one or two substitutions of released CRISPRon development sequences | revision-overlap-external.zip, matrix-source-inputs.zip |
| `external-endpoint`, `cross-source` | Marquart source tables and native forecasts for the canonical window, with exact target membership and paired association | revision-overlap-external.zip, additional-task-inputs.zip, matrix-source-inputs.zip |
| `common` | Identical product keys and returned position keys across included models | revision-common-random.zip, revision-overlap-external.zip |
| `random` | Three input-only random panels, candidate products, native predictions and canonical-window scores | revision-common-random.zip, additional-task-inputs.zip |
| `calibration`, `scalar-followups` | Guide-disjoint cross-fitting, calibration bins, positive-only sensitivity, near-zero calls and pairwise model agreement | revision-calibration-robustness.zip |
| `missing` | Returned-only, zero-completion and equal-target sensitivities using the corrected eligibility manifest | revision-calibration-robustness.zip, revision-overlap-external.zip |
| `shortlist`, `paired` | Simple rules, matched group contrasts, source/editor strata and shared-guide subsampling | editor-and-guide-recommendation.zip, matrix-v5.zip, additional-task-inputs.zip |
| `source-audit` | Source, assay and scale inventory of the full matrix | matrix-v5.zip, revision-calibration-robustness.zip |
| `substitution` | Matched, single-base substitutions with frozen native predictions and replay of original scores | revision-perturbations.zip |
| `shuffle` | Matched original and shuffled sequences, held-out discrimination, execution coverage and original-score replay | revision-perturbations.zip |

The revision directories retain their source analysis names. Each `data/` or `raw/` directory contains selected keys, predictions, or observed values used by its numeric script. Each `results/` directory contains the frozen result tables. The source accession index covers datasets that remain at their original repositories. [Upstream catalog](../data/upstream_catalog.csv) · [File-level provenance](../payload_manifest.csv)
