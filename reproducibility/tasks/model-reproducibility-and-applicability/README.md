# Model reproducibility and applicability

Manuscript task T6.


`inputs/` contains the four timed pilot panels, model registry and recorded timing status. `predictions/` and `skipped/` document the execution evidence. `code/build_model_applicability_tables.py` recomputes the tabular summary from the registry. `../revision/02_R2.1/denominator_correction/data/applicability_tables/` records the execution-denominator amendment. The timing runner requires the upstream backends.

From the repository root, run the command below.

```bash
python -B reproducibility/scripts/replay.py model-reproducibility-and-applicability --work-dir benchmark-replay/model-reproducibility-and-applicability
```

[Reproduction commands and asset index](../README.md) · [Source and hash inventory](../../payload_manifest.csv)
