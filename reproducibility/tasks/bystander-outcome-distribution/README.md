# Bystander and product outcome distribution

Manuscript task T2.


`inputs/` contains target, outcome and position profiles, the source-file identifiers and the exact adapter panels. `predictions/` contains the returned outcomes and positions by model. `code/score_bystander_outcomes.py` recomputes the product and position summaries. For matched and random-window comparisons use `common` and `random`.

From the repository root, run the command below.

```bash
python -B reproducibility/scripts/replay.py bystander-outcome-distribution --work-dir benchmark-replay/bystander-outcome-distribution
```

[Reproduction commands and asset index](../README.md) · [Source and hash inventory](../../payload_manifest.csv)
