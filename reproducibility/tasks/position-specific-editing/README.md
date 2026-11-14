# Position-specific editing

Manuscript task T3.


`inputs/` and `predictions/` contain the observed profiles, scoring panels and native returned position rows. The current replay uses the execution eligibility and frozen joined scoring keys in `../revision/02_R2.1/`. Initial estimates remain under `expected/initial/` for verification of the denominator correction.

From the repository root, run the command below.

```bash
python -B reproducibility/scripts/replay.py position-specific-editing --work-dir benchmark-replay/position-specific-editing
```

[Reproduction commands and asset index](../README.md) · [Source and hash inventory](../../payload_manifest.csv)
