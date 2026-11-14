# Scalar editing activity

Manuscript task T1.


V8 label-blind common set and V8.1 calibration. The 10,000 observation IDs and 40,000 joined model-observation rows are in `inputs/`. `predictions/raw/` contains the four frozen adapter outputs, and `predictions/harmonized/` contains the scoring values. `code/replay_scalar_editing_activity.py` recomputes association, cluster intervals and metadata strata. `code/replay_scalar_calibration_v8_1.py` recomputes the corrected decile summary.

From the repository root, run the command below.

```bash
python -B reproducibility/scripts/replay.py scalar-editing-activity --work-dir benchmark-replay/scalar-editing-activity
```

[Reproduction commands and asset index](../README.md) · [Source and hash inventory](../../payload_manifest.csv)
