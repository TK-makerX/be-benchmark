# T1 Scalar editing activity

V8 label-blind common set and V8.1 calibration. The 10,000 observation IDs and 40,000 joined model-observation rows are in `inputs/`. `predictions/raw/` contains the four frozen adapter outputs, and `predictions/harmonized/` contains the scoring values. `code/replay_scalar.py` recomputes association, cluster intervals and metadata strata. `code/replay_calibration_v81.py` recomputes the corrected decile summary.

[Reproduction commands and asset index](../README.md) · [Source and hash inventory](../../payload_manifest.csv)
