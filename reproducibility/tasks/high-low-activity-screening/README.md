# High/low editing activity screening

Manuscript task T4.


V2.1 uses design, editor and cell context units. `source_data/` holds the frozen observation-to-unit join, reference labels, thresholds, splits and bootstrap rows. `code/verify_activity_screening_v2_1.py` independently verifies the denominators, primary scores, confidence intervals and paired comparisons.

From the repository root, run the command below.

```bash
python -B reproducibility/scripts/replay.py high-low-activity-screening --work-dir benchmark-replay/high-low-activity-screening
```

[Reproduction commands and asset index](../README.md) · [Source and hash inventory](../../payload_manifest.csv)
