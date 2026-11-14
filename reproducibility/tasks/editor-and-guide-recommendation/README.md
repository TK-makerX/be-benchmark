# Editor and guide recommendation

Manuscript task T7.


`inputs/T7_design39_observation_membership.csv` contains the 503,545 pre-aggregation records. `T7_observation_scores_used.csv.gz` preserves all 1,007,090 native model-observation records for the ten fields read by the scoring script. The final candidate and model-score tables cover 438,768 design-group instances across 39 groups. `code/replay_recommendation_aggregation.py` reconstructs the membership and scores. `code/replay_editor_and_guide_recommendation.py` recalculates group metrics and 1,000 random baselines. The simple rule and source/editor strata replay is `shortlist`.

From the repository root, run the command below.

```bash
python -B reproducibility/scripts/replay.py editor-and-guide-recommendation --work-dir benchmark-replay/editor-and-guide-recommendation
```

[Reproduction commands and asset index](../README.md) · [Source and hash inventory](../../payload_manifest.csv)
