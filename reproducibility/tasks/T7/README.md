# T7 Unique-design recommendation

`inputs/T7_design39_observation_membership.csv` contains the 503,545 pre-aggregation records. `T7_observation_scores_used.csv.gz` preserves all 1,007,090 native model-observation records for the ten fields read by the scoring script. The final candidate and model-score tables cover 438,768 design-group instances across 39 groups. `code/replay_aggregation.py` reconstructs the membership and scores. `code/replay_recommendation.py` recalculates group metrics and 1,000 random baselines. The simple rule and source/editor strata replay is `shortlist`.

[Reproduction commands and asset index](../README.md) · [Source and hash inventory](../../payload_manifest.csv)
