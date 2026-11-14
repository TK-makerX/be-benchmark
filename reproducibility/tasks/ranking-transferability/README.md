# Ranking transferability

Manuscript task T5.


`tables/` and `intermediate/` contain the exact eligible joint units, paired bootstrap and model-level estimates. `code/replay_matched_ranking_transfer.py` uses the 10,000 scalar activity observations and the four-model frozen score table. The three comparisons contain 661 ABE-to-ABE-CP1040, 206 HEK293T-to-U2OS and 448 HEK293T-to-mES joint units. `expected/narrative_amendment/` holds the V1.2.1 tables.

From the repository root, run the command below.

```bash
python -B reproducibility/scripts/replay.py ranking-transferability --work-dir benchmark-replay/ranking-transferability
```

[Reproduction commands and asset index](../README.md) · [Source and hash inventory](../../payload_manifest.csv)
