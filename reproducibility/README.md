# Base-editor benchmark reproducibility package

This deposit accompanies the revised base-editor benchmark. It contains the full V5 harmonized observation matrix, the exact inputs used to build it, the T1–T7 frozen evidence, the revision analyses and the local prediction system.

| Contents | Entry point |
|---|---|
| 1,147,701 V5 observations and 61 exact upstream input files | [Matrix and sources](data/README.md) |
| T1–T7 and revision data, predictions, scoring and uncertainty estimates | [Analysis and asset index](tasks/README.md) |
| Local prediction router, inference adapters and external checkpoint provenance | [Model deployment](deployment/README.md) |

## Download and verify

Use Python 3.10 or 3.11. From the repository root, run the commands below. The ZIPs extract relative to `reproducibility/`. Select an individual ZIP with `--asset NAME` or use `--from-dir` with existing downloads.

```bash
python -B reproducibility/scripts/verify_bundle.py --git-only
python -B reproducibility/scripts/fetch_assets.py --download-dir benchmark-assets
python -B reproducibility/scripts/verify_bundle.py
python -m pip install -r reproducibility/requirements-analysis.txt
python -B reproducibility/scripts/replay.py --list
python -B reproducibility/scripts/replay.py T1 --work-dir benchmark-replay/T1
```

[assets.json](assets.json) contains the release URLs, sizes, SHA-256 and member hashes. [payload_manifest.csv](payload_manifest.csv) records the source and deposited hashes of individual files, along with any changes to code or provenance fields. [analysis_code_provenance.json](analysis_code_provenance.json) maps numeric functions extracted from mixed scripts back to their preserved sources. [replay_verification.json](replay_verification.json) records the checks performed against the frozen results.

Project-authored analysis and deployment code is provided for reproduction. The upstream datasets and models retain their original terms. Model archives and their required versions are linked in the deployment index.
