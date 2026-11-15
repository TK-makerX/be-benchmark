# Code and data availability

The base-editing benchmark data and code are publicly available in the [versioned GitHub release](https://github.com/TK-makerX/be-benchmark/releases/tag/benchmark-revision-2026-11). Its source archive contains the project code and its download assets contain the analysis data.

## Data

The release includes the V5 matrix of 1,147,701 observations and its 61 source inputs, plus the observation identifiers, split definitions, frozen predictions and result tables for the main and revision analyses. The source index provides original download locations and checksums. Larger upstream sequencing collections remain available through the accessions listed in the catalog. [Matrix and source data](../reproducibility/data/README.md) · [Task and data index](../reproducibility/tasks/README.md)

HEK2 experimental source tables are handled through the manuscript's supplementary materials.

## Code

The repository provides the scripts for matrix construction, the seven benchmark tasks, the revision analyses and local model deployment. The reproduction guide gives commands to download and verify the inputs and recompute the benchmark statistics from the frozen predictions. The release covers numerical analysis and deployment. Plotting scripts fall outside its scope. [Reproduction guide](../reproducibility/README.md) · [Model deployment](../reproducibility/deployment/README.md)

The deployment index gives the upstream repositories and checkpoint hashes required for fresh inference. Obtain those components from their providers. The MIT License covers the standalone interface package. Upstream datasets and models retain their documented terms. [Third-party notices](../reproducibility/deployment/THIRD_PARTY_NOTICES.md)
