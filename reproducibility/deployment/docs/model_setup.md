# Model setup

Third-party model code and weights are not vendored. Install each model from `third_party/model_sources.csv`, retain its upstream notices, and verify asset hashes.

Portable variables include:

```text
BE_MODELS_DIR
BE_DATA_DIR
BE_RESULTS_DIR
BE_REPORTS_DIR
BE_SCHEMA_DIR
BE_TOOLS_DIR
BE_BENCHMARK_CORE
BE_ADAPTERS_DIR
BE_FORECAST_PYTHON
BE_TF210_PYTHON
BE_BEDICT_PYTHON
BE_BEHIVE_PYTHON
```

Files under `environments/models/` record the locally reproduced environments and first-party wrappers. They do not combine incompatible model environments into one universal environment.

CRISPRon-BE uses a model environment plus an RNAfold-dependent feature step. Its license restricts some production and hosted-service uses; do not deploy it publicly without an independent license review.
