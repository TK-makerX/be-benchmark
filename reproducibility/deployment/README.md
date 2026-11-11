# Local model deployment

This is the scientific registry and the actual local routing and execution layer, version 0.6.0. It provides the single-sequence CLI, local web interface, candidate generation, and adapters for the runnable backends.

Run from the repository root with Python 3.10 or later.

```bash
python reproducibility/deployment/app/run_cli.py --sequence GGGGAAAAACCCCCGGGGGTTTTTGGGAAA --editor ABE --task auto --json
python reproducibility/deployment/app/run_web_app.py
```

Open `http://127.0.0.1:8765` for the local interface. The router-only Docker configuration lives under `deployment/`. Start it with `docker compose -f reproducibility/deployment/deployment/docker-compose.router-only.yml up --build`.

## Run a native backend

1. Obtain the pinned sources in [model_sources.csv](model_sources.csv) and place each at its listed path under `BE_MODELS_DIR`. Obtain the required checkpoints from [external_weights_sha256.csv](external_weights_sha256.csv) and verify their hashes before loading them.
2. Use the matching [environment files](environments/models/). Set the environment variables named in the model index to the absolute Python executable paths. The shared model root holds the upstream repos and their assets. Model-specific repo variables in [model setup](docs/model_setup.md) override the defaults.
3. For CRISPRon-BE, install the native CRISPRon and CRISPRoff components and place the extracted checkpoint directories under its repository's `data/`. Set `BE_TOOLS_DIR` to a directory with an RNAfold executable or compatible wrapper.

```bash
python reproducibility/deployment/app/run_cli.py --sequence GGGGAAAAACCCCCGGGGGTTTTTGGGAAA --outcome GGGGAAAAGCCCCCGGGGGTTTTTGGGAAA --editor ABE8e --cas SpRY --task auto --run-models --backend forecast_be --backend deepbaseeditor --json
```

The two connectors in this example were exercised using the archived Windows environments. The FORECasT-BE native score and the DeepBaseEditor score use different scales. The frozen examples under [examples/frozen/](examples/frozen/) preserve the 30-nt all-backend execution and the separate 77-nt candidate generation run.

## Dependency versions

- BEDICT-V2 uses the recorded CPU PyTorch environment.
- FORECasT-BE uses its dedicated environment and the adapter's scikit-learn/joblib compatibility path.
- DeepBaseEditor uses the TF1 compatibility runner in a TensorFlow 2.10 environment. The worker creates an isolated working copy for the legacy script.
- BE-Hive efficiency uses the archived legacy scikit-learn environment. BE-Hive bystander, Yachie and EPCNNBE use the shared PyTorch environment. BE_Endo sequence-only and DeepBE use the TensorFlow 2.10 environment.
- DeepBE's original `version1` weights are split into ten upstream parts. Concatenate `model_v1a` through `model_v1j` in order and extract into the adapter's `resources/extracted_models/` layout. EPCNNBE's upstream `be_weights.zip` extracts to `repo/be_weights/`.
- The LLM-EmBEditor source pin records a method-only pipeline. The archived source has no author-scale regression checkpoint.

[Model code, environments and checkpoint terms](THIRD_PARTY_NOTICES.md) · [External checkpoint checksums](external_weights_sha256.csv)
