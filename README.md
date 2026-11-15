# BE-Benchmark

BE-Benchmark compares base-editor predictors across seven tasks. The repository contains the analysis and model deployment code. The [versioned GitHub release](https://github.com/TK-makerX/be-benchmark/releases/tag/benchmark-revision-2026-11) contains the V5 observation matrix, the source inputs used to build it, and the frozen observations, splits, predictions and results used in the benchmark.

| Start here | Contents |
|---|---|
| [Reproduction guide](reproducibility/README.md) | Download the data, verify the checksums and rerun the analyses. |
| [Analysis index](reproducibility/tasks/README.md) | Editing activity, bystander outcomes, position-specific editing, activity screening, ranking transfer, model applicability, editor and guide recommendation, and revision analyses. |
| [Model deployment](reproducibility/deployment/README.md) | Local CLI and web interface, guide generation, model adapters and environment setup. |
| [Code and data availability](docs/publication_scope.md) | Published files, upstream sources and license scope. |

## Reproduce the benchmark

Use Python 3.10 or 3.11 and run these commands from the repository root.

```bash
python -B reproducibility/scripts/fetch_assets.py
python -B reproducibility/scripts/verify_bundle.py
python -m pip install -r reproducibility/requirements-analysis.txt
python -B reproducibility/scripts/replay.py --list
```

Select a command from the [analysis index](reproducibility/tasks/README.md). Replays use the released observations and frozen predictions. For new inference, follow the [model setup](reproducibility/deployment/README.md) and obtain the required upstream checkpoints.

The separate `be-benchmark-shell` package contains a [standalone interface demonstration](docs/running_examples.md) with an empty backend registry.

## License

The standalone interface package is under the MIT License. Benchmark data and upstream model components follow the terms listed in the [availability statement](docs/publication_scope.md). [MIT License](LICENSE)
