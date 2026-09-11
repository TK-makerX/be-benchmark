# BE-Benchmark interface scaffold

This repository is a generic software scaffold for a base-editor prediction interface. It intentionally contains no manuscript-specific dataset, source-data table, result value, analysis workflow, plotting workflow, trained model, model weight or third-party implementation.

The scaffold provides only:

- request validation;
- an empty backend registry contract;
- abstract routing and connector interfaces;
- a command-line demonstration;
- a local web-interface demonstration;
- lightweight packaging and contract tests.

The default registry is empty. Consequently, the application validates requests but does not produce a scientific prediction. Add a separately licensed backend in a private deployment before numerical inference is possible.

## Quick start: no model or dataset required

Python 3.10 or later is required. The source-tree demonstration uses only the Python standard library; it does not require package installation, a GPU or a model download.

Run these commands from a clean checkout, before creating a virtual environment or installing the package:

```text
python -B scripts/verify_release.py
python -B examples/run_demo.py
python -B -m unittest discover -s tests -v
```

If your interpreter is named `python3`, use that command instead. The demonstration generates a repeated-character input in memory and explicitly selects the bundled empty registry. Its expected status is `backend_not_configured`, with a null backend identifier and no prediction field. This means the interface ran successfully; it is not a model prediction or a biological example.

- [Environment and installation](docs/environment.md): source-only use, virtual environments, configuration and checks.
- [Running the examples](docs/running_examples.md): CLI commands, the local web form, expected responses and troubleshooting.
- [Publication scope](docs/publication_scope.md): what this public repository does and does not include.

This scaffold is not the analysis code used to generate an article and must not be described as a result-reproduction package.

## License

Released under the MIT License. See `LICENSE`.

The MIT License applies only to the first-party interface scaffold in this repository. It does not grant rights to excluded model implementations, weights, datasets or private backends.
