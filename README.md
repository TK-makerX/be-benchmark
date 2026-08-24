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

## Local demonstration

Python 3.10 or later is required.

Run the integrity check on a clean checkout before creating a virtual environment or installing the package.

```bash
python scripts/verify_release.py
python -m venv .venv
.venv/Scripts/python -m pip install -e .
.venv/Scripts/python app/run_cli.py --sequence REPLACE_WITH_SEQUENCE --editor-family example --task example_operation
.venv/Scripts/python app/run_web_app.py
```

On Linux or macOS, use `.venv/bin/python`. The web interface binds to `127.0.0.1:8765` unless changed through environment variables.

This scaffold is not the analysis code used to generate an article and must not be described as a result-reproduction package.

## License

Released under the MIT License. See `LICENSE`.

The MIT License applies only to the first-party interface scaffold in this repository. It does not grant rights to excluded model implementations, weights, datasets or private backends.
