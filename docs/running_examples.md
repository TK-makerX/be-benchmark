# Running the public examples

All inputs below are generated test strings. They are not target sequences, guides, experimental observations or examples from an article. The editor and task labels are placeholders. No example downloads a dataset, resolves a model checkpoint or calls a numerical predictor.

Use Python 3.10 or later and run commands from the repository root. See [environment and installation](environment.md) if Python or package installation is not ready.

## 1. One-command interface demonstration

```text
python -B examples/run_demo.py
```

The script builds its input as `"ACGT" * 6` and explicitly uses the bundled empty registry. It does not read a private registry selected in your shell. The expected response is:

```json
{
  "request": {
    "sequence_length": 24,
    "editor_family": "example",
    "task": "example_operation"
  },
  "status": "backend_not_configured",
  "backend_id": null,
  "message": "This public scaffold validates and routes requests but does not run numerical inference."
}
```

The length describes the generated test string, not a required model window or a study sample count. Exit code `0` confirms that the demonstration completed. The absence of a prediction is intentional.

## 2. Call the existing CLI directly

Select the public empty registry explicitly so that a private environment setting cannot change this example.

### Windows PowerShell

```powershell
$env:BE_SHELL_REGISTRY = (Resolve-Path .\configs\backend_registry.template.json).Path
$sequence = 'ACGT' * 6
python -B app/run_cli.py --sequence $sequence --editor-family example --task example_operation
```

### Linux or macOS shell

```sh
sequence=$(python3 -c 'print("ACGT" * 6)')
BE_SHELL_REGISTRY="$PWD/configs/backend_registry.template.json" python3 -B app/run_cli.py --sequence "$sequence" --editor-family example --task example_operation
```

The expected response is the same as in the one-command demonstration. After package installation, `python -m be_benchmark_shell` or the `be-benchmark-shell` console command accepts the same arguments; use the interpreter or console script inside your virtual environment.

To check input validation, submit a deliberately short string:

```text
python -B app/run_cli.py --sequence ACGT --editor-family example --task example_operation
```

This returns `invalid_request` and exit code `2`. It is an expected validation failure, not a broken installation. The public interface validates its own general input contract; passing this check does not establish compatibility with a particular editor or predictor.

## 3. Use the local web form

Keep the empty registry setting from the CLI example. On PowerShell:

```powershell
$env:BE_SHELL_REGISTRY = (Resolve-Path .\configs\backend_registry.template.json).Path
$env:BE_WEB_HOST = '127.0.0.1'
$env:BE_WEB_PORT = '8765'
python -B app/run_web_app.py
```

On Linux or macOS:

```sh
BE_SHELL_REGISTRY="$PWD/configs/backend_registry.template.json" BE_WEB_HOST=127.0.0.1 BE_WEB_PORT=8765 python3 -B app/run_web_app.py
```

Open `127.0.0.1:8765` in a browser using HTTP. In a second terminal, run `python -c 'print("ACGT" * 6)'` (or `python3`) and copy its generated string into the sequence field. Use `example` for the editor family and `example_operation` for the task, then select **Validate request**. Stop the local server with Ctrl+C when finished.

| Request | Expected HTTP status | Meaning |
| --- | --- | --- |
| GET `/health` | 200 | Web process is responding; this is not a model-readiness check |
| Valid form with the empty registry | 503 | The form was processed, but no numerical backend is configured; the page displays `backend_not_configured` |
| Invalid form input | 400 | The page displays `invalid_request` |
| Explicit registry path is missing or invalid | 503 | The page displays `invalid_registry` |

The web form returns an HTML page containing the status object, not a standalone JSON prediction API. A 503 response for the empty registry is expected and differs from a server that cannot start. These are software interface states, not editing efficiencies or model-performance results.

## Common problems

| Symptom | What to check |
| --- | --- |
| Python command is missing | Check which Python 3.10+ interpreter is installed; on some systems the command is `python3` or `py -3` |
| `No module named be_benchmark_shell` | Use the source-tree entry point `app/run_cli.py`, or install the package into the interpreter you are invoking |
| `invalid_registry` | Select the bundled empty template for the demonstration; copying `.env.example` does not load it automatically |
| Port is already in use | Choose another unused `BE_WEB_PORT` and open that port; do not stop an unrelated service |
| Release verification fails after installation | The checker expects an exact clean checkout; `.venv`, editable-install metadata and generated files are not part of the release |
| Setup fails while obtaining build requirements | Check access to the configured package index, or use the source-tree demonstration without installation |

The automated tests exercise the CLI and web status contracts using generated inputs only. They do not validate a scientific predictor. Do not attach private registry files, data, logs containing real inputs or model environments when sharing a problem report.
