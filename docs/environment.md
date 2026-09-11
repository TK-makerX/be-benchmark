# Environment and installation

These instructions apply only to the public interface scaffold. They do not describe the environments, dependencies or entry points of private prediction models.

## Requirements

| Component | Requirement |
| --- | --- |
| Interpreter | Python 3.10 or later |
| Source-tree runtime and tests | Python standard library only |
| Package installation or wheel build | pip and the build requirements declared in `pyproject.toml`: `setuptools>=77` and `wheel` |
| GPU, Conda, model weights or datasets | Not needed for these demonstrations |
| Network | Not needed for source-tree execution; pip may need access to a package index when installing build requirements |

The CI configuration checks Python 3.10 and 3.12 on Linux, and Python 3.12 on Windows. The POSIX commands below also describe the usual macOS layout, but macOS is not part of that CI matrix. The build requirements are minimum versions, not a frozen environment for a prediction model.

## Use the source tree without installing anything

From the repository root:

```text
python --version
python -B scripts/verify_release.py
python -B examples/run_demo.py
python -B -m unittest discover -s tests -v
```

Use `python3` if that is the name of your interpreter. The `-B` flag avoids creating bytecode files in the checkout. The release checker expects the exact source manifest: run it before creating `.venv`, installing an editable package or producing build files. An installed working directory is not a clean release snapshot.

## Optional virtual environment

The existing setup scripts check the release before creating a new `.venv`, then install the interface package. They do not install or configure prediction backends. Run them from a fresh checkout.

### Windows PowerShell

```powershell
$env:BE_SHELL_PYTHON = 'python'
& .\scripts\setup.ps1
& .\.venv\Scripts\python.exe -B examples\run_demo.py
& .\.venv\Scripts\python.exe -m pip check
```

Set `BE_SHELL_PYTHON` to the interpreter you intend to use; it must be available on your machine. The script otherwise tries the Windows `py` launcher, then `python`. Activation is optional because the commands call the environment's interpreter directly.

### Linux or macOS shell

```sh
BE_SHELL_PYTHON=python3 sh scripts/setup.sh
.venv/bin/python -B examples/run_demo.py
.venv/bin/python -m pip check
```

The setup scripts use pip's build isolation by default. For an offline machine, source-tree execution is the simplest option; installation also requires the build packages to be available locally. `scripts/wheel_smoke.py` supports `BE_SHELL_OFFLINE_BUILD=1` when compatible build requirements are already installed. This switch does not supply missing packages.

## Configuration

| Variable | Default or effect |
| --- | --- |
| `BE_SHELL_PYTHON` | Interpreter selection for the setup scripts only |
| `BE_SHELL_REGISTRY` | Unset means no backend registry; a supplied path is read by the CLI and web interface |
| `BE_WEB_HOST` | `127.0.0.1` for the local web server |
| `BE_WEB_PORT` | `8765` for the local web server |
| `BE_SHELL_OFFLINE_BUILD` | `1` disables build isolation in the wheel smoke test only |

`.env.example` is a configuration reference, not an automatically loaded configuration file. Export variables in your shell when needed. Leave `BE_SHELL_REGISTRY` unset for the public CLI/web demonstration, or point it to the bundled empty template. `examples/run_demo.py` always selects that template explicitly, regardless of the environment variable.

Keep any actual registry and backend configuration outside the public checkout. Do not publish a private configuration file, an environment dump or credentials when reporting an installation issue.

## Optional container demonstration

With Docker available, run from a clean checkout:

```sh
docker build -f deployment/Dockerfile -t be-benchmark-shell .
docker run --rm -p 127.0.0.1:8765:8765 be-benchmark-shell
```

The existing image uses Python 3.10 and starts the same empty-backend web interface. Pulling the base image and installing build requirements may require network access. No data or model directories need to be mounted. Keep the host port bound to loopback; the demonstration server is not an authenticated public service.

See [running examples](running_examples.md) for the expected CLI and web responses.
