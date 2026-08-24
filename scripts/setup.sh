#!/usr/bin/env sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
BASE_PYTHON=${BE_SHELL_PYTHON:-python3}
if [ ! -x "$ROOT/.venv/bin/python" ]; then
    "$BASE_PYTHON" -c 'import sys; assert sys.version_info >= (3, 10), sys.version'
    (cd "$ROOT" && "$BASE_PYTHON" scripts/verify_release.py)
    "$BASE_PYTHON" -m venv "$ROOT/.venv"
fi
"$ROOT/.venv/bin/python" -c 'import sys; assert sys.version_info >= (3, 10), sys.version'
"$ROOT/.venv/bin/python" -m pip install -e "$ROOT"
