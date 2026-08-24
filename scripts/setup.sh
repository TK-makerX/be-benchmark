#!/usr/bin/env sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
python3 -c 'import sys; assert sys.version_info >= (3, 10), sys.version'
python3 -m venv "$ROOT/.venv"
"$ROOT/.venv/bin/python" -m pip install -e "$ROOT"
"$ROOT/.venv/bin/python" "$ROOT/scripts/verify_release.py"
