#!/usr/bin/env sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
PYTHON="$ROOT/.venv/bin/python"
if [ ! -x "$PYTHON" ]; then PYTHON=python3; fi
: "${BE_WEB_HOST:=127.0.0.1}"
: "${BE_WEB_PORT:=8765}"
export BE_WEB_HOST BE_WEB_PORT
exec "$PYTHON" "$ROOT/app/run_web_app.py"
