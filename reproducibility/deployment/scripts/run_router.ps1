$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"
$Python = if (Test-Path -LiteralPath $VenvPython) { $VenvPython } else { "python" }
if (-not $env:BE_WEB_HOST) { $env:BE_WEB_HOST = "127.0.0.1" }
if (-not $env:BE_WEB_PORT) { $env:BE_WEB_PORT = "8765" }
& $Python (Join-Path $Root "app\run_web_app.py")
