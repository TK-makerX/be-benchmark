$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Venv = Join-Path $Root ".venv"
$Python = Join-Path $Venv "Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        & py -3 -c "import sys; assert sys.version_info >= (3, 10), sys.version"
        & py -3 -m venv $Venv
    } else {
        & python -c "import sys; assert sys.version_info >= (3, 10), sys.version"
        & python -m venv $Venv
    }
}
Push-Location $Root
try {
    & $Python -m pip install -e .
    & $Python scripts\verify_release.py
} finally {
    Pop-Location
}
