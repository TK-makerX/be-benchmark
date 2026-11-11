$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $VenvPython)) {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        & py -3 -c "import sys; assert sys.version_info >= (3, 10), sys.version"
        & py -3 -m venv (Join-Path $Root ".venv")
    } else {
        & python -c "import sys; assert sys.version_info >= (3, 10), sys.version"
        & python -m venv (Join-Path $Root ".venv")
    }
}
& $VenvPython -c "import sys; assert sys.version_info >= (3, 10), sys.version"
Push-Location $Root
try {
    & $VenvPython -m pip install --upgrade pip
    & $VenvPython -m pip install -e ".[analysis,test]"
    & $VenvPython scripts\verify_release.py
} finally {
    Pop-Location
}
