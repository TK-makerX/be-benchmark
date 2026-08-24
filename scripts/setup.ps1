$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Venv = Join-Path $Root ".venv"
$Python = Join-Path $Venv "Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    Push-Location $Root
    try {
        if ($env:BE_SHELL_PYTHON) {
            & $env:BE_SHELL_PYTHON -c "import sys; assert sys.version_info >= (3, 10), sys.version"
            & $env:BE_SHELL_PYTHON scripts\verify_release.py
            & $env:BE_SHELL_PYTHON -m venv $Venv
        } elseif (Get-Command py -ErrorAction SilentlyContinue) {
            & py -3 -c "import sys; assert sys.version_info >= (3, 10), sys.version"
            & py -3 scripts\verify_release.py
            & py -3 -m venv $Venv
        } else {
            & python -c "import sys; assert sys.version_info >= (3, 10), sys.version"
            & python scripts\verify_release.py
            & python -m venv $Venv
        }
    } finally {
        Pop-Location
    }
}
Push-Location $Root
try {
    & $Python -c "import sys; assert sys.version_info >= (3, 10), sys.version"
    & $Python -m pip install -e .
} finally {
    Pop-Location
}
