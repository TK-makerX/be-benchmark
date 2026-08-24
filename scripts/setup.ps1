$ErrorActionPreference = "Stop"

function Assert-NativeCommandSucceeded {
    param([string]$Step)
    if ($LASTEXITCODE -ne 0) {
        throw "$Step failed with exit code $LASTEXITCODE"
    }
}

$Root = Split-Path -Parent $PSScriptRoot
$Venv = Join-Path $Root ".venv"
$Python = Join-Path $Venv "Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    Push-Location $Root
    try {
        if ($env:BE_SHELL_PYTHON) {
            & $env:BE_SHELL_PYTHON -c "import sys; assert sys.version_info >= (3, 10), sys.version"
            Assert-NativeCommandSucceeded "Python version check"
            & $env:BE_SHELL_PYTHON scripts\verify_release.py
            Assert-NativeCommandSucceeded "Release verification"
            & $env:BE_SHELL_PYTHON -m venv $Venv
            Assert-NativeCommandSucceeded "Virtual environment creation"
        } elseif (Get-Command py -ErrorAction SilentlyContinue) {
            & py -3 -c "import sys; assert sys.version_info >= (3, 10), sys.version"
            Assert-NativeCommandSucceeded "Python version check"
            & py -3 scripts\verify_release.py
            Assert-NativeCommandSucceeded "Release verification"
            & py -3 -m venv $Venv
            Assert-NativeCommandSucceeded "Virtual environment creation"
        } else {
            & python -c "import sys; assert sys.version_info >= (3, 10), sys.version"
            Assert-NativeCommandSucceeded "Python version check"
            & python scripts\verify_release.py
            Assert-NativeCommandSucceeded "Release verification"
            & python -m venv $Venv
            Assert-NativeCommandSucceeded "Virtual environment creation"
        }
    } finally {
        Pop-Location
    }
}
Push-Location $Root
try {
    & $Python -c "import sys; assert sys.version_info >= (3, 10), sys.version"
    Assert-NativeCommandSucceeded "Virtual environment Python version check"
    & $Python -m pip install -e .
    Assert-NativeCommandSucceeded "Package installation"
} finally {
    Pop-Location
}
