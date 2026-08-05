$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$py = Join-Path $root "..\.venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    $py = Join-Path $root "..\Project-for-Work\.venv\Scripts\python.exe"
}
if (-not (Test-Path $py)) {
    throw "Python venv not found."
}

& $py (Join-Path $root "bench_models.py")
