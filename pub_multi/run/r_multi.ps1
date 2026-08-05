param(
    [double]$RandomNegRatio = 0.75,
    [double]$HardNegRatio = 2.0,
    [int]$Seed = 42
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$projectRoot = Split-Path -Parent $root
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$script = Join-Path $PSScriptRoot "run_multimodal_suite.py"

& $python $script `
    --random-neg-ratio $RandomNegRatio `
    --hard-neg-ratio $HardNegRatio `
    --seed $Seed
