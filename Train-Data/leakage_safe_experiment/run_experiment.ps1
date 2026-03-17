param(
    [double]$RandomNegRatio = 4.0,
    [double]$HardNegRatio = 1.0,
    [int]$Seed = 42
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent (Split-Path -Parent $scriptDir)
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$script = Join-Path $scriptDir "run_experiment.py"

if (-not (Test-Path $python)) {
    throw "Python not found at $python"
}

if (-not (Test-Path $script)) {
    throw "Experiment script not found at $script"
}

$args = @(
    $script,
    "--random-neg-ratio", $RandomNegRatio,
    "--hard-neg-ratio", $HardNegRatio,
    "--seed", $Seed
)

Write-Host ""
Write-Host ">>> Running leakage-safe experiment"
Write-Host "$python $($args -join ' ')"
& $python @args
