param(
    [int]$Epochs = 5,
    [int]$Patience = 3,
    [switch]$Cpu
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$pipeline = Join-Path $scriptDir "stage10_13_training_pipeline.py"
$excludeFile = Join-Path $scriptDir "leakage_excluded_features.txt"
$chunkedRoot = Join-Path $scriptDir "stage9_pipeline_chunked"
$trainingRoot = Join-Path $scriptDir "stage10_13_training_noleak"

if (-not (Test-Path $python)) {
    throw "Python not found at $python"
}

if (-not (Test-Path $pipeline)) {
    throw "Pipeline script not found at $pipeline"
}

$args = @(
    $pipeline,
    "--chunked-root", $chunkedRoot,
    "--training-root", $trainingRoot,
    "--exclude-features-file", $excludeFile,
    "--epochs", $Epochs,
    "--patience", $Patience
)

if ($Cpu) {
    $args += "--cpu"
}

Write-Host ""
Write-Host ">>> Running no-leak training pipeline"
Write-Host "$python $($args -join ' ')"
& $python @args
