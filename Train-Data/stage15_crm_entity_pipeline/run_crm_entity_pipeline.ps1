param(
    [double]$MatchThreshold = 0.98,
    [double]$ReviewThreshold = 0.95
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$projectRoot = Split-Path -Parent $root
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$script = Join-Path $PSScriptRoot "run_crm_entity_pipeline.py"

& $python $script --match-threshold $MatchThreshold --review-threshold $ReviewThreshold
