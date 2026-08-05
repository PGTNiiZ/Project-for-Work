param(
    [int]$ChunkSize = 100000
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$projectRoot = Split-Path -Parent $root
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$script = Join-Path $PSScriptRoot "run_full_candidate_pipeline.py"

& $python $script --chunk-size $ChunkSize
