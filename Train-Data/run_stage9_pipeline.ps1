param(
    [ValidateSet("all", "split", "embeddings", "profile-cache", "features", "merge", "train")]
    [string]$Stage = "all",

    [string]$Splits = "train,val,test",

    [int]$ChunkSize = 50000,

    [switch]$WithImage,

    [switch]$ForceSbert,

    [switch]$ForceProfileCache,

    [switch]$ForceImageCache
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$pipeline = Join-Path $scriptDir "stage9_features_pipeline_chunked.py"

if (-not (Test-Path $python)) {
    throw "Python not found at $python"
}

if (-not (Test-Path $pipeline)) {
    throw "Pipeline script not found at $pipeline"
}

$prefix = if ($WithImage) { "feature_matrix_chunked_img" } else { "feature_matrix_chunked" }
$imageArgs = @()
if (-not $WithImage) {
    $imageArgs += "--skip-image"
}

function Run-Stage9 {
    param(
        [string]$RunStage,
        [string]$RunSplits = $Splits
    )

    $args = @(
        $pipeline,
        "--stage", $RunStage,
        "--splits", $RunSplits,
        "--chunk-size", $ChunkSize,
        "--output-prefix", $prefix
    )
    $args += $imageArgs

    if ($RunStage -eq "features") {
        $args += @("--skip-existing", "--auto-resume")
    }
    if ($ForceSbert) {
        $args += "--force-sbert"
    }
    if ($ForceProfileCache) {
        $args += "--force-profile-cache"
    }
    if ($ForceImageCache) {
        $args += "--force-image-cache"
    }

    Write-Host ""
    Write-Host ">>> Running stage: $RunStage"
    Write-Host "$python $($args -join ' ')"
    & $python @args
}

switch ($Stage) {
    "all" {
        Run-Stage9 -RunStage "split"
        Run-Stage9 -RunStage "embeddings"
        Run-Stage9 -RunStage "profile-cache"
        Run-Stage9 -RunStage "features"
        Run-Stage9 -RunStage "merge"
    }
    "train" {
        Run-Stage9 -RunStage "split" -RunSplits "train,val,test"
        Run-Stage9 -RunStage "embeddings" -RunSplits "train,val,test"
        Run-Stage9 -RunStage "profile-cache" -RunSplits "train,val,test"
        Run-Stage9 -RunStage "features" -RunSplits "train"
        Run-Stage9 -RunStage "merge" -RunSplits "train"
    }
    default {
        Run-Stage9 -RunStage $Stage
    }
}

Write-Host ""
Write-Host "Stage 9 runner completed."
