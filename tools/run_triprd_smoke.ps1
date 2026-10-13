[CmdletBinding()]
param(
    [string]$Python = "python",
    [string]$VdtRoot = "datasets/VDT-2048",
    [string]$CodRoot = "datasets/COD_benchmark",
    [string]$OutputRoot = "results/triprd_ab",
    [string]$Groups = "A,B,C,D",
    [ValidateSet("tiny", "resnet50")]
    [string]$ModelVariant = "tiny",
    [switch]$Pretrained,
    [int]$Epochs = 2,
    [int]$VdtEpochs = 2,
    [int]$MaxSamples = 64,
    [int]$MaxTestSamples = 64,
    [int]$BatchSize = 2,
    [int]$ImageSize = 64,
    [int]$BaseChannels = 8,
    [string[]]$Datasets = @("COD10K")
)

<#
.SYNOPSIS
    Run the controlled TriPRD A/B/C/D validation experiment.

.DESCRIPTION
    The runner fixes train/validation manifests, trains the requested groups
    with the same model/seed/budget, selects checkpoints by validation Dice,
    and exports RGB-only test predictions plus official COD metrics when the
    metric dependencies are installed. Defaults are a small smoke; set
    MaxSamples/MaxTestSamples to 0 and increase epochs for the full protocol.
#>

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$VdtRoot = if ([System.IO.Path]::IsPathRooted($VdtRoot)) { $VdtRoot } else { Join-Path $RepoRoot $VdtRoot }
$CodRoot = if ([System.IO.Path]::IsPathRooted($CodRoot)) { $CodRoot } else { Join-Path $RepoRoot $CodRoot }
$OutputRoot = if ([System.IO.Path]::IsPathRooted($OutputRoot)) { $OutputRoot } else { Join-Path $RepoRoot $OutputRoot }
$Runner = Join-Path $RepoRoot "tools/run_triprd_ab.py"

function Invoke-TriprdPython {
    param([string[]]$Arguments)
    Write-Host (">> " + $Python + " " + ($Arguments -join " ")) -ForegroundColor Cyan
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "TriPRD A/B/C/D command failed with exit code $LASTEXITCODE"
    }
}

Push-Location $RepoRoot
try {
    $arguments = @(
        $Runner,
        "--vdt-root", $VdtRoot,
        "--cod-root", $CodRoot,
        "--out", $OutputRoot,
        "--groups", $Groups,
        "--model-variant", $ModelVariant,
        $(if ($Pretrained) { "--pretrained" } else { "--no-pretrained" }),
        "--epochs", $Epochs,
        "--vdt-epochs", $VdtEpochs,
        "--max-samples", $MaxSamples,
        "--max-test-samples", $MaxTestSamples,
        "--batch-size", $BatchSize,
        "--image-size", $ImageSize,
        "--base-channels", $BaseChannels,
        "--test-datasets", ($Datasets -join ",")
    )
    Invoke-TriprdPython $arguments
    Write-Host "TriPRD A/B/C/D experiment completed." -ForegroundColor Green
    Write-Host "Split/checkpoints/logs/predictions/metrics: $OutputRoot"
}
finally {
    Pop-Location
}

