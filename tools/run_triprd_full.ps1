[CmdletBinding()]
param(
    [string]$Python = "python",
    [string]$VdtRoot = "datasets/VDT-2048",
    [string]$CodRoot = "datasets/COD_benchmark",
    [string]$OutputRoot = "results/triprd_full",
    [string]$Groups = "A,B,C,D",
    [int]$VdtEpochs = 100,
    [int]$CodEpochs = 100,
    [int]$BatchSize = 8,
    [int]$ImageSize = 352,
    [int]$BaseChannels = 256,
    [int]$Seed = 20261012,
    [string]$Device = "cuda",
    [string[]]$Datasets = @("NC4K", "COD10K", "CAMO", "CHAMELEON")
)

<# Run the complete paper-comparable TriPRD A/B/C/D experiment. #>
$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$VdtRoot = if ([System.IO.Path]::IsPathRooted($VdtRoot)) { $VdtRoot } else { Join-Path $RepoRoot $VdtRoot }
$CodRoot = if ([System.IO.Path]::IsPathRooted($CodRoot)) { $CodRoot } else { Join-Path $RepoRoot $CodRoot }
$OutputRoot = if ([System.IO.Path]::IsPathRooted($OutputRoot)) { $OutputRoot } else { Join-Path $RepoRoot $OutputRoot }
$Runner = Join-Path $RepoRoot "tools/run_triprd_ab.py"

Push-Location $RepoRoot
try {
    $arguments = @(
        $Runner, "--vdt-root", $VdtRoot, "--cod-root", $CodRoot,
        "--out", $OutputRoot, "--groups", $Groups,
        "--model-variant", "resnet50", "--pretrained",
        "--vdt-epochs", $VdtEpochs, "--epochs", $CodEpochs,
        "--max-samples", 0, "--max-test-samples", 0,
        "--batch-size", $BatchSize, "--image-size", $ImageSize,
        "--base-channels", $BaseChannels, "--seed", $Seed,
        "--device", $Device, "--test-datasets", ($Datasets -join ",")
    )
    Write-Host (">> " + $Python + " " + ($arguments -join " ")) -ForegroundColor Cyan
    & $Python @arguments
    if ($LASTEXITCODE -ne 0) { throw "TriPRD full experiment failed with exit code $LASTEXITCODE" }
    Write-Host "TriPRD full experiment completed. Results: $OutputRoot" -ForegroundColor Green
}
finally { Pop-Location }
