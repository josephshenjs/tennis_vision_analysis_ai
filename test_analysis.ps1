<#
.SYNOPSIS
Reuse saved court calibration, or select four corners, then analyze.
#>
[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [Parameter(Position = 0)]
    [string]$VideoPath = 'C:\videos\tennis.mp4',
    [ValidatePattern('^[A-Za-z0-9_-]+$')]
    [string]$CameraId = 'court1',
    [ValidateRange(1,32)]
    [int]$BatchSize = 4,
    [int[]]$FarRoi,
    [switch]$Overwrite,
    [switch]$OverwriteResults
)
$ErrorActionPreference = 'Stop'
$python = Join-Path $PSScriptRoot 'runtime\python.exe'
$entryPoint = Join-Path $PSScriptRoot 'run.py'
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) { throw "Runtime missing: $python" }
if (-not (Test-Path -LiteralPath $entryPoint -PathType Leaf)) { throw "Launcher missing: $entryPoint" }
if (-not (Test-Path -LiteralPath $VideoPath -PathType Leaf)) { throw "Video not found: $VideoPath" }
$resolvedVideo = (Resolve-Path -LiteralPath $VideoPath).Path
if ($FarRoi -and $FarRoi.Count -ne 4) { throw 'FarRoi requires four integers: X,Y,WIDTH,HEIGHT.' }
$calibrationFolder = Join-Path $PSScriptRoot 'calibrations'
$calibration = Join-Path $calibrationFolder ($CameraId + '.json')
$reuseCalibration = (Test-Path -LiteralPath $calibration -PathType Leaf) -and -not $Overwrite
$action = if ($reuseCalibration) { 'Reuse saved court calibration and run analysis' } else { 'Manually select four corners, save calibration, then run analysis' }
if (-not $PSCmdlet.ShouldProcess($resolvedVideo, $action)) { return }
if ($reuseCalibration) {
    # Parse before invoking analysis; geometry and resolution are validated by the analyzer.
    $null = Get-Content -LiteralPath $calibration -Raw | ConvertFrom-Json
    Write-Host "Step 1/2: Reusing calibration: $calibration"
} else {
New-Item -ItemType Directory -Path $calibrationFolder -Force | Out-Null
# Unique staging prevents cancellation from reusing an older calibration.
$pendingStem = '.pending-' + $CameraId + '-' + [Guid]::NewGuid().ToString('N')
$pending = Join-Path $calibrationFolder ($pendingStem + '.json')
$calibrateArgs = @('calibrate','--video',$resolvedVideo,'--camera-id',$CameraId,'--output',$pending)
Write-Host 'Step 1/2: Select FAR LEFT, FAR RIGHT, NEAR RIGHT, NEAR LEFT; S saves, Q cancels.'
& $python $entryPoint @calibrateArgs
if ($LASTEXITCODE -eq 130 -or $LASTEXITCODE -eq -1073741510) { Write-Host 'Calibration cancelled.'; return }
if ($LASTEXITCODE -ne 0) { throw "Calibration failed (exit $LASTEXITCODE). Analysis was not started." }
if (-not (Test-Path -LiteralPath $pending -PathType Leaf)) {
    Write-Host 'Calibration cancelled. Analysis was not started; previous calibration is unchanged.'
    return
}
$confirmed = Get-Content -LiteralPath $pending -Raw | ConvertFrom-Json
if ($confirmed.review_status -ne 'confirmed_interactively') { throw "Calibration not manually confirmed: $pending" }
foreach ($extension in @('.preview.jpg','.crop.jpg','.court_mask.png','.topdown.jpg','.json')) {
    $source = Join-Path $calibrationFolder ($pendingStem + $extension)
    $destination = Join-Path $calibrationFolder ($CameraId + $extension)
    if (Test-Path -LiteralPath $source -PathType Leaf) {
        Move-Item -LiteralPath $source -Destination $destination -Force:$Overwrite
    }
}
}
$analyzeArgs = @('analyze',$resolvedVideo,'--court-calibration',$calibration,'--batch-size',$BatchSize)
if ($FarRoi) {
    $analyzeArgs += '--far-roi'
    $analyzeArgs += @($FarRoi | ForEach-Object { $_.ToString() })
}
if ($Overwrite -or $OverwriteResults) { $analyzeArgs += '--overwrite' }
Write-Host 'Step 2/2: Running CUDA rally analysis...'
& $python $entryPoint @analyzeArgs
if ($LASTEXITCODE -eq 130 -or $LASTEXITCODE -eq -1073741510) { Write-Host 'Analysis cancelled. Saved calibration and completed outputs are preserved.'; return }
if ($LASTEXITCODE -ne 0) { throw "Analysis failed (exit $LASTEXITCODE). Check output status.json and error.log." }
Write-Host "Complete. Calibration: $calibration"
Write-Host ('Results: ' + (Join-Path (Join-Path $PSScriptRoot 'outputs') ([IO.Path]::GetFileNameWithoutExtension($resolvedVideo))))
