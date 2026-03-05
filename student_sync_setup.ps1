# One command for students: update code + setup environment.
Set-Location $PSScriptRoot
$ErrorActionPreference = "Stop"

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  Student Sync + Setup" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

try {
    if ((Test-Path ".git") -and (Get-Command git -ErrorAction SilentlyContinue)) {
        Write-Host "[1/2] Pulling latest changes..." -ForegroundColor Yellow
        git pull --ff-only
        if ($LASTEXITCODE -ne 0) {
            throw "git pull failed. Commit/stash local changes, then rerun."
        }
    } else {
        Write-Host "[1/2] Git repo not detected (or git missing) - skipping pull." -ForegroundColor Yellow
    }

    Write-Host "[2/2] Running setup..." -ForegroundColor Yellow
    & "$PSScriptRoot\setup.ps1"
    if ($LASTEXITCODE -ne 0) {
        throw "setup.ps1 failed. See messages above."
    }

    Write-Host "`nDone. Start coding with:" -ForegroundColor Green
    Write-Host "  .\venv\Scripts\Activate.ps1" -ForegroundColor White
    Write-Host "  python src/01_cnn/cnn_image_classifier.py" -ForegroundColor White
} catch {
    Write-Host "`nStudent sync/setup failed: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
