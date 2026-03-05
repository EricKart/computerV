<# ================================================================
    Computer Vision Project - One-Click Setup (Windows PowerShell)
   ================================================================
   Usage:  .\setup.ps1
   This script creates a virtual environment and installs every
   dependency listed in requirements.txt.
#>

Set-Location $PSScriptRoot
$ErrorActionPreference = "Stop"

function Invoke-BasePython {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Args
    )

    if (Get-Command py -ErrorAction SilentlyContinue) {
        & py -3 @Args
    } elseif (Get-Command python -ErrorAction SilentlyContinue) {
        & python @Args
    } else {
        throw "Python 3 was not found. Install Python 3.10+ and rerun setup."
    }

    if ($LASTEXITCODE -ne 0) {
        throw "Base Python command failed: $($Args -join ' ')"
    }
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  Computer Vision Project - Setup" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

try {
    # Step 1 - Check Python
    Write-Host "[1/5] Checking Python installation..." -ForegroundColor Yellow
    Invoke-BasePython @("--version")

    # Step 2 - Create virtual environment
    if (-Not (Test-Path "venv")) {
        Write-Host "[2/5] Creating virtual environment..." -ForegroundColor Yellow
        Invoke-BasePython @("-m", "venv", "venv")
    } else {
        Write-Host "[2/5] Virtual environment already exists - skipping." -ForegroundColor Green
    }

    $venvPython = Join-Path $PSScriptRoot "venv\Scripts\python.exe"
    if (-Not (Test-Path $venvPython)) {
        throw "Virtual environment Python not found at venv\\Scripts\\python.exe"
    }

    # Step 3 - Upgrade pip (inside venv)
    Write-Host "[3/5] Upgrading pip in virtual environment..." -ForegroundColor Yellow
    & $venvPython -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed." }

    # Step 4 - Install requirements (inside venv)
    if (-Not (Test-Path "requirements.txt")) {
        throw "requirements.txt not found in $PSScriptRoot"
    }
    Write-Host "[4/5] Installing dependencies from requirements.txt..." -ForegroundColor Yellow
    & $venvPython -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed." }

    # Step 5 - Create necessary directories
    Write-Host "[5/5] Ensuring project directories exist..." -ForegroundColor Yellow
    $dirs = @("data", "models", "outputs", "logs")
    foreach ($d in $dirs) {
        if (-Not (Test-Path $d)) {
            New-Item -ItemType Directory -Path $d | Out-Null
        }
    }

    Write-Host "`n========================================" -ForegroundColor Green
    Write-Host "  Setup complete!" -ForegroundColor Green
    Write-Host "  Activate venv with: .\venv\Scripts\Activate.ps1" -ForegroundColor White
    Write-Host "  Or run directly: .\venv\Scripts\python.exe your_script.py" -ForegroundColor White
    Write-Host "========================================`n" -ForegroundColor Green
} catch {
    Write-Host "`nSetup failed: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "Troubleshooting:" -ForegroundColor Yellow
    Write-Host "  1) Install Python 3.10+ and ensure it is on PATH." -ForegroundColor Yellow
    Write-Host "  2) Run this script from project root: .\setup.ps1" -ForegroundColor Yellow
    Write-Host "  3) Retry with: powershell -ExecutionPolicy Bypass -File .\setup.ps1" -ForegroundColor Yellow
    exit 1
}
