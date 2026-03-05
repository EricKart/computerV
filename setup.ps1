<# ================================================================
    Computer Vision Project – One-Click Setup (Windows PowerShell)
   ================================================================
   Usage:  .\setup.ps1
   This script creates a virtual environment, activates it, and
   installs every dependency listed in requirements.txt.
#>

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  Computer Vision Project – Setup" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# ── Step 1 – Create virtual environment ─────────────────
if (-Not (Test-Path "venv")) {
    Write-Host "[1/4] Creating virtual environment..." -ForegroundColor Yellow
    python -m venv venv
} else {
    Write-Host "[1/4] Virtual environment already exists – skipping." -ForegroundColor Green
}

# ── Step 2 – Activate ───────────────────────────────────
Write-Host "[2/4] Activating virtual environment..." -ForegroundColor Yellow
& .\venv\Scripts\Activate.ps1

# ── Step 3 – Upgrade pip ────────────────────────────────
Write-Host "[3/4] Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# ── Step 4 – Install requirements ───────────────────────
Write-Host "[4/4] Installing dependencies from requirements.txt..." -ForegroundColor Yellow
pip install -r requirements.txt

# ── Create necessary directories ────────────────────────
$dirs = @("data", "models", "outputs", "logs")
foreach ($d in $dirs) {
    if (-Not (Test-Path $d)) { New-Item -ItemType Directory -Path $d | Out-Null }
}

Write-Host "`n========================================" -ForegroundColor Green
Write-Host "  Setup complete!  Activate venv with:" -ForegroundColor Green
Write-Host "  .\venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "========================================`n" -ForegroundColor Green
