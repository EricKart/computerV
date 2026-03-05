#!/usr/bin/env bash
# ================================================================
#  Computer Vision Project – One-Click Setup (Linux / macOS)
# ================================================================
#  Usage:  chmod +x setup.sh && ./setup.sh
# ================================================================

set -e

echo ""
echo "========================================"
echo "  Computer Vision Project – Setup"
echo "========================================"
echo ""

# ── Step 1 – Create virtual environment ─────────────────
if [ ! -d "venv" ]; then
    echo "[1/4] Creating virtual environment..."
    python3 -m venv venv
else
    echo "[1/4] Virtual environment already exists – skipping."
fi

# ── Step 2 – Activate ───────────────────────────────────
echo "[2/4] Activating virtual environment..."
source venv/bin/activate

# ── Step 3 – Upgrade pip ────────────────────────────────
echo "[3/4] Upgrading pip..."
pip install --upgrade pip

# ── Step 4 – Install requirements ───────────────────────
echo "[4/4] Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# ── Create necessary directories ────────────────────────
mkdir -p data models outputs logs

echo ""
echo "========================================"
echo "  Setup complete!  Activate venv with:"
echo "  source venv/bin/activate"
echo "========================================"
echo ""
