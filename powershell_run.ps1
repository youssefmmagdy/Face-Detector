
# 2. Create venv if not exists
if (-not (Test-Path "venv")) {
    Write-Host "Creating virtual environment..."
    python -m venv venv
} else {
    Write-Host "Virtual environment already exists."
}

# 3. Activate venv and install requirements
$venvActivate = Join-Path $WORKSPACE "venv\Scripts\Activate.ps1"
. $venvActivate

Write-Host "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# 4. Download weights
Write-Host "Downloading weights..."
python scripts/download_weights.py

# 5. Run main app
Write-Host "Running yoloface_app/main.py..."
python yoloface_app/main.py
