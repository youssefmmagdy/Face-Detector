
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



# 4. Download weights if not already present
if (-not (Test-Path "model-weights/yolov3-wider_16000.weights")) {
    Write-Host "Downloading weights..."
    python download_weights.py
} else {
    Write-Host "Weights already exist. Skipping download."
}

# 5. Run main app
Write-Host "Running yoloface_app/main.py..."
python yoloface_app/main.py
