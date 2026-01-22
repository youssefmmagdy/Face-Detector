#!/bin/bash



# 1. Create venv if not exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
else
    echo "Virtual environment already exists."
fi

# 2. Activate venv and install requirements
source venv/Scripts/activate


echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt



# 3. Download weights if not already present
if [ ! -f "model-weights/yolov3-wider_16000.weights" ]; then
    echo "Downloading weights..."
    python download_weights.py
else
    echo "Weights already exist. Skipping download."
fi

# 4. Run main app
echo "Running yoloface_app/main.py..."
python yoloface_app/main.py
