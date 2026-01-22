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

# 3. Download weights
echo "Downloading weights..."
python download_weights.py

# 4. Run main app
echo "Running yoloface_app/main.py..."
python yoloface_app/main.py
