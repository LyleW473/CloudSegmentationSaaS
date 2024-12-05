#!/bin/bash

# Update the package list and install system dependencies
echo "Updating the package list and installing system dependencies..."
apt-get update
apt-get install -y libhdf5-dev libnetcdf-dev

# Create a virtual environment
echo "Creating a virtual environment..."
python3 -m venv venv

# Activate the virtual environment
echo "Activating the virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
python3 -m pip install --upgrade pip

# Install dependencies from requirements.txt
echo "Installing dependencies from requirements.txt..."
pip install --no-cache-dir -r requirements.txt

# Install PyTorch (Only CPU version)
echo "Installing PyTorch (CPU version)..."
pip install --no-cache-dir torch torchvision torchaudio # --index-url https://download.pytorch.org/whl/cu121

# Install FastAPI
echo "Installing FastAPI..."
pip install --no-cache-dir "fastapi[standard]"