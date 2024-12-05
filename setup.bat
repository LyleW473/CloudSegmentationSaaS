@echo off

REM Create a virtual environment
python -m venv venv

REM Activate the virtual environment
call venv\Scripts\activate

REM Upgrade pip
python -m pip install --upgrade pip

REM Install dependencies from requirements.txt
pip install --no-cache-dir -r requirements.txt

REM Install PyTorch (Only CPU version)
pip install --no-cache-dir torch torchvision torchaudio

REM Install FastAPI
pip install --no-cache-dir "fastapi[standard]"

REM Pause to keep the window open
pause
