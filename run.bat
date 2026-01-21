@echo off
REM AURA-VAE Setup and Run Script for Windows
REM
REM This script:
REM 1. Creates a Python virtual environment
REM 2. Installs dependencies
REM 3. Runs the full pipeline

echo ============================================================
echo AURA-VAE Setup Script
echo ============================================================

cd /d "%~dp0"

REM Check if venv exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo Error: Failed to create virtual environment
        echo Make sure Python 3.8+ is installed and in PATH
        pause
        exit /b 1
    )
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install/upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo Error: Failed to install dependencies
    pause
    exit /b 1
)

REM Run the pipeline
echo.
echo ============================================================
echo Running AURA-VAE Pipeline
echo ============================================================
python run_pipeline.py %*

echo.
echo ============================================================
echo Done!
echo ============================================================
pause
