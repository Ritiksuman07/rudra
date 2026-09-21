@echo off
REM RUDRA alpha v0.1 Quick Start
REM =================================

echo [RUDRA] Initializing project...
cd /d "%~dp0"

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.10+.
    pause
    exit /b 1
)

REM Run data generation
echo [RUDRA] Generating training data...
python src\generate_data.py

REM Check if data generation succeeded
if not exist "data\rudra_combined_train.jsonl" (
    echo [ERROR] Data generation failed.
    pause
    exit /b 1
)

echo [RUDRA] Data generated successfully!
echo [RUDRA] Run tests: python tests\__init__.py
echo [RUDRA] Run training: python src\train.py --config configs\train_config.yaml --start-stage 1

pause