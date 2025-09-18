@echo off
echo Starting Advanced Trading Bot...
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.11+ and try again
    pause
    exit /b 1
)

REM Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt

REM Create necessary directories
if not exist "data" mkdir data
if not exist "logs" mkdir logs
if not exist "models" mkdir models
if not exist "static" mkdir static

REM Copy environment file if it doesn't exist
if not exist ".env" (
    echo Creating .env file from template...
    copy env_example.txt .env
    echo.
    echo IMPORTANT: Please edit .env file with your configuration before running the bot
    echo.
    pause
)

REM Start the trading bot
echo Starting trading bot in paper trading mode...
echo Dashboard will be available at: http://localhost:8000
echo Note: If running alongside Claude Bot, use port 8003 for dashboard
echo.
echo Press Ctrl+C to stop the bot
echo.

python main.py --mode bot

pause
