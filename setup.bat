@echo off
REM Check if Python 3.10 is installed using the 'py' launcher
py -3.10 --version >nul 2>&1
if errorlevel 1 (
    echo Python 3.10 is not installed. Please install Python 3.10 and add it to your PATH.
    pause
    exit /b 1
)

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating virtual environment...
    py -3.10 -m venv venv
)

REM Activate the virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
pip install --upgrade pip

REM Install dependencies from requirements.txt
echo Installing dependencies...
pip install -r requirements.txt

echo.
echo Setup complete.
echo To activate the environment later, run: venv\Scripts\activate.bat
pause
