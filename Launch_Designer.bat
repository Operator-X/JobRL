@echo off
rem ==============================================================================
rem JobRL Visual Scenario Designer Launcher (Windows)
rem Double-click this file to start the Visual Scenario Designer.
rem ==============================================================================

cd /d "%~dp0"

echo ============================================================
echo          JobRL Visual Scenario Designer (Windows)
echo ============================================================

rem Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python was not found on your system.
    echo Please install Python (from https://www.python.org/downloads/) and make sure to check "Add Python to PATH".
    pause
    exit /b 1
)

rem Step 1: Check or create virtual environment
if not exist "venv" (
    echo [1/2] Creating Python virtual environment (venv)...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
)

rem Activate virtual environment
call venv\Scripts\activate.bat

rem Step 2: Check if Streamlit and dependencies are installed
python -c "import streamlit" >nul 2>&1
if %errorlevel% neq 0 (
    echo [2/2] First-time setup: Installing required libraries...
    echo This may take about 30-60 seconds. Please wait...
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo [ERROR] Dependency installation encountered an error.
        pause
        exit /b 1
    )
)

echo ============================================================
echo Launching Visual Designer in your default browser...
echo Close this window or press Ctrl+C when you are finished.
echo ============================================================

rem Launch Streamlit web app
streamlit run app.py --server.headless=false
pause
