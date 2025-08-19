@echo off
echo.
echo ========================================
echo   Integrated Web Chatbot System
echo ========================================
echo.
echo Starting the integrated web chatbot system...
echo.
echo This will:
echo - Start automatic web scraping
echo - Launch the Flask web server
echo - Open your default browser
echo.
echo Press any key to continue...
pause >nul

echo.
echo Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.7+ and try again
    pause
    exit /b 1
)

echo.
echo Installing/updating required packages...
pip install -r requirements.txt

echo.
echo Starting the system...
echo.
echo The web interface will open automatically in your browser
echo If it doesn't, go to: http://localhost:5000
echo.
echo Press Ctrl+C to stop the system
echo.

python run_system.py

echo.
echo System stopped.
pause
