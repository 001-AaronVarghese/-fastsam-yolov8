@echo off
echo Starting ADAS Ensemble Server...
echo Please wait a moment for the models to load.

:: Wait 3 seconds, then open the default web browser
timeout /t 3 /nobreak >nul
start http://127.0.0.1:5000

:: Start the Flask backend
python web_app.py

pause