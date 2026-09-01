@echo off
title AI Human Fall Detection System
echo ===================================================
echo   Starting AI Human Fall Detection Desktop App...
echo ===================================================

if exist "dist\FallDetectionApp\FallDetectionApp.exe" (
    echo Launching compiled executable...
    start "" "dist\FallDetectionApp\FallDetectionApp.exe"
) else (
    echo Compiled executable not found, running from python source...
    python app_gui.py
)
