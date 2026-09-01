@echo off
title Build Fall Detection Standalone Executable
echo =========================================================
echo   Building Fall Detection App using PyInstaller...
echo =========================================================

py -3.12 build_exe.py
if errorlevel 1 (
    python build_exe.py
)
pause
