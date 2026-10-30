@echo off
title SAFEPASS Multi-Camera Strategic Topology Tracker & Web UI
echo =========================================================================
echo SAFEPASS MULTI-CAMERA STRATEGIC TOPOLOGY TRACKER & OPERATOR DASHBOARD
echo Dataset: Mayfair Apartment 8-Camera CCTV Surveillance Network
echo =========================================================================
echo.

set PYTHON_EXE=C:\Users\zain sulaiman\AppData\Local\Python\bin\python.exe
if not exist "%PYTHON_EXE%" (
    set PYTHON_EXE=c:\project\safepass_appearance\venv\Scripts\python.exe
)
if not exist "%PYTHON_EXE%" (
    set PYTHON_EXE=python
)

echo [SAFEPASS] Starting FastAPI server on http://localhost:8000 ...
echo [SAFEPASS] Idle cameras consume 0%% GPU. Active tracking & neighbor handoffs enabled.
echo.

"%PYTHON_EXE%" -m uvicorn web.app:app --host 0.0.0.0 --port 8000

pause
