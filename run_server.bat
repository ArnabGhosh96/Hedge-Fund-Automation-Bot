@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo [SYSTEM] Missing .venv. Create it and install requirements.txt first.
  exit /b 1
)
".venv\Scripts\python.exe" main.py >> server.log 2>&1
