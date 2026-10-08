@echo off
setlocal

pushd "%~dp0"
if errorlevel 1 (
    echo [ERROR] Cannot open the project directory.
    pause
    exit /b 1
)

set "PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%PYTHON%" (
    echo [ERROR] Project virtual environment was not found.
    echo Run these commands in the project directory first:
    echo   py -3 -m venv .venv
    echo   .venv\Scripts\python.exe -m pip install -r requirement.txt
    set "EXIT_CODE=1"
    goto finish
)

"%PYTHON%" -c "import jupyterlab" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Cannot load JupyterLab using the project Python.
    echo Check the virtual environment and install the dependencies:
    echo   .venv\Scripts\python.exe -m pip install -r requirement.txt
    set "EXIT_CODE=1"
    goto finish
)

echo Starting JupyterLab in the project directory...
echo Keep this window open. Press Ctrl+C to stop the server.
"%PYTHON%" -m jupyterlab --ServerApp.ip=127.0.0.1 %*
set "EXIT_CODE=%ERRORLEVEL%"

:finish
popd
if not "%EXIT_CODE%"=="0" (
    echo [ERROR] JupyterLab did not exit successfully. Exit code: %EXIT_CODE%
    pause
)
endlocal & exit /b %EXIT_CODE%
