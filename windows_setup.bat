@echo off
setlocal enabledelayedexpansion

echo ==========================================
echo Jobify initial setup
echo ==========================================

rem Ensure we run from the repository root
cd /d "%~dp0"

set "ERROR_CMD="

:run_step
necho.
echo 1) Pulling latest repository changes...
where git >nul 2>&1
if errorlevel 1 (
    echo ERROR: Git is not installed or not on PATH. Please install Git first.
    pause
    exit /b 1
)
set "ERROR_CMD=git pull"
git pull
call :check_exit "Failed to pull repository. Please check your network and git settings."

echo.
echo 2) Checking for Python 3.12...
set PYTHON_EXE=

for %%I in (py python) do (
    if not defined PYTHON_EXE (
        where %%I >nul 2>&1
        if not errorlevel 1 (
            for /f "delims=" %%V in ('%%I -c "import sys; sys.stdout.write(''{}.{}'' .format(sys.version_info[0], sys.version_info[1]))" 2^>nul') do (
                if "%%V"=="3.12" set PYTHON_EXE=%%I
            )
        )
    )
)

if not defined PYTHON_EXE (
    echo Python 3.12 was not found on this machine.
    echo Attempting to install Python 3.12 using winget...
    where winget >nul 2>&1
    if errorlevel 1 (
        echo ERROR: winget is not available.
        echo Please install Python 3.12 manually from https://www.python.org/downloads/windows/ and re-run this setup.
        pause
        exit /b 1
    )
    set "ERROR_CMD=winget install --id Python.Python.3.12 -e --source winget"
    winget install --id Python.Python.3.12 -e --source winget
    call :check_exit "Python 3.12 installation failed. Please install Python 3.12 manually and re-run this setup."
    set PYTHON_EXE=py
)

echo Found Python 3.12 as %PYTHON_EXE%.

echo.
echo 3) Checking Ollama and pulling model qwen3:8b...
where ollama >nul 2>&1
if errorlevel 1 (
    echo Ollama is not installed. Attempting installation with winget...
    where winget >nul 2>&1
    if errorlevel 1 (
        echo ERROR: winget is not available.
        echo Please install Ollama manually from https://ollama.com and re-run this setup.
        pause
        exit /b 1
    )
    set "ERROR_CMD=winget install --id Ollama.Ollama -e --source winget"
    winget install --id Ollama.Ollama -e --source winget
    call :check_exit "Ollama installation failed. Please install Ollama manually and re-run this setup."
) else (
    echo Ollama is already installed.
)

set "ERROR_CMD=ollama pull qwen3:8b"
ollama pull qwen3:8b
call :check_exit "Failed to pull Ollama model qwen3:8b. Please check your network and try again."

echo.
echo 4) Creating virtual environment and installing dependencies...
if not exist .venv (
    set "ERROR_CMD=%PYTHON_EXE% -m venv .venv"
    %PYTHON_EXE% -m venv .venv
    call :check_exit "Failed to create virtual environment. Please check Python installation and folder permissions."
) else (
    echo Virtual environment already exists.
)

set "ERROR_CMD=.venv\Scripts\activate"
call .venv\Scripts\activate
call :check_exit "Failed to activate virtual environment."

set "ERROR_CMD=python -m pip install --upgrade pip"
python -m pip install --upgrade pip
call :check_exit "Failed to upgrade pip."

set "ERROR_CMD=python -m pip install -r requirements.txt"
python -m pip install -r requirements.txt
call :check_exit "Failed to install dependencies from requirements.txt. Please inspect output above."

echo.
echo 5) Executing setup notebook...
set "ERROR_CMD=python -m nbconvert --to notebook --execute \"jobify\0_setup_jobspy.ipynb\" --ExecutePreprocessor.timeout=600"
python -m pip install nbconvert
call :check_exit "Failed to install nbconvert."
python -m nbconvert --to notebook --execute "jobify\0_setup_jobspy.ipynb" --ExecutePreprocessor.timeout=600
call :check_exit "Notebook execution failed. Please inspect the notebook output above."

echo.
echo 6) Starting Ollama service and the API...
set "ERROR_CMD=start Ollama"
start "Ollama" cmd /k "ollama serve --host 0.0.0.0 --port 11434"

set "ERROR_CMD=start API"
start "Jobify API" cmd /k "cd /d "%~dp0jobify-api" && "%~dp0.venv\Scripts\python" -m uvicorn api:app --host 127.0.0.1 --port 8000"

timeout /t 8 /nobreak >nul
start "" "http://127.0.0.1:8000"

echo Setup complete.
pause

exit /b 0

:check_exit
if errorlevel 1 (
    echo.
    echo ERROR: %~1
    if defined ERROR_CMD echo Failed command: %ERROR_CMD%
    echo.
    pause
    exit /b 1
)
goto :eof
