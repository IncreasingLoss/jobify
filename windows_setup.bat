@echo off
setlocal enabledelayedexpansion

:: =====================================================================
::  Jobify - Windows setup & launch script
::  - Pulls repo, installs Python 3.12, Ollama + qwen3:8b
::  - Creates venv, installs deps, runs setup notebook
::  - Launches Ollama service and the FastAPI backend
::  - Safe to re-run: skips steps that are already done
:: =====================================================================

set "BASE_DIR=%~dp0"
cd /d "%BASE_DIR%"
echo [i] Working directory: %BASE_DIR%

echo ==========================================
echo   Jobify initial setup
echo ==========================================

:: ---------------------------------------------------------------------
:: Upfront: do we have winget? (needed for any auto-install below)
:: ---------------------------------------------------------------------
set "HAS_WINGET=0"
where winget >nul 2>&1
if !errorlevel! equ 0 set "HAS_WINGET=1"
if "!HAS_WINGET!"=="0" (
    echo [!] winget was not found on this system.
    echo     Automatic installs of Python / Git / Ollama below will fail if missing.
    echo     You can get winget via the "App Installer" package in the Microsoft Store.
)

:: ---------------------------------------------------------------------
:: 1) Pull latest repository changes (needs git)
:: ---------------------------------------------------------------------
echo.
echo [1/6] Pulling latest repository changes...
where git >nul 2>&1
if !errorlevel! neq 0 (
    echo [!] Git not found. Installing via winget...
    if "!HAS_WINGET!"=="0" goto :no_winget
    winget install -e --id Git.Git --accept-source-agreements --accept-package-agreements
    if !errorlevel! neq 0 (
        echo [X] Failed to install Git automatically.
        echo     Please install it manually from https://git-scm.com and re-run.
        pause
        exit /b 1
    )
    call :RefreshPath
    where git >nul 2>&1
    if !errorlevel! neq 0 (
        echo [!] Git installed but this window can't see it yet.
        echo     Close this window, open a NEW terminal, and run the script again.
        pause
        exit /b 1
    )
)

git pull
if !errorlevel! neq 0 (
    echo [!] Git pull failed - continuing with the local copy already on disk.
)

:: ---------------------------------------------------------------------
:: 2) Check / install Python 3.12
:: ---------------------------------------------------------------------
echo.
echo [2/6] Checking for Python 3.12...
set "PYTHON_EXE="

:: Prefer the py launcher with an explicit version (most reliable)
where py >nul 2>&1
if !errorlevel! equ 0 (
    py -3.12 --version >nul 2>&1
    if !errorlevel! equ 0 set "PYTHON_EXE=py -3.12"
)

:: Fall back to `python` if it really is 3.12
if not defined PYTHON_EXE (
    where python >nul 2>&1
    if !errorlevel! equ 0 (
        python --version 2>nul | findstr /C:"3.12" >nul
        if !errorlevel! equ 0 set "PYTHON_EXE=python"
    )
)

if not defined PYTHON_EXE (
    echo [!] Python 3.12 not found in PATH.
    if "!HAS_WINGET!"=="0" goto :no_winget
    echo [i] Installing Python 3.12 via winget...
    winget install -e --id Python.Python.3.12 --accept-source-agreements --accept-package-agreements
    if !errorlevel! neq 0 (
        echo [X] Winget failed to install Python 3.12.
        echo     Please install it manually from https://www.python.org/downloads/windows/ and re-run.
        pause
        exit /b 1
    )
    call :RefreshPath
    py -3.12 --version >nul 2>&1
    if !errorlevel! equ 0 (
        set "PYTHON_EXE=py -3.12"
    ) else (
        python --version 2>nul | findstr /C:"3.12" >nul
        if !errorlevel! equ 0 (
            set "PYTHON_EXE=python"
        ) else (
            echo [!] Python 3.12 was installed but this window can't see it yet.
            echo     Close this window, open a NEW terminal, and run the script again.
            pause
            exit /b 1
        )
    )
)

echo [i] Using Python 3.12: !PYTHON_EXE!

:: ---------------------------------------------------------------------
:: 3) Check / install Ollama, then pull qwen3:8b
:: ---------------------------------------------------------------------
echo.
echo [3/6] Checking Ollama...
where ollama >nul 2>&1
if !errorlevel! neq 0 (
    echo [!] Ollama not found. Installing via winget...
    if "!HAS_WINGET!"=="0" goto :no_winget
    winget install -e --id Ollama.Ollama --accept-source-agreements --accept-package-agreements
    if !errorlevel! neq 0 (
        echo [X] Automated Ollama install failed.
        echo     Please install manually from https://ollama.com and re-run.
        pause
        exit /b 1
    )
    call :RefreshPath
    where ollama >nul 2>&1
    if !errorlevel! neq 0 (
        echo [!] Ollama installed but this window can't see it yet.
        echo     Close this window, open a NEW terminal, and run the script again.
        pause
        exit /b 1
    )
    echo [i] Giving the Ollama service a moment to start...
    timeout /t 8 /nobreak >nul
) else (
    echo [i] Ollama already installed.
)

:: Make sure the Ollama background service is actually responding
ollama list >nul 2>&1
if !errorlevel! neq 0 (
    echo [!] Ollama is installed but its service isn't responding yet.
    echo     Launch the Ollama app once from the Start Menu, then re-run this script.
    pause
    exit /b 1
)

:: Pull model only if not already present
echo [i] Checking for qwen3:8b model...
ollama list | findstr /C:"qwen3:8b" >nul
if !errorlevel! neq 0 (
    echo [i] Pulling model qwen3:8b ...
    ollama pull qwen3:8b
    if !errorlevel! neq 0 (
        echo [X] Model download failed. Check your connection and re-run.
        pause
        exit /b 1
    )
) else (
    echo [i] Model already present.
)

:: ---------------------------------------------------------------------
:: 4) Create / activate the virtual environment, install deps
:: ---------------------------------------------------------------------
echo.
echo [4/6] Setting up virtual environment and dependencies...
if not exist ".venv\Scripts\activate.bat" (
    echo [i] Creating virtual environment (.venv)...
    !PYTHON_EXE! -m venv .venv
    if !errorlevel! neq 0 (
        echo [X] Failed to create virtual environment.
        echo     Check Python installation and folder permissions.
        pause
        exit /b 1
    )
) else (
    echo [i] Virtual environment already exists.
)

call ".venv\Scripts\activate.bat"
if !errorlevel! neq 0 (
    echo [X] Failed to activate virtual environment.
    pause
    exit /b 1
)

python -m pip install --upgrade pip
if !errorlevel! neq 0 (
    echo [X] Failed to upgrade pip.
    pause
    exit /b 1
)

if exist "requirements.txt" (
    echo [i] Installing requirements.txt...
    python -m pip install -r requirements.txt
    if !errorlevel! neq 0 (
        echo [X] Failed to install dependencies from requirements.txt.
        pause
        exit /b 1
    )
) else (
    echo [!] requirements.txt not found - skipping dependency install.
)

:: ---------------------------------------------------------------------
:: 5) Execute the setup notebook
:: ---------------------------------------------------------------------
echo.
echo [5/6] Executing setup notebook...
python -m pip install nbconvert ipykernel
if !errorlevel! neq 0 (
    echo [X] Failed to install nbconvert / ipykernel.
    pause
    exit /b 1
)

set "NB_PATH=jobify\0_setup_jobspy.ipynb"
if not exist "%NB_PATH%" (
    echo [!] Notebook not found at %NB_PATH% - skipping execution.
) else (
    python -m nbconvert --to notebook --execute "%NB_PATH%" --ExecutePreprocessor.timeout=600
    if !errorlevel! neq 0 (
        echo [X] Notebook execution failed. Inspect the output above.
        pause
        exit /b 1
    )
)

:: ---------------------------------------------------------------------
:: 6) Start Ollama service and the API
:: ---------------------------------------------------------------------
echo.
echo [6/6] Starting Ollama service and the API...

:: Only start `ollama serve` if the port isn't already listening
netstat -ano | findstr ":11434" | findstr "LISTENING" >nul
if !errorlevel! neq 0 (
    start "Ollama" cmd /k "ollama serve --host 0.0.0.0 --port 11434"
    echo [i] Waiting for Ollama service to come up...
    timeout /t 5 /nobreak >nul
) else (
    echo [i] Ollama service already listening on 11434.
)

:: Resolve venv python + API dir, then launch the API in a new window
set "API_DIR=%BASE_DIR%jobify-api"
set "VENV_PY=%BASE_DIR%.venv\Scripts\python.exe"
if not exist "%VENV_PY%" (
    echo [X] Virtualenv python not found at %VENV_PY%
    pause
    exit /b 1
)
if not exist "%API_DIR%" (
    echo [X] API directory not found: %API_DIR%
    pause
    exit /b 1
)

netstat -ano | findstr ":8000" | findstr "LISTENING" >nul
if !errorlevel! neq 0 (
    :: Use /D for the working directory to avoid nested-cd quoting issues.
    :: The doubled outer quotes around the cmd /k argument are required by cmd's
    :: quote-stripping rules when the inner command itself contains quotes.
    start "Jobify API" /D "%API_DIR%" cmd /k ""%VENV_PY%" -m uvicorn api:app --host 127.0.0.1 --port 8000"
    echo [i] Waiting for the API to come up...
    timeout /t 8 /nobreak >nul
) else (
    echo [i] Something already listening on port 8000 - assuming API is up.
)

start "" "http://127.0.0.1:8000"

echo.
echo ==========================================
echo  Setup complete!
echo   - Ollama running in its own window
echo   - API running in its own window
echo   - Browser opening http://127.0.0.1:8000
echo ==========================================
pause
exit /b 0

:: =====================================================================
::  winget missing but we needed it
:: =====================================================================
:no_winget
echo [X] winget is required for automatic installs but is not available.
echo     Install Python 3.12, Git, and Ollama manually, then re-run.
pause
exit /b 1

:: =====================================================================
::  Helper: refresh PATH in this session from the registry, so that
::  something installed a few lines above (Python/Git/Ollama) can be
::  found without forcing the user to open a new terminal window.
:: =====================================================================
:RefreshPath
set "SYS_PATH="
set "USR_PATH="
for /f "skip=2 tokens=2,*" %%A in ('reg query "HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\Environment" /v Path 2^>nul') do set "SYS_PATH=%%B"
for /f "skip=2 tokens=2,*" %%A in ('reg query "HKCU\Environment" /v Path 2^>nul') do set "USR_PATH=%%B"
set "PATH=%SYS_PATH%;%USR_PATH%"
exit /b 0