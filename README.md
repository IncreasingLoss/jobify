# Jobify Windows Setup

This repository includes a Windows setup script for non-technical users.

## Initial setup steps

1. Open the repository folder in File Explorer.
2. Double-click `windows_setup.bat`.
3. Follow any on-screen prompts.

The script will:

- pull the latest repository changes from Git
- check for Python 3.12
- install Python 3.12 using `winget` if needed
- create a virtual environment in `.venv`
- install dependencies from `requirements.txt`
- execute `jobify\0_setup_jobspy.ipynb`
- start the API and open `http://127.0.0.1:8000`

## Requirements

- Git must be installed and available on `PATH`
- Internet access to download Python and Python packages
- write permission inside this repository folder
- `winget` is required only for auto-installing Python 3.12

## If automatic Python installation fails

1. Download and install Python 3.12 manually from https://www.python.org/downloads/windows/
2. Make sure the Python installer adds Python to `PATH`.
3. Re-run `windows_setup.bat`.

## Troubleshooting

- If the script prints an error, read the red text and the command output shown above.
- If Git is missing, install it from https://git-scm.com/download/win
- If `winget` is missing, install it from the Microsoft Store or update Windows App Installer

## Note

The script is intended for initial setup only. Once setup is complete, use the API at `http://127.0.0.1:8000`.
