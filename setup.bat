@echo off

:: set "pythonfile=spectraProcessing.py"

setlocal
echo Checking for previous installation...
:: Get the user's Documents and Downloads folders
set "CHECK=%USERPROFILE%\Documents\Spectra Processing\Executable"
set "DOWNLOADS_FOLDER=%USERPROFILE%\Downloads"
set "SCRIPT_DIR=%~dp0"
:: for /f "delims=" %%I in ('powershell -command "[System.IO.Path]::Combine($env:USERPROFILE, 'Documents', 'Spectra Processing', 'Executable', 'spectraProcessing.exe')"') do set "CHECK_FILE=%%I"
:: for /f "delims=" %%I in ('powershell -command "[System.IO.Path]::Combine($env:USERPROFILE, 'Downloads')"') do set "DOWNLOADS_FOLDER=%%I"

:: Check if the required file exists
if exist "%CHECK_FILE%" (
    :: Define the repository URL and destination folder
    set "REPO_URL=https://github.com/somark2000/Spectra-Processing-Tool/archive/refs/heads/main.zip"
    set "ZIP_FILE=%DOWNLOADS_FOLDER%\Spectra-Processing-Tool.zip"
    set "EXTRACT_FOLDER=%DOWNLOADS_FOLDER%\Spectra-Processing-Tool"

    :: Download the repository ZIP using PowerShell
    powershell -command "(New-Object Net.WebClient).DownloadFile('%REPO_URL%', '%ZIP_FILE%')"

    :: Create extraction folder if it doesn't exist
    if not exist "%EXTRACT_FOLDER%" mkdir "%EXTRACT_FOLDER%"

    :: Extract the ZIP file
    powershell -command "Expand-Archive -Path '%ZIP_FILE%' -DestinationPath '%EXTRACT_FOLDER%' -Force"

    :: Remove the ZIP file
    del "%ZIP_FILE%"

    echo Repository downloaded and extracted to: %EXTRACT_FOLDER%
    :: set "pythonfile=%EXTRACT_FOLDER%\spectraProcessing.py"
    :: Get the directory of the script
    set "SCRIPT_DIR=%~dp0"

    :: Remove trailing backslash if present
    set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
    copy "%EXTRACT_FOLDER%\spectraProcessing.py" "%SCRIPT_DIR%"
    copy "%EXTRACT_FOLDER%\fluorophor_data.txt" "%SCRIPT_DIR%"
    copy "%EXTRACT_FOLDER%\install.sh" "%SCRIPT_DIR%"
    copy "%EXTRACT_FOLDER%\install.bat" "%SCRIPT_DIR%"

    del "%EXTRACT_FOLDER%"
)

call "%SCRIPT_DIR%\install.bat"
pause