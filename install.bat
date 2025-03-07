@echo off

:: set "pythonfile=spectraProcessing.py"

setlocal
echo Checking for previous installation...
:: Get the user's Documents and Downloads folders
set "CHECK=%USERPROFILE%\Documents\Spectra Processing\Executable"
set "DOWNLOADS_FOLDER=%USERPROFILE%\Downloads"
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

    del "%EXTRACT_FOLDER%"
)

echo Checking for Python installation...

rem Check for multiple Python versions and select the highest
set "highestVersion="
set "pythonPath="

for /f "tokens=2 delims==" %%a in ('wmic datafile where name^="C:\\\\Python*\\\\python.exe" get Version /value ^| findstr "="') do (
    set "version=%%a"
    call :compareVersions "%%a"
)

if defined pythonPath (
    echo Using Python version: %highestVersion% at %pythonPath%
    set "python=%pythonPath%"
) else (
    where python >nul 2>&1
    if %errorlevel% == 0 (
        echo Python found in PATH.
        set "python=python"
    ) else (
        echo Python is not installed or not found in PATH.
        echo Exiting script.
        pause
        exit /b 1
    )
)

echo Checking if pip is installed...
"%python%" -m pip --version >nul 2>&1
if %errorlevel% == 0 (
    echo pip is already installed.
) else (
    echo Installing pip...
    "%python%" -m ensurepip --upgrade
    if %errorlevel% == 1 (
        echo Failed to install pip.
        echo Exiting script.
        pause
        exit /b 1
    )
)

echo Installing required libraries...
"%python%" -m pip install PyInstaller pandas matplotlib scipy numpy tk scikit-learn openpyxl xlrd scikit-learn-intelex
if %errorlevel% == 1 (
    echo Failed to install libraries.
    echo Exiting script.
    pause
    exit /b 1
)

echo Creating User Directories...
mkdir "%USERPROFILE%\Documents\Spectra Processing"
mkdir "%USERPROFILE%\Documents\Spectra Processing\Executable"
mkdir "%USERPROFILE%\Pictures\Spectra Processing"
copy "fluorophor_data.txt" "%USERPROFILE%\Documents\Spectra Processing\Executable"
copy "icon.ico" "%USERPROFILE%\Documents\Spectra Processing\Executable"

echo Creating executable using PyInstaller...
"%python%" -m PyInstaller --distpath "%USERPROFILE%\Documents\Spectra Processing\Executable" --noconsole --icon="icon.ico" --onefile "%pythonfile%"
if %errorlevel% == 1 (
    echo Failed to create executable.
    echo Exiting script.
    pause
    exit /b 1
)

set "target=%USERPROFILE%\Documents\Spectra Processing\Executable\spectraProcessing.exe"  
set "shortcut=%USERPROFILE%\Desktop\spectraProcessing.lnk" 
set "working_dir=%USERPROFILE%\Documents\Spectra Processing\Executable\"

@REM mklink /H "%shortcut%" "%target%"
@REM powershell  -Command "Start-Process -Verb RunAs powerhell -Command cd \\\"%working_dir%\\\" New-Item -ItemType SymbolicLink -Path '%shortcut%' -Target '%target%'"
powershell "$s=(New-Object -COM WScript.Shell).CreateShortcut('%shortcut%');$s.TargetPath='%target%'; $s.Save()"

@REM set SCRIPT="%TEMP%\%RANDOM%-%RANDOM%-%RANDOM%-%RANDOM%.vbs"
@REM echo Set oWS = WScript.CreateObject("WScript.Shell") >> %SCRIPT%
@REM echo sLinkFile = "%shortcut%" >> %SCRIPT%
@REM echo Set oLink = oWS.CreateShortcut(sLinkFile) >> %SCRIPT%
@REM echo oLink.TargetPath = "%target%" >> %SCRIPT%
@REM echo oLink.WorkingDirectory = "%working_dir%" >> %SCRIPT%
@REM echo oLink.IconLocation = "%working_dir%/icon.ico" >> %SCRIPT%
@REM echo oLink.Save >> %SCRIPT%

@REM cscript /nologo %SCRIPT%
@REM del %SCRIPT%


echo Executable created successfully!
pause
exit /b 0

:compareVersions
set "currentVersion=%~1"
if not defined highestVersion (
    set "highestVersion=%currentVersion%"
    for %%a in ("%currentVersion:\= %") do set "pythonPath=C:\Python%%~na\python.exe"
) else (
    for /f "tokens=1-4 delims=." %%a in ("%highestVersion%") do set "hv1=%%a" & set "hv2=%%b" & set "hv3=%%c" & set "hv4=%%d"
    for /f "tokens=1-4 delims=." %%a in ("%currentVersion%") do set "cv1=%%a" & set "cv2=%%b" & set "cv3=%%c" & set "cv4=%%d"
    if %cv1% GTR %hv1% (
        set "highestVersion=%currentVersion%"
        for %%a in ("%currentVersion:\= %") do set "pythonPath=C:\Python%%~na\python.exe"
    ) else if %cv1% EQU %hv1% (
        if %cv2% GTR %hv2% (
            set "highestVersion=%currentVersion%"
            for %%a in ("%currentVersion:\= %") do set "pythonPath=C:\Python%%~na\python.exe"
        ) else if %cv2% EQU %hv2% (
            if %cv3% GTR %hv3% (
                set "highestVersion=%currentVersion%"
                for %%a in ("%currentVersion:\= %") do set "pythonPath=C:\Python%%~na\python.exe"
            ) else if %cv3% EQU %hv3% (
                if %cv4% GTR %hv4% (
                    set "highestVersion=%currentVersion%"
                    for %%a in ("%currentVersion:\= %") do set "pythonPath=C:\Python%%~na\python.exe"
                )
            )
        )
    )
)
exit /b