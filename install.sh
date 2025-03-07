#!/bin/bash
set -e  # Exit on error

# Get the user's home directory
HOME_DIR="$HOME"

# Define the file to check for existence
CHECK_FILE="$HOME_DIR/Documents/Spectra Processing/Executable/spectraProcessing.exe"

# Define the repository URL and destination folder
REPO_URL="https://github.com/somark2000/Spectra-Processing-Tool/archive/refs/heads/main.zip"
DOWNLOADS_FOLDER="$HOME_DIR/Downloads"
ZIP_FILE="$DOWNLOADS_FOLDER/Spectra-Processing-Tool.zip"
EXTRACT_FOLDER="$DOWNLOADS_FOLDER/Spectra-Processing-Tool"

# Check if the required file exists
if [ -f "$CHECK_FILE" ]; then
    # Download the repository ZIP file
    echo "Downloading repository..."
    curl -L -o "$ZIP_FILE" "$REPO_URL"

    # Create the extraction folder if it doesn't exist
    mkdir -p "$EXTRACT_FOLDER"

    # Extract the ZIP file
    echo "Extracting repository..."
    unzip -o "$ZIP_FILE" -d "$EXTRACT_FOLDER"

    # Remove the ZIP file after extraction
    rm "$ZIP_FILE"
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    echo "Repository downloaded and extracted to: $EXTRACT_FOLDER"
    cp "$EXTRACT_FOLDER/spectraprocessing.py" "$SCRIPT_DIR"

fi


# Check if Python is installed
echo "Checking for Python installation..."
if command -v python3 &>/dev/null; then
    python="python3"
elif command -v python &>/dev/null; then
    python="python"
else
    echo "Python is not installed. Please install Python and rerun this script."
    exit 1
fi

echo "Using Python: $(which $python)"

# Ensure pip is installed
echo "Checking if pip is installed..."
$python -m ensurepip --default-pip
$python -m pip install --upgrade pip

# Install required Python libraries
echo "Installing required libraries..."
$python -m pip install --upgrade PyInstaller pandas matplotlib scipy numpy tk scikit-learn openpyxl xlrd FuzzyTM gensim

# Create user directories
BASE_DIR="$HOME/Documents/Spectra Processing/Executable"
PICTURES_DIR="$HOME/Pictures/Spectra Processing"
mkdir -p "$BASE_DIR" "$PICTURES_DIR"

# Copy necessary files
cp fluorophor_data.txt "$BASE_DIR"
cp icon.ico "$BASE_DIR"

# Create executable using PyInstaller
echo "Creating executable..."
$python -m PyInstaller --distpath "$BASE_DIR" --noconfirm --noconsole --icon="$BASE_DIR/icon.ico" --onefile "spectraProcessing.py"

# Handle macOS specific installation
if [[ "$(uname)" == "Darwin" ]]; then
    MACOS_APP_DIR="/Applications/SpectraProcessing.app"
    mkdir -p "$MACOS_APP_DIR/Contents/MacOS"
    cp "$BASE_DIR/spectraProcessing" "$MACOS_APP_DIR/Contents/MacOS/"
    echo "Application installed in $MACOS_APP_DIR"
else
    # Create a desktop shortcut for Linux
    SHORTCUT="$HOME/Desktop/spectraProcessing.desktop"
    echo "Creating shortcut at $SHORTCUT"
    cat <<EOF > "$SHORTCUT"
[Desktop Entry]
Version=1.0
Type=Application
Name=Spectra Processing
Exec=$BASE_DIR/spectraProcessing
Icon=$BASE_DIR/icon.ico
Terminal=false
EOF
    chmod +x "$SHORTCUT"
fi

echo "Installation complete!"
echo "Executable is located at $BASE_DIR/spectraProcessing"
rm "$EXTRACT_FOLDER"