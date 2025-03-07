#!/bin/bash

set -e  # Exit on error

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
$python -m PyInstaller --distpath "$BASE_DIR" --noconfirm --noconsole --icon="$BASE_DIR/icon.ico" --onefile spectraProcessing.py

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