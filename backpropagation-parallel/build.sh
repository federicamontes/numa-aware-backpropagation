#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

ZIP_FILE=$1

# 1. Check if the zip file argument was provided
if [ -z "$ZIP_FILE" ]; then
    echo "Usage: $0 <path_to_zip_file>"
    exit 1
fi

# 2. Run the Data Setup script
if [ -f "./setup_input_data.sh" ]; then
    echo "--- Step 1: Setting up data ---"
    chmod +x setup_input_data.sh
    ./setup_input_data.sh "$ZIP_FILE"
else
    echo "Error: setup_input_data.sh not found!"
    exit 1
fi

# 3. Create and enter the build directory
echo "--- Step 2: Preparing Build Directory ---"
if [ -d "build" ]; then
    echo "Cleaning existing build directory..."
    rm -rf build
fi
mkdir build
cd build

# 4. Run CMake and Make
echo "--- Step 3: Compiling Project ---"
cmake ..
make

echo "--- Build Complete! ---"
echo "You can now run your project from the build folder."
