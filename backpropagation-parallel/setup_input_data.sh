#!/bin/bash

# Define variables
ZIP_FILE=$1
TARGET_DIR="input"

# 1. Check if an argument was provided
if [ -z "$ZIP_FILE" ]; then
    echo "Usage: $0 <path_to_zip_file>"
    echo "Example: $0 mnist_data.zip"
    exit 1
fi

# 2. Check if the specified zip file exists
if [ ! -f "$ZIP_FILE" ]; then
    echo "Error: File '$ZIP_FILE' not found."
    exit 1
fi

# 3. Create the target directory if it doesn't exist
if [ ! -d "$TARGET_DIR" ]; then
    echo "Creating '$TARGET_DIR' directory..."
    mkdir -p "$TARGET_DIR"
fi

# 4. Extract the contents
# -o: overwrite existing files without prompting
# -j: junk paths (ignore internal folder structure of the zip)
echo "Extracting '$ZIP_FILE' into '$TARGET_DIR'..."
unzip -o -j "$ZIP_FILE" -d "$TARGET_DIR"

# 5. Check if unzip succeeded
if [ $? -eq 0 ]; then
    echo "-----------------------------------------------"
    echo "Success! Content of $ZIP_FILE is now in $TARGET_DIR/"
    ls -1 "$TARGET_DIR"
else
    echo "Error: Extraction failed. Please check if 'unzip' is installed."
    exit 1
fi
