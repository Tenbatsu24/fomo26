#!/usr/bin/env bash

PROJECT_ID="syn77192481"
SOURCE_DIR="."

for file in "$SOURCE_DIR"/Task{2,3,4,5,6}*.sif; do
    # Skip if no .sif files matched
    [ -e "$file" ] || continue

    echo "Uploading: $file"
    synapse store "$file" --parentId "$PROJECT_ID"
done
