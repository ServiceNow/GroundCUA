#!/bin/bash
# =============================================================================
# Download VideoCUA data from HuggingFace and extract ZIP files.
#
# This script:
#   1. Downloads the VideoCUA dataset from HuggingFace
#   2. Extracts platform ZIP files from raw_data/ into data/
#
# USAGE:
#   bash download_data.sh
#   bash download_data.sh --repo "AgentsResearch/ActCUA" --output_dir ./VideoCUA
#   bash download_data.sh --skip_download   # Only extract ZIPs
# =============================================================================

set -e

# -----------------------------------------------------------------------------
# CONFIGURATION (override via command-line arguments)
# -----------------------------------------------------------------------------
HF_REPO="AgentsResearch/ActCUA"
OUTPUT_DIR="./VideoCUA"
SKIP_DOWNLOAD=false

# -----------------------------------------------------------------------------
# ARGUMENT PARSING
# -----------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        --repo)
            HF_REPO="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --skip_download)
            SKIP_DOWNLOAD=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --repo REPO_ID        HuggingFace dataset repo (default: AgentsResearch/ActCUA)"
            echo "  --output_dir DIR       Local directory to download into (default: ./VideoCUA)"
            echo "  --skip_download        Skip download, only extract ZIPs"
            echo ""
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=============================================="
echo "VideoCUA Data Download & Extract"
echo "=============================================="
echo "  HF Repo:    $HF_REPO"
echo "  Output Dir: $OUTPUT_DIR"
echo "=============================================="
echo ""

# -----------------------------------------------------------------------------
# STEP 1: DOWNLOAD FROM HUGGINGFACE
# -----------------------------------------------------------------------------
if [ "$SKIP_DOWNLOAD" = false ]; then
    echo "[Step 1/2] Downloading dataset from HuggingFace..."

    mkdir -p "$OUTPUT_DIR"

    if command -v huggingface-cli &> /dev/null; then
        echo "Using huggingface-cli..."
        huggingface-cli download "$HF_REPO" --repo-type dataset --local-dir "$OUTPUT_DIR"
    elif command -v git &> /dev/null && git lfs version &> /dev/null 2>&1; then
        echo "Using git-lfs..."
        git clone "https://huggingface.co/datasets/$HF_REPO" "$OUTPUT_DIR"
    else
        echo "Error: Neither huggingface-cli nor git-lfs is installed."
        echo ""
        echo "Install one of the following:"
        echo "  1. pip install -U 'huggingface_hub[cli]'"
        echo "  2. sudo apt-get install git-lfs && git lfs install"
        exit 1
    fi

    echo "Download complete."
    echo ""
else
    echo "[Step 1/2] Skipping download."
    echo ""
fi

# -----------------------------------------------------------------------------
# STEP 2: EXTRACT ZIP FILES
# -----------------------------------------------------------------------------
RAW_DATA_DIR="$OUTPUT_DIR/raw_data"
DATA_DIR="$OUTPUT_DIR/data"

if [ ! -d "$RAW_DATA_DIR" ]; then
    echo "Error: raw_data directory not found at $RAW_DATA_DIR"
    echo "Make sure the dataset was downloaded correctly."
    exit 1
fi

# Check if data/ already exists with content
if [ -d "$DATA_DIR" ] && [ "$(ls -A "$DATA_DIR" 2>/dev/null)" ]; then
    EXISTING_COUNT=$(find "$DATA_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)
    echo "[Step 2/2] data/ directory already exists with $EXISTING_COUNT platform folders."
    echo "Skipping extraction. Delete $DATA_DIR to re-extract."
    echo ""
else
    echo "[Step 2/2] Extracting ZIP files from raw_data/ to data/..."
    mkdir -p "$DATA_DIR"

    ZIP_COUNT=$(find "$RAW_DATA_DIR" -name "*.zip" | wc -l)
    echo "Found $ZIP_COUNT ZIP files to extract."

    EXTRACTED=0
    FAILED=0

    for zip_file in "$RAW_DATA_DIR"/*.zip; do
        if [ ! -f "$zip_file" ]; then
            continue
        fi

        # Get platform name from ZIP filename (e.g., "Blender.zip" -> "Blender")
        platform_name=$(basename "$zip_file" .zip)
        platform_dir="$DATA_DIR/$platform_name"

        if [ -d "$platform_dir" ] && [ "$(ls -A "$platform_dir" 2>/dev/null)" ]; then
            echo "  [skip] $platform_name (already extracted)"
            EXTRACTED=$((EXTRACTED + 1))
            continue
        fi

        mkdir -p "$platform_dir"
        echo "  [extract] $platform_name..."

        if unzip -q -o "$zip_file" -d "$platform_dir" 2>/dev/null; then
            EXTRACTED=$((EXTRACTED + 1))
        else
            echo "  [FAILED] $platform_name"
            FAILED=$((FAILED + 1))
        fi
    done

    echo ""
    echo "Extraction complete: $EXTRACTED succeeded, $FAILED failed"
fi

# -----------------------------------------------------------------------------
# SUMMARY
# -----------------------------------------------------------------------------
if [ -d "$DATA_DIR" ]; then
    PLATFORM_COUNT=$(find "$DATA_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)
    TASK_COUNT=$(find "$DATA_DIR" -name "action_log.json" | wc -l)
    echo ""
    echo "=============================================="
    echo "Data ready!"
    echo "  Location:   $DATA_DIR"
    echo "  Platforms:  $PLATFORM_COUNT"
    echo "  Tasks:      $TASK_COUNT"
    echo "=============================================="
fi
