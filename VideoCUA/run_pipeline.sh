#!/bin/bash
# =============================================================================
# VideoCUA: End-to-End Trajectory Synthesis Pipeline
#
# This script runs the complete pipeline:
#   1. Download VideoCUA data from HuggingFace
#   2. Extract ZIP files
#   3. Convert raw data to trace format
#   4. Generate CoT (Chain-of-Thought) trajectory annotations
#
# USAGE:
#   bash run_pipeline.sh
#   bash run_pipeline.sh --skip_download --skip_convert
#   bash run_pipeline.sh --model openai/gpt-4o --num_threads 8
# =============================================================================

set -e

# Get the directory where this script lives
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# -----------------------------------------------------------------------------
# DEFAULT CONFIGURATION
# -----------------------------------------------------------------------------

# Data source
HF_REPO="ServiceNow/VideoCUA"         # HuggingFace dataset repository
DATA_DIR="./VideoCUA"                    # Where to download/find the raw data

# Conversion
OUTPUT_DIR="./videocua_processed"        # Where to put processed data
NUM_WORKERS=4                            # Parallel workers for video frame extraction
PLATFORMS=""                             # Filter platforms (comma-separated, empty=all)

# CoT Generation
MODEL="anthropic/claude-sonnet-4.5"      # LLM for CoT generation
SUFFIX="cot_v1"                          # Experiment suffix
NUM_THREADS=4                            # Parallel threads for LLM calls
MAX_NUM=""                               # Max tasks to process (empty=all)
NEED_DOUBLE_CHECK=false                  # Enable LLM double-check pass
BASE_URL=""                              # Custom API endpoint (for vLLM, etc.)

# Skip flags
SKIP_DOWNLOAD=false
SKIP_CONVERT=false
SKIP_COT=false

# Supported model formats:
#   OpenRouter (recommended): anthropic/claude-sonnet-4.5, openai/gpt-4o
#   Direct Anthropic:         claude-sonnet-4-5-20250929
#   Direct OpenAI:            gpt-4o
#   vLLM local:               YourModel --base_url http://localhost:8000/v1

# -----------------------------------------------------------------------------
# ARGUMENT PARSING
# -----------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        --hf_repo)       HF_REPO="$2";       shift 2 ;;
        --data_dir)      DATA_DIR="$2";       shift 2 ;;
        --output_dir)    OUTPUT_DIR="$2";     shift 2 ;;
        --num_workers)   NUM_WORKERS="$2";    shift 2 ;;
        --platforms)     PLATFORMS="$2";       shift 2 ;;
        --model)         MODEL="$2";          shift 2 ;;
        --suffix)        SUFFIX="$2";         shift 2 ;;
        --num_threads)   NUM_THREADS="$2";    shift 2 ;;
        --max_num)       MAX_NUM="$2";        shift 2 ;;
        --base_url)      BASE_URL="$2";       shift 2 ;;
        --need_double_check) NEED_DOUBLE_CHECK=true; shift ;;
        --skip_download) SKIP_DOWNLOAD=true;  shift ;;
        --skip_convert)
            SKIP_DOWNLOAD=true
            SKIP_CONVERT=true
            shift ;;
        --skip_cot)
            SKIP_DOWNLOAD=true
            SKIP_CONVERT=true
            SKIP_COT=true
            shift ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Data Options:"
            echo "  --hf_repo REPO        HuggingFace repo ID (default: ServiceNow/VideoCUA)"
            echo "  --data_dir DIR         Local data directory (default: ./VideoCUA)"
            echo "  --output_dir DIR       Output for processed data (default: ./videocua_processed)"
            echo "  --num_workers N        Parallel workers for conversion (default: 4)"
            echo "  --platforms LIST       Comma-separated platform filter (default: all)"
            echo ""
            echo "CoT Generation Options:"
            echo "  --model MODEL          LLM model name (default: anthropic/claude-sonnet-4.5)"
            echo "  --suffix SUFFIX        Experiment suffix (default: cot_v1)"
            echo "  --num_threads N        Parallel LLM threads (default: 4)"
            echo "  --max_num N            Max tasks to process (default: all)"
            echo "  --base_url URL         Custom API endpoint (for vLLM, etc.)"
            echo "  --need_double_check    Enable double-check LLM pass"
            echo ""
            echo "Skip Flags:"
            echo "  --skip_download        Skip HF download + extraction"
            echo "  --skip_convert         Skip download + conversion (start from CoT)"
            echo "  --skip_cot             Skip everything (dry run / check config)"
            echo ""
            echo "API Key Environment Variables:"
            echo "  OPENROUTER_API_KEY     For OpenRouter models (provider/model format)"
            echo "  ANTHROPIC_API_KEY      For direct Anthropic API"
            echo "  OPENAI_API_KEY         For direct OpenAI API"
            echo "  API_KEY                Universal fallback"
            echo ""
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Derived paths
TASK_LIST="$OUTPUT_DIR/task_list.json"

# Sanitize model name for file paths (replace / with -)
MODEL_SAFE="${MODEL//\//-}"
MODEL_FOLDER="${MODEL_SAFE}_${SUFFIX}"

# -----------------------------------------------------------------------------
# CHECK API KEY (for CoT step)
# -----------------------------------------------------------------------------
if [ "$SKIP_COT" = false ]; then
    if [[ "$MODEL" == *"/"* ]]; then
        if [ -z "$OPENROUTER_API_KEY" ] && [ -z "$API_KEY" ]; then
            echo "WARNING: Using OpenRouter model but OPENROUTER_API_KEY is not set."
            echo "Please set: export OPENROUTER_API_KEY='your-key-here'"
            echo ""
        fi
    elif [[ "$MODEL" == *"claude"* ]]; then
        if [ -z "$ANTHROPIC_API_KEY" ] && [ -z "$API_KEY" ]; then
            echo "WARNING: Using Anthropic model but ANTHROPIC_API_KEY is not set."
            echo "Please set: export ANTHROPIC_API_KEY='your-key-here'"
            echo ""
        fi
    elif [[ "$MODEL" == *"gpt"* ]]; then
        if [ -z "$OPENAI_API_KEY" ] && [ -z "$API_KEY" ]; then
            echo "WARNING: Using OpenAI model but OPENAI_API_KEY is not set."
            echo "Please set: export OPENAI_API_KEY='your-key-here'"
            echo ""
        fi
    fi
fi

# -----------------------------------------------------------------------------
# DISPLAY CONFIGURATION
# -----------------------------------------------------------------------------
echo "=============================================="
echo "VideoCUA: Trajectory Synthesis Pipeline"
echo "=============================================="
echo ""
echo "Configuration:"
echo "  HF Repo:      $HF_REPO"
echo "  Data Dir:      $DATA_DIR"
echo "  Output Dir:    $OUTPUT_DIR"
echo "  Model:         $MODEL"
echo "  Model Folder:  $MODEL_FOLDER"
echo "  Suffix:        $SUFFIX"
echo "  Num Threads:   $NUM_THREADS"
[ -n "$MAX_NUM" ] && echo "  Max Tasks:     $MAX_NUM"
[ -n "$PLATFORMS" ] && echo "  Platforms:     $PLATFORMS"
[ -n "$BASE_URL" ] && echo "  Base URL:      $BASE_URL"
echo ""
echo "Steps to run:"
[ "$SKIP_DOWNLOAD" = false ] && echo "  [x] Step 1: Download & extract data"
[ "$SKIP_CONVERT" = false ] && echo "  [x] Step 2: Convert to trace format"
[ "$SKIP_COT" = false ] && echo "  [x] Step 3: Generate CoT annotations"
echo "=============================================="
echo ""

# -----------------------------------------------------------------------------
# STEP 1: DOWNLOAD AND EXTRACT DATA
# -----------------------------------------------------------------------------
if [ "$SKIP_DOWNLOAD" = false ]; then
    echo "[Step 1/3] Downloading and extracting data..."
    bash "$SCRIPT_DIR/download_data.sh" \
        --repo "$HF_REPO" \
        --output_dir "$DATA_DIR"

    if [ $? -ne 0 ]; then
        echo "Error: Download/extraction failed"
        exit 1
    fi
    echo ""
fi

# Verify data directory exists
ACTUAL_DATA_DIR="$DATA_DIR/data"
if [ ! -d "$ACTUAL_DATA_DIR" ]; then
    echo "Error: Data directory not found at $ACTUAL_DATA_DIR"
    echo "Make sure the dataset has been downloaded and extracted."
    exit 1
fi

# -----------------------------------------------------------------------------
# STEP 2: CONVERT DATA
# -----------------------------------------------------------------------------
if [ "$SKIP_CONVERT" = false ]; then
    echo "[Step 2/3] Converting VideoCUA data to trace format..."

    CONVERT_ARGS="--data_dir $ACTUAL_DATA_DIR --output_dir $OUTPUT_DIR --task_list_output $TASK_LIST --num_workers $NUM_WORKERS"
    [ -n "$PLATFORMS" ] && CONVERT_ARGS="$CONVERT_ARGS --platforms $PLATFORMS"

    python "$SCRIPT_DIR/convert_videocua.py" $CONVERT_ARGS

    if [ $? -ne 0 ]; then
        echo "Error: Conversion failed"
        exit 1
    fi
    echo ""
fi

# Verify task list exists
if [ ! -f "$TASK_LIST" ]; then
    echo "Error: Task list not found at $TASK_LIST"
    echo "Make sure conversion step completed successfully."
    exit 1
fi

# -----------------------------------------------------------------------------
# STEP 3: GENERATE COT
# -----------------------------------------------------------------------------
if [ "$SKIP_COT" = false ]; then
    echo "[Step 3/3] Generating CoT annotations..."

    COT_ARGS="--task_list_path $TASK_LIST --model $MODEL --num_threads $NUM_THREADS --suffix $SUFFIX"
    [ "$NEED_DOUBLE_CHECK" = true ] && COT_ARGS="$COT_ARGS --need_double_check"
    [ -n "$MAX_NUM" ] && COT_ARGS="$COT_ARGS --max_num $MAX_NUM"
    [ -n "$BASE_URL" ] && COT_ARGS="$COT_ARGS --base_url $BASE_URL"

    python "$SCRIPT_DIR/gen_cot.py" $COT_ARGS

    if [ $? -ne 0 ]; then
        echo "Error: CoT generation failed"
        exit 1
    fi

    # Generate CoT task list for downstream use
    echo "Generating CoT task list..."
    TASK_LIST_COT="$OUTPUT_DIR/task_list_cot.json"
    python "$SCRIPT_DIR/generate_task_list.py" \
        --data_dir "$OUTPUT_DIR" \
        --output "$TASK_LIST_COT" \
        --model_suffix "$MODEL_FOLDER"

    echo ""
fi

# -----------------------------------------------------------------------------
# DONE
# -----------------------------------------------------------------------------
echo "=============================================="
echo "Pipeline complete!"
echo "=============================================="
echo ""
echo "Output files:"
echo "  Raw data:           $DATA_DIR/data/"
echo "  Processed data:     $OUTPUT_DIR/"
echo "  Task list (raw):    $TASK_LIST"
[ "$SKIP_COT" = false ] && echo "  Task list (CoT):    $TASK_LIST_COT"
echo ""
