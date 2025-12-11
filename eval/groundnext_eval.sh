#!/bin/bash

# Sample evaluation script for GroundNext model
# Usage: ./eval.sh <model_path> [benchmark]
# Example: ./eval.sh /path/to/model screenspot-pro

# Default settings
DEFAULT_BENCHMARK="screenspot-pro"
DEFAULT_PROMPT="groundcua"
DEFAULT_ENGINE="vllm"
DEFAULT_TP=4
DEFAULT_BATCH_SIZE=16

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Get model path from argument
if [ -z "$1" ]; then
    echo "Usage: $0 <model_path> [benchmark]"
    echo ""
    echo "Arguments:"
    echo "  model_path  Path to model (required)"
    echo "  benchmark   Benchmark name (default: $DEFAULT_BENCHMARK)"
    echo ""
    echo "Example:"
    echo "  $0 /path/to/model screenspot-pro"
    exit 1
fi

MODEL_PATH=$1
BENCHMARK=${2:-$DEFAULT_BENCHMARK}
MODEL_NAME=$(basename "$MODEL_PATH")

# Check if model exists
if [ ! -d "$MODEL_PATH" ] && [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model not found at $MODEL_PATH"
    exit 1
fi

echo "=========================================="
echo "Starting Evaluation"
echo "=========================================="
echo "Model: $MODEL_NAME"
echo "Model Path: $MODEL_PATH"
echo "Benchmark: $BENCHMARK"
echo "Prompt: $DEFAULT_PROMPT"
echo "Engine: $DEFAULT_ENGINE"
echo "=========================================="
echo ""

# Change to script directory
cd "$SCRIPT_DIR" || exit 1

# Run evaluation
python eval.py \
    "$MODEL_PATH" \
    --benchmark "$BENCHMARK" \
    --prompt "$DEFAULT_PROMPT" \
    --engine "$DEFAULT_ENGINE" \
    --tensor-parallel "$DEFAULT_TP" \
    --batch-size "$DEFAULT_BATCH_SIZE" \
    --temperature 0.0 \
    --no-cache

echo ""
echo "=========================================="
echo "Evaluation Complete!"
echo "Results saved to: ./output/$MODEL_NAME/$BENCHMARK/"
echo "=========================================="
