#!/bin/bash

BASE_DIR="/home/GroundCUA/rl/checkpoints/groundnext"
SCRIPT="/home/envs/grpo-gui/bin/python3.11 model_merger.py"

STEPS=(2000)

TARGET_DIRS=(
    groundnext-3b
    groundnext-7b
)

for step in "${STEPS[@]}"; do
    
    for dir_name in "${TARGET_DIRS[@]}"; do
        full_dir="${BASE_DIR}/${dir_name}"
        ACTOR_DIR="${full_dir}/global_step_${step}/actor"
        
        if [ ! -d "$ACTOR_DIR" ]; then
            echo "Skipping: $ACTOR_DIR (not found)"
            continue
        fi
        
        # Check for .safetensors files
        if ls "$ACTOR_DIR"/huggingface/*.safetensors 1> /dev/null 2>&1; then
            echo "Skipping: $ACTOR_DIR (.safetensors already exists)"
            continue
        fi
        
        echo "Running for: $ACTOR_DIR"
        $SCRIPT --local_dir "$ACTOR_DIR"
        
    done
done
