# GroundNext Training

This directory contains the training code and configurations for GroundNext models.

## Overview

GroundNext models are trained using a two-stage approach:

### 1. Supervised Fine-tuning (SFT)
- **Framework**: [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
- **Purpose**: Initial fine-tuning on human-annotated demonstrations
- **Location**: `sft/` directory
- **Base Models**: Qwen2.5-VL-3B-Instruct and Qwen2.5-VL-7B-Instruct
- **Data Format**: ShareGPT format with vision support

### 2. Reinforcement Learning (RL)
- **Framework**: [verl](https://github.com/volcengine/verl)
- **Algorithm**: RLOO (Reward Learning with Likelihood Optimization)
- **Purpose**: Refine model performance using custom GUI reward functions
- **Location**: `rl/` directory
- **Base Models**: SFT checkpoints from stage 1

## Quick Start

We have provided configuration files in both `sft/config/` and `rl/recipe/` directories for quick experimentation:

- **SFT Configs**: `sft/config/sft/groundnext-{3b,7b}.json`
- **RL Scripts**: `rl/recipe/groundnext/groundnext-{3b,7b}.sh`
- **DeepSpeed Configs**: `sft/config/deepspeed/` for distributed training

## Documentation

🚧 **Detailed training guidelines, including:**
- Complete setup instructions
- Data preparation steps
- Hyperparameter explanations
- Troubleshooting tips
- Best practices

**Will be provided soon!**