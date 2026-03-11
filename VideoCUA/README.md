# VideoCUA: Trajectory Synthesis Pipeline

End-to-end pipeline for synthesizing Chain-of-Thought (CoT) trajectories from the [VideoCUA](https://huggingface.co/datasets/AgentsResearch/ActCUA) dataset. Converts raw human demonstration videos and action logs into rich, LLM-annotated trajectories with observation, thought, action, and reflection for each step.

## Pipeline Overview

```
HuggingFace (VideoCUA)          download_data.sh
    |
    v
raw_data/*.zip                 (87 platform ZIPs)
    |
    v
data/{Platform}/{task_id}/     (extracted: action_log.json + video/)
    |                          convert_videocua.py
    v
output/{Platform}/{task_id}/   (opencua_trace.jsonl + processed_images/)
    |                          gen_cot.py
    v
output/{Platform}/{task_id}/   (CoT-annotated trajectories with
  {model}_{suffix}/              observation/thought/action/reflection)
    cot_tasks/
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Your API Key

Pick one based on your model provider:

```bash
# OpenRouter (recommended - supports Claude, GPT-4o, etc.)
export OPENROUTER_API_KEY="your-key-here"

# Or direct Anthropic API
export ANTHROPIC_API_KEY="your-key-here"

# Or direct OpenAI API
export OPENAI_API_KEY="your-key-here"
```

### 3. Run the Full Pipeline

```bash
bash run_pipeline.sh
```

This will download the VideoCUA dataset, convert all tasks, and generate CoT annotations. See [Configuration](#configuration) for customization.

## Step-by-Step Usage

If you prefer to run each step manually:

### Step 1: Download Data

```bash
bash download_data.sh --repo AgentsResearch/ActCUA --output_dir ./VideoCUA
```

This downloads the dataset and extracts platform ZIP files from `raw_data/` into `data/`.

**After this step:**
```
VideoCUA/
  data/
    Blender/
      46551/
        action_log.json
        video/video.mp4
      46552/
        ...
    Firefox/
      ...
```

### Step 2: Convert to Trace Format

```bash
python convert_videocua.py \
    --data_dir ./VideoCUA/data \
    --output_dir ./videocua_processed \
    --num_workers 4
```

Options:
- `--platforms "Blender,GIMP"` - Process specific platforms only
- `--task_list_output path/to/task_list.json` - Custom task list output path
- `--num_workers 8` - More parallel workers for faster frame extraction

**After this step:**
```
videocua_processed/
  task_list.json               # List of all converted tasks
  Blender/
    46551/
      opencua_trace.jsonl       # Standardized trajectory
      processed_images/         # Video frames at each action timestamp
        0.png
        1.png
        ...
      action_log.json           # Copy of original
```

### Step 3: Generate CoT Annotations

```bash
python gen_cot.py \
    --task_list_path ./videocua_processed/task_list.json \
    --model anthropic/claude-sonnet-4.5 \
    --num_threads 4 \
    --suffix cot_v1
```

Options:
- `--model` - See [Supported Models](#supported-models) below
- `--base_url` - Custom API endpoint (for vLLM or other OpenAI-compatible servers)
- `--num_threads` - Number of parallel LLM calls
- `--max_num 10` - Process only the first N tasks (useful for testing)
- `--need_double_check` - Extra LLM pass to refine annotations
- `--no_auto_merge` - Skip automatic JSONL merge after processing

**After this step:**
```
videocua_processed/
  Blender/
    46551/
      anthropic-claude-sonnet-4.5_cot_v1/
        cot_tasks/
          46551/
            meta.json           # Task metadata + trajectory evaluation
            000.json            # Step 0: observation, thought, action, reflection
            001.json            # Step 1
            ...
        opencua_trace_with_cot.jsonl  # Merged: all steps in one file
```

### Step 4: Generate CoT Task List (Optional)

Create a task list pointing to CoT trajectory files for downstream use:

```bash
python generate_task_list.py \
    --data_dir ./videocua_processed \
    --output ./videocua_processed/task_list_cot.json \
    --model_suffix anthropic-claude-sonnet-4.5_cot_v1
```

## Configuration

### `run_pipeline.sh` Options

| Flag | Default | Description |
|------|---------|-------------|
| `--hf_repo` | `AgentsResearch/ActCUA` | HuggingFace dataset repo ID |
| `--data_dir` | `./VideoCUA` | Local directory for raw data |
| `--output_dir` | `./videocua_processed` | Output directory for processed data |
| `--model` | `anthropic/claude-sonnet-4.5` | LLM model for CoT generation |
| `--suffix` | `cot_v1` | Experiment suffix for output folder |
| `--num_threads` | `4` | Parallel threads for LLM calls |
| `--num_workers` | `4` | Parallel workers for video conversion |
| `--max_num` | all | Max tasks to process |
| `--platforms` | all | Comma-separated platform filter |
| `--base_url` | auto | Custom API endpoint |
| `--need_double_check` | off | Enable LLM double-check |
| `--skip_download` | - | Skip download + extraction |
| `--skip_convert` | - | Skip download + conversion |
| `--skip_cot` | - | Skip all steps (config check) |

### Supported Models

| Format | Example | API Key |
|--------|---------|---------|
| OpenRouter | `anthropic/claude-sonnet-4.5` | `OPENROUTER_API_KEY` |
| OpenRouter | `openai/gpt-4o` | `OPENROUTER_API_KEY` |
| Direct Anthropic | `claude-sonnet-4-5-20250929` | `ANTHROPIC_API_KEY` |
| Direct OpenAI | `gpt-4o` | `OPENAI_API_KEY` |
| Qwen (DashScope) | `qwen-vl-max` | `DASHSCOPE_API_KEY` |
| vLLM (local) | `YourModel` + `--base_url http://localhost:8000/v1` | `API_KEY` |

For all providers, `API_KEY` can be used as a universal fallback.

## Data Formats

### Input: `action_log.json`

```json
{
    "task_id": 46551,
    "task_instruction": "Switch to Wireframe mode using the Wireframe button.",
    "platform": "Blender",
    "action_log": [
        {
            "action_type": "CLICK",
            "timestamp": 1.2,
            "action_params": {"x": 1748, "y": 95, "text": "Left", "numClicks": 1}
        },
        {
            "action_type": "AFTER_LAST_ACTION",
            "timestamp": 3.362
        }
    ]
}
```

### Intermediate: `opencua_trace.jsonl`

```json
{
    "task_id": "46551",
    "instruction": "On this Blender platform, please Switch to Wireframe mode...",
    "traj": [
        {"index": 0, "image": "/path/to/processed_images/0.png", "value": {"code": "pyautogui.click(x=0.9109, y=0.0880)"}},
        {"index": 1, "image": "/path/to/processed_images/1.png", "value": {"code": "computer.terminate(status='success')"}}
    ]
}
```

### Output: CoT Trajectory (per-step JSON)

```json
{
    "index": 0,
    "image": "/path/to/processed_images/0.png",
    "value": {
        "code": "pyautogui.click(x=0.9109, y=0.0880)",
        "observation": "The Blender 4.3.0 interface is open with...",
        "thought": "I need to switch to Wireframe mode. Looking at the header bar...",
        "action": "Click the Wireframe display mode button in the header bar",
        "last_step_correct": true,
        "last_step_redundant": false,
        "reflection": "The viewport changed from Solid to Wireframe mode..."
    }
}
```

### Output: Trajectory Evaluation (`meta.json`)

After all steps are processed, an LLM evaluates the full trajectory:

```json
{
    "task_id": "46551",
    "instruction": "...",
    "task_completed": true,
    "alignment_score": 9,
    "efficiency_score": 8,
    "task_difficulty": 3,
    "reason": "Task completed efficiently with one click.",
    "actual_task": "Switch to Wireframe display mode in Blender.",
    "natural_language_task": "Switch the 3D viewport to Wireframe mode."
}
```

## Supported Action Types

| Action Type | pyautogui Code |
|-------------|---------------|
| CLICK | `pyautogui.click(x, y)` |
| Double Click | `pyautogui.doubleClick(x, y)` |
| Right Click | `pyautogui.rightClick(x, y)` |
| MOVE_TO | `pyautogui.moveTo(x, y)` |
| DRAG_TO | `pyautogui.dragTo(x, y, button)` |
| SCROLL | `pyautogui.scroll(amount)` |
| TYPE/TYPING | `pyautogui.typewrite(text)` |
| PRESS | `pyautogui.press(key)` |
| HOTKEY | `pyautogui.hotkey(key1, key2)` |
| KEY_DOWN/UP | `pyautogui.keyDown(key)` / `keyUp(key)` |
| MOUSE_DOWN/UP | `pyautogui.mouseDown(button)` / `mouseUp(button)` |
| TERMINATE | `computer.terminate(status='success')` |

All coordinates are normalized to `[0, 1]` range.

## Resume Capability

The pipeline supports resuming from where it left off:

- **gen_cot.py** skips steps that already have output JSON files (e.g., `000.json`, `001.json`)
- **gen_cot.py** skips tasks that are already fully processed
- **download_data.sh** skips extraction if `data/` already exists
- **batch_merge.py** skips tasks that already have merged JSONL files

To re-process, delete the corresponding output files.

## File Structure

```
VideoCUA/
  README.md                        # This file
  requirements.txt                 # Python dependencies
  download_data.sh                 # Download from HuggingFace + extract ZIPs
  convert_videocua.py                # Convert raw data to opencua_trace format
  gen_cot.py                       # Generate CoT annotations via LLM
  merge_json.py                    # Merge per-step JSONs into JSONL
  batch_merge.py                   # Batch merge across tasks/platforms
  generate_task_list.py            # Generate task list JSON for downstream use
  utils.py                         # Shared utilities (image, LLM, coordinates)
  run_pipeline.sh                  # End-to-end pipeline script
  module/
    __init__.py
    generator.py                   # CoT generation prompts + parsing
    evaluator.py                   # Trajectory evaluation prompts
    reflector.py                   # Step correctness reflection
    reflector_with_prior_judge.py  # Reflection with prior judge feedback
```

## Acknowledgement

This codebase includes components adapted from the [OpenCUA project](https://github.com/xlang-ai/OpenCUA).
