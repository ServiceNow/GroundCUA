<h1 style="
  font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;
  font-size:48px;
  font-weight:700;
  line-height:1.25;
  text-align:center;
  margin:0 0 24px;">
  GroundCUA: Grounding Computer Use Agents on Human Demonstrations
</h1>

<p align="center">
&nbsp&nbsp🌐 <a href="https://groundcua.github.io">Website</a>&nbsp&nbsp | &nbsp&nbsp📑 <a href="#">Paper</a>&nbsp&nbsp | &nbsp&nbsp🤗 <a href="https://huggingface.co/datasets/ServiceNow/GroundCUA">Dataset</a>&nbsp&nbsp | &nbsp&nbsp🤖 <a href="https://huggingface.co/ServiceNow/GroundNext-7B-V0">Models</a>&nbsp&nbsp
</p>

<div align="center">
  
### Authors

**Aarash Feizi<sup>1,2,4\*</sup>**, **Shravan Nayak<sup>1,3\*</sup>**, <br>
**Xiangru Jian<sup>5</sup>**, **Kevin Qinghong Lin<sup>6</sup>**, **Kaixin Li<sup>6</sup>**,
**Rabiul Awal<sup>1,3,4</sup>**, **Xing Han Lü<sup>1,2</sup>**, **Johan Obando-Ceron<sup>1,3</sup>**, **Juan A. Rodriguez<sup>1,8</sup>**,
**Nicolas Chapados<sup>4</sup>**, **David Vazquez<sup>4</sup>**, **Adriana Romero-Soriano<sup>1,2</sup>**, **Reihaneh Rabbany<sup>1,2</sup>**,<br>
**Perouz Taslakian<sup>4</sup>**, **Christopher Pal<sup>4</sup>**, **Spandana Gella<sup>4</sup>**, **Sai Rajeswar<sup>4,1,3</sup>**

<sup>1</sup>Mila - Quebec AI Institute, <sup>2</sup>McGill University, <sup>3</sup>Université de Montréal,<br>
<sup>4</sup>ServiceNow Research, <sup>5</sup>University of Waterloo, <sup>6</sup>National University of Singapore,<br>
<sup>7</sup>Polytechnique Montréal, <sup>8</sup>École de Technologie Supérieure, <sup>9</sup>CIFAR AI Chair

<sup>*</sup>Equal contribution

</div>


---

## Introduction

<div style="
  max-width: 880px;
  margin: 0 auto;
  text-align: justify;
  text-justify: inter-word;
  line-height: 1.6;">

Building reliable computer-use agents requires **grounding**: accurately connecting natural language instructions to the correct on-screen elements. While large datasets exist for web and mobile interactions, high-quality resources for desktop environments are limited.

**GroundCUA** addresses this gap through:
- **GroundCUA Dataset**: A large-scale, human-annotated desktop grounding dataset with **56K screenshots** across **87 applications** and **3.56M+ human-verified annotations**
- **GroundNext Models**: Vision-language models at **3B and 7B scales** achieving **state-of-the-art results** across five benchmarks
- **Efficient Training**: SOTA performance using **less than one-tenth the training data** (700K vs 9M) of prior work

</div>

### Key Features

🎯 **High-Quality Desktop Dataset**
- Dense, expert-annotated supervision with maximum annotation density
- Coverage of almost every visible element including small icons and controls
- Fine-grained category information for 50% of UI elements

⚡ **Efficient Model Training**
- State-of-the-art performance with 700K datapoints vs 9M in prior work
- Two-stage training: Supervised fine-tuning + Reinforcement learning
- Models at 3B and 7B scales for efficiency and accuracy

🌐 **Cross-Platform Generalization**
- Strong performance across desktop, mobile, and web environments
- Comprehensive evaluation on five challenging benchmarks
- Robust generalization despite training only on desktop data

---

## Performance

### Desktop Grounding Benchmarks

<div align="center">

| **Model** | **ScreenSpot-Pro** | **OSWorld-G** | **UI-Vision** | **Avg (Desktop)** |
|-----------|:------------------:|:-------------:|:-------------:|:-----------------:|
| Qwen2.5-VL-7B | 27.6 | 31.4 | 0.85 | - |
| UI-TARS-72B | 38.1 | 57.1 | 25.5 | - |
| **GroundNext-3B** | **45.2** | **52.8** | **27.1** | **41.7** |
| **GroundNext-7B** | **48.9** | **55.6** | **31.3** | **45.3** |

</div>

### Cross-Platform Generalization

<div align="center">

| **Model** | **MMBench-GUI** | **ScreenSpot-v2** | **Avg (Mobile/Web)** |
|-----------|:---------------:|:-----------------:|:--------------------:|
| Qwen2.5-VL-7B | 72.3 | 88.8 | 80.6 |
| UI-TARS-72B | 78.5 | 90.3 | 84.4 |
| **GroundNext-3B** | **81.2** | **91.5** | **86.4** |
| **GroundNext-7B** | **83.7** | **92.8** | **88.3** |

</div>

*Performance numbers demonstrate strong cross-domain generalization despite training only on desktop data.*

### Agentic Performance on OSWorld

GroundNext models also demonstrate strong agentic capabilities when integrated with reasoning models. When combined with OpenAI o3, **GroundNext-3B** achieves competitive performance on OSWorld, matching or exceeding much larger models.

<div align="center">

| **Model** | **OS** | **Office** | **Daily** | **Pro** | **Workflow** | **Overall** |
|------------|:------:|:----------:|:----------:|:--------:|:-------------:|:------------:|
| OpenAI o3 | 62.5 | 14.5 | 21.4 | 38.8 | 16.5 | 23.0 |
| CUA | 23.9 | 34.6 | 55.1 | 18.3 | 18.3 | 31.4 |
| OpenCUA-7B | 41.7 | 22.5 | 35.4 | 46.3 | 9.8 | 26.5 |
| OpenCUA-72B | 58.3 | 47.0 | 53.8 | 73.5 | 20.4 | 46.1 |
| UI-TARS-1.5-7B | 33.3 | 29.9 | 37.9 | 53.1 | 9.1 | 29.6 |
| JEDI-7B w/ o3 | *50.0* | 46.1 | **61.9** | **75.5** | *35.3* | **51.0** |
| **GroundNext-3B w/ o3 (ours)** | **62.5** | **47.0** | *55.0* | *73.5* | **36.5** | *50.6* |

</div>

*Task categories: OS (operating system tasks), Office (productivity applications), Daily (common user tasks), Pro (professional software), Workflow (multi-step workflows).*

### Key Results

- **Data Efficiency**: Achieves SOTA with only 700K training examples vs 9M+ in prior work
- **Cross-Domain Excellence**: Strong performance across desktop, mobile, and web despite desktop-only training
- **Fine-Grained Grounding**: Superior performance on small UI elements and complex workflows

---

## 🚀 Quick Start

### Installation & Setup

```bash
# Create and activate environment
conda create -n groundcua python=3.11.3 -y
conda activate groundcua

# Clone repository with submodules
git clone --recurse-submodules <repo_url>
cd GroundCUA

# Install dependencies
pip install -r requirements.txt

# Install LLaMA-Factory for SFT training
cd LLaMA-Factory/
pip install --no-deps -e .
cd ..

# Install verl for RL training
cd verl/
pip install -e .
cd ..
```

### Quick Model Inference

```python
import torch

from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from transformers.models.qwen2_vl.image_processing_qwen2_vl_fast import smart_resize

from PIL import Image

TEMP = 0.0
GroundNext_GROUNDER_SYS_PROMPT = """You are a helpful assistant.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{{"type": "function", "function": {{"name": "computer_use", "description": "Use a mouse and keyboard to interact with a computer, and take screenshots.\n* This is an interface to a desktop GUI. You do not have access to a terminal or applications menu. You must click on desktop icons to start applications.\n* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions. E.g. if you click on Firefox and a window doesn't open, try wait and taking another screenshot.\n* The screen's resolution is {width}x{height}.\n* Whenever you intend to move the cursor to click on an element like an icon, you should consult a screenshot to determine the coordinates of the element before moving the cursor.\n* If you tried clicking on a program or link but it failed to load, even after waiting, try adjusting your cursor position so that the tip of the cursor visually falls on the element that you want to click.\n* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.", "parameters": {{"properties": {{"action": {{"description": "The action to perform. The available actions are:\n* `key`: Performs key down presses on the arguments passed in order, then performs key releases in reverse order.\n* `type`: Type a string of text on the keyboard.\n* `mouse_move`: Move the cursor to a specified (x, y) pixel coordinate on the screen.\n* `left_click`: Click the left mouse button.\n* `left_click_drag`: Click and drag the cursor to a specified (x, y) pixel coordinate on the screen.\n* `right_click`: Click the right mouse button.\n* `middle_click`: Click the middle mouse button.\n* `double_click`: Double-click the left mouse button.\n* `scroll`: Performs a scroll of the mouse scroll wheel.\n* `wait`: Wait specified seconds for the change to happen.\n* `terminate`: Terminate the current task and report its completion status.", "enum": ["key", "type", "mouse_move", "left_click", "left_click_drag", "right_click", "middle_click", "double_click", "scroll", "wait", "terminate"], "type": "string"}}, "keys": {{"description": "Required only by `action=key`.", "type": "array"}}, "text": {{"description": "Required only by `action=type`.", "type": "string"}}, "coordinate": {{"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=mouse_move`, `action=left_click_drag`, `action=left_click`, `action=right_click`, `action=double_click`.", "type": "array"}}, "pixels": {{"description": "The amount of scrolling to perform. Positive values scroll up, negative values scroll down. Required only by `action=scroll`.", "type": "number"}}, "time": {{"description": "The seconds to wait. Required only by `action=wait`.", "type": "number"}}, "status": {{"description": "The status of the task. Required only by `action=terminate`.", "type": "string", "enum": ["success", "failure"]}}}}, "required": ["action"], "type": "object"}}}}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>"""

model_name = "ServiceNow/GroundNext-7B-V0"

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,       
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
            trust_remote_code=True
        ).eval()

processor = AutoProcessor.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


model.generation_config.temperature = TEMP
model.generation_config.do_sample = False if TEMP == 0.0 else True
model.generation_config.use_cache = True

image_path = "./screenshot.png"
instruction = "Click on the 'Save' icon"


# inference
image = Image.open(image_path).convert('RGB')
width, height = image.size
resized_height, resized_width = smart_resize(
    height,
    width,
    min_pixels=78_400,
    max_pixels=6_000_000,
)

image = image.resize((resized_width, resized_height))

img_width, img_height = resized_width, resized_height

full_prompt = f'{instruction}'

messages = [
    {
    "role": "system",
    "content": GroundNext_GROUNDER_SYS_PROMPT.format(img_width=img_width, img_height=img_height)
    },
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image,
            },
            {"type": "text", "text": full_prompt},
        ],
    }
]

input_text = tokenizer.apply_chat_template(messages,
                                           add_generation_prompt=True,
                                           tokenize=False)
inputs = processor(
                text=[input_text],
                images=[image],
                videos=None,
                padding=True,
                return_tensors="pt",
            ).to(model.device)

generated_ids = model.generate(**inputs, max_new_tokens=64)

generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
response = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0]

print(response)

```

---

## Dataset

### GroundCUA Dataset Overview


GroundCUA is a large-scale, human-annotated desktop grounding dataset with dense supervision:

- **📊 Scale**: 56K annotated screenshots, 3.56M element annotations
- **🎯 Density**: Maximum annotation density covering almost every visible UI element
- **✅ Quality**: Human-verified annotations from trained experts
- **🖥️ Coverage**: 87 desktop applications across 12 categories
- **📐 Resolution**: High-resolution images (500K to 7M pixels)
- **🏷️ Categories**: Fine-grained category information for 50% of elements

### Dataset Access

Download the GroundCUA dataset:

```bash
pip install -U huggingface_hub
huggingface-cli download xlangai/GroundCUA --repo-type dataset --local-dir ./GroundCUA
```

### Data Format

**SFT Data Format** (ShareGPT format, compatible with LLaMA-Factory):
```python
[
  {
    "conversations": [
      {
        "from": "human",
        "value": "<image>Press `Close`."
      },
      {
        "from": "function_call",
        "value": "{\"name\": \"computer_use\", \"arguments\": {\"action\": \"mouse_move\", \"coordinate\": [999, 19]}}"
      }
    ],
    "system": "You are a helpful assistant.",
    "images": ["./GroundCUA/screenshot.png"],
    "tool": "[{\"name\": \"computer_use\", \"description\": \"...\"}]"
  }
]
```

**RL Data Format** (compatible with verl):
```python
{
  "system": "You are a helpful assistant...",
  "instruction": "Click on 'All Deleted Documents'...",
  "images": ["./GroundCUA/screenshot.png"],
  "gt_response": "{\"name\": \"computer_use\", ...}",
  "gt_bbox": [181, 456, 20, 432]
}
```

---

## Training

### Supervised Fine-tuning (SFT)

<div style="border-left: 6px solid #f28c28; background: #fff8e6; padding: 12px 16px; margin: 16px 0;">
  <strong>📝 Note:</strong> We use <a href="https://github.com/hiyouga/LLaMA-Factory">LLaMA-Factory</a> for initial supervised fine-tuning on human demonstrations.
</div>

Training configurations are in `sft/config/sft/` directory.

**Example Configuration** (`sft/config/sft/groundnext-3b.json`):

```json
{
    "stage": "sft",
    "do_train": true,
    "model_name_or_path": "Qwen/Qwen2.5-VL-3B-Instruct",
    "dataset": "groundcua-sft",
    "template": "qwen2_vl",
    "output_dir": "/home/GroundCUA/sft/checkpoints/GroundNext-sft-3b",
    "learning_rate": 3e-6,
    "num_train_epochs": 1.0,
    "bf16": true,
    "flash_attn": "fa2"
}
```

**Run SFT Training**:

```bash
cd LLaMA-Factory/

# Train 3B model
python main.py train ../sft/config/sft/groundnext-3b.json

# Train 7B model
python main.py train ../sft/config/sft/groundnext-7b.json
```

**Checkpoints**: Models saved to `sft/checkpoints/` directory.

### Reinforcement Learning (RLOO)

<div style="border-left: 6px solid #f28c28; background: #fff8e6; padding: 12px 16px; margin: 16px 0;">
  <strong>🔧 Note:</strong> We use <a href="https://github.com/volcengine/verl">verl</a> framework for reinforcement learning with RLOO algorithm.
</div>

Training scripts are in `rl/recipe/groundnext/` directory.

**Run RL Training**:

```bash
# For 3B model
./rl/recipe/groundnext/groundnext-3b.sh

# For 7B model  
./rl/recipe/groundnext/groundnext-7b.sh
```

**Key RL Parameters**:
- **Algorithm**: RLOO (Reward Learning with Likelihood Optimization)
- **Reward Function**: Custom GUI reward function (`reward_clipped.py`)
- **Base Model**: SFT checkpoint from previous stage
- **Batch Size**: 64 for training, 8 for rollout
- **Learning Rate**: 1e-6

**Checkpoints**: Models saved to `rl/checkpoints/` directory.

---

## 📊 Evaluation

<div style="border-left: 6px solid #9ca3af; background: #f5f5f5; padding: 12px 16px; margin: 16px 0;">
  <em>Our evaluation framework builds upon <a href="https://github.com/InfiXAI/InfiGUI-G1/tree/main/eval">InfiGUI-G1</a> and provides comprehensive evaluation across multiple benchmarks.</em>
</div>

### Supported Benchmarks

- **ScreenSpot-Pro**: Desktop element grounding
- **ScreenSpot-v2**: Web and mobile interface grounding
- **MMBench-GUI**: GUI understanding tasks
- **OSWorld-G**: Operating system grounding
- **UI-Vision**: Diverse desktop application grounding

### Running Evaluations

```bash
cd eval/

# Evaluate on specific benchmark
python eval.py \
    --model_type qwen25vl \
    --model_name_or_path /path/to/trained/model \
    --benchmark screenspot \
    --data_path /path/to/benchmark/data \
    --output_dir results/

# Evaluate on all benchmarks
python eval.py \
    --model_type qwen25vl \
    --model_name_or_path /path/to/trained/model \
    --benchmark all \
    --task all \
    --language en
```

### Evaluation Metrics

- **Accuracy**: Precision of GUI element localization
- **Success Rate**: Percentage of correctly grounded elements
- **Cross-Domain Performance**: Generalization to unseen platforms
- **Fine-Grained Performance**: Accuracy on small UI elements

---

## Project Structure

```
GroundCUA/
├── README.md                    # This file
├── assets/                      # Images and resources
├── eval/                        # Evaluation framework
│   ├── eval.py                 # Main evaluation script
│   ├── data.py                 # Data loading utilities
│   ├── prompts.py              # Prompt processing
│   └── models/                 # Model implementations
├── sft/                        # Supervised Fine-tuning
│   ├── config/                 # Training configurations
│   │   ├── sft/               # SFT-specific configs
│   │   └── deepspeed/         # DeepSpeed configurations
│   └── checkpoints/           # SFT model checkpoints
├── LLaMA-Factory/              # LLaMA-Factory framework
│   └── data/                  # SFT training data
├── rl/                         # Reinforcement Learning
│   ├── recipe/                # Training recipes
│   │   └── groundnext/       # GroundNext-specific scripts
│   ├── data/                  # RL training data
│   └── checkpoints/           # RL model checkpoints
└── verl/                       # verl framework
```

---

## Advanced Usage

### Custom Data Preparation

1. **Format demonstration data** according to the schemas above
2. **Place data files** in appropriate directories:
   - SFT: `LLaMA-Factory/data/`
   - RL: `rl/data/`
3. **Update dataset configurations** in config files
4. **Run training** with custom configurations

### Model Customization

- **Architecture**: Modify model configurations in training scripts
- **Hyperparameters**: Adjust learning rates, batch sizes, epochs
- **Reward Functions**: Implement custom reward functions for RL
- **Evaluation**: Add custom benchmarks in `eval/`

---

## Acknowledgements

<p>
We thank the following projects and teams for their contributions to the open-source community:
</p>

- [InfiGUI-G1](https://github.com/InfiXAI/InfiGUI-G1) for the evaluation framework foundation
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for the excellent SFT training framework
- [verl](https://github.com/volcengine/verl) for the robust RL infrastructure
- [Qwen-2.5-VL](https://huggingface.co/collections/Qwen/qwen25-vl) for the foundation vision-language models
- The computer use and GUI automation research community

---

## Research Use and Disclaimer

GroundCUA is intended for **research and educational purposes only**.

### Prohibited Uses
- The model, dataset, and code may **not** be used for any purpose that violates applicable laws or regulations
- Use for illegal, unethical, or harmful activities is strictly prohibited

### Disclaimer
- The authors and contributors are **not responsible** for any illegal, unethical, or harmful use
- Users are solely responsible for ensuring compliance with applicable laws and regulations

---

## Citation

If you use GroundCUA in your research, please cite our work:

```bibtex
@article{groundcua2025,
  title={Grounding Computer Use Agents on Human Demonstrations},
  author={Aarash Feizi and Shravan Nayak and Xiangru Jian and Kevin Qinghong Lin and Kaixin Li and Rabiul Awal and Xing Han Lü and Johan Obando-Ceron and Juan A. Rodriguez and Nicolas Chapados and David Vazquez and Adriana Romero-Soriano and Reihaneh Rabbany and Perouz Taslakian and Christopher Pal and Spandana Gella and Sai Rajeswar},
  journal={arXiv preprint arXiv:2501.XXXXX},
  year={2025}
}

