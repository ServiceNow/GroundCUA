<div align="center">
  <h1 style="
    font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;
    font-size:48px;
    font-weight:700;
    line-height:1.2;
    margin:0 0 24px;
  ">
    <span style="display:inline-flex; align-items:baseline;">
      <img
        src="./assets/logo.png"
        alt="GroundCUA Logo"
        style="
          height:1.5em;
          width:1.5em;
          vertical-align:middle;
          margin-right:0.25em;
          position:relative;
          top:0.05em;"
      />
      GroundCUA: Grounding Computer Use Agents on Human Demonstrations
    </span>
  </h1>
</div>




<p align="center">
&nbsp&nbsp🌐 <a href="https://groundcua.github.io">Website</a>&nbsp&nbsp | &nbsp&nbsp📑 <a href="https://arxiv.org/abs/2511.07332">Paper</a>&nbsp&nbsp | &nbsp&nbsp🤗 <a href="https://huggingface.co/datasets/ServiceNow/GroundCUA">Dataset</a>&nbsp&nbsp | &nbsp&nbsp🤖 <a href="https://huggingface.co/ServiceNow/GroundNext-7B-V0">Models</a>&nbsp&nbsp
</p>

<div align="center">
  <img src="./assets/groundcua-hq.png" width="700" alt="GroundCUA Overview">
</div>

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
conda create -n groundcua python=3.10 -y
conda activate groundcua

pip install --upgrade pip

# Clone repository
git clone https://github.com/ServiceNow/GroundCUA.git
cd GroundCUA

# Install PyTorch (adjust for your CUDA version)
pip install torch torchvision

# Install core dependencies
pip install -r requirements.txt

# Install Flash Attention (recommended for faster inference)
pip install flash-attn --no-build-isolation
```

### Optional: Install Training Frameworks

<div style="border-left: 6px solid #f28c28; background: #fff8e6; padding: 12px 16px; margin: 16px 0;">
  <strong>📝 Note:</strong> Training frameworks are only needed if you plan to train models. For inference only, skip this section.
</div>

```bash
# Initialize submodules if not already done
git submodule update --init --recursive

# Install LLaMA-Factory
cd LLaMA-Factory/
pip install -e ".[torch,metrics]" --no-build-isolation
cd ..

# Install verl
cd verl/
pip install -e .

# Install the latest stable version of vLLM
pip install vllm==0.8.3

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
@misc{feizi2025groundingcomputeruseagents,
      title={Grounding Computer Use Agents on Human Demonstrations}, 
      author={Aarash Feizi and Shravan Nayak and Xiangru Jian and Kevin Qinghong Lin and Kaixin Li and Rabiul Awal and Xing Han Lù and Johan Obando-Ceron and Juan A. Rodriguez and Nicolas Chapados and David Vazquez and Adriana Romero-Soriano and Reihaneh Rabbany and Perouz Taslakian and Christopher Pal and Spandana Gella and Sai Rajeswar},
      year={2025},
      eprint={2511.07332},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2511.07332}, 
}

