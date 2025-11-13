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
  <img src="./assets/groundcua-hq.png" width="100%" alt="GroundCUA Overview">
</div>

<div align="center">
  
### Authors

**Aarash Feizi<sup>1,2,4\*</sup>**, **Shravan Nayak<sup>1,3\*</sup>**, <br>
**Xiangru Jian<sup>5</sup>**, **Kevin Qinghong Lin<sup>6</sup>**, **Kaixin Li<sup>6</sup>**,
**Rabiul Awal<sup>1,3,4</sup>**, **Xing Han Lù<sup>1,2</sup>**, **Johan Obando-Ceron<sup>1,3</sup>**, **Juan A. Rodriguez<sup>1,8</sup>**,
**Nicolas Chapados<sup>4</sup>**, **David Vazquez<sup>4</sup>**, **Adriana Romero-Soriano<sup>1,2</sup>**, **Reihaneh Rabbany<sup>1,2</sup>**,<br>
**Perouz Taslakian<sup>4</sup>**, **Christopher Pal<sup>4</sup>**, **Spandana Gella<sup>4</sup>**, **Sai Rajeswar<sup>4,1,3</sup>**

<sup>1</sup>Mila - Quebec AI Institute, <sup>2</sup>McGill University, <sup>3</sup>Université de Montréal,<br>
<sup>4</sup>ServiceNow Research, <sup>5</sup>University of Waterloo, <sup>6</sup>National University of Singapore,<br>
<sup>7</sup>Polytechnique Montréal, <sup>8</sup>École de Technologie Supérieure, <sup>9</sup>CIFAR AI Chair

<sup>*</sup>Equal contribution

</div>


## Introduction

<div style="
  max-width: 880px;
  margin: 0 auto;
  text-align: justify;
  text-justify: inter-word;
  line-height: 1.6;">

Building reliable computer-use agents requires **grounding**: accurately connecting natural language instructions to the correct on-screen elements. While large datasets exist for web and mobile interactions, high-quality resources for desktop environments are limited. We address this gap through:
- **GroundCUA Dataset**: A large-scale, human-annotated desktop grounding dataset with **56K screenshots** from over 10,000 real-world human tasks across **87 applications** and **3.56M+ human-verified annotations**
- **GroundNext Models**: Vision-language models at **3B and 7B scales** achieving **state-of-the-art results** across five benchmarks
- **Efficient Training**: SOTA performance using **one-tenth the training data** of prior work

</div>

### Key Features

🎯 **High-Quality Desktop Dataset**
- Dense, expert-annotated screenshots with maximum annotation density
- Coverage of almost every visible element, including small icons and controls
- Fine-grained category information (menus, sidebars, etc.) for 50% of UI elements—**fully open-source!**

⚡ **Efficient Model Training**
- State-of-the-art performance with 700K datapoints vs 9M+ in prior work
- Two-stage training: supervised fine-tuning + reinforcement learning with fully open-source code
- Models at 3B and 7B scales for efficiency and accuracy

🌐 **Cross-Platform Generalization**
- Comprehensive evaluation on five challenging benchmarks
- Robust generalization across desktop, mobile, and web environments despite training only on desktop data


## 🚀 Quick Start

### Installation & Setup

To install from PyPI (recommended):

```bash
# Create and activate environment
python -m venv .venv
source .venv/bin/activate
# optional: update pip and install dependencies
pip install --upgrade pip

# Install PyTorch (adjust for CUDA version) and Flash Attention (for faster inference)
pip install torch torchvision
pip install flash-attn --no-build-isolation

# Install GroundCUA package for utilities (prepare_image, create_messages, etc.)
pip install groundcua

# optional: install all dependencies
pip install groundcua[all]
```

<details>
<summary>Alternative: Install from Source</summary>

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

# Install Flash Attention (recommended for faster inference)
pip install flash-attn --no-build-isolation

# Install in development mode
pip install -r requirements.txt

```
</details>


### Quick GroundNext Model Inference

```python
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from PIL import Image
import groundcua
import io
from urllib.request import urlopen

model_name = "ServiceNow/GroundNext-7B-V0"

# Load model and processor
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
    trust_remote_code=True
).eval()

processor = AutoProcessor.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Configure generation
model.generation_config.temperature = groundcua.DEFAULT_TEMPERATURE
model.generation_config.do_sample = False
model.generation_config.use_cache = True

# Load and prepare image
url = "https://huggingface.co/datasets/ServiceNow/GroundCUA/resolve/main/images/7-Zip/001f0079a489909eb94e47c2374b7bf36ab1842e314592ce30a34d18a54eb1df.png"
image = Image.open(io.BytesIO(urlopen(url).read()))
image, (width, height) = groundcua.prepare_image(image)

# Create messages and generate
instruction = "Click on the 'File' button"
messages = groundcua.create_messages(instruction, image, width, height)

input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
inputs = processor(text=[input_text], images=[image], videos=None, padding=True, return_tensors="pt").to(model.device)

generated_ids = model.generate(**inputs, max_new_tokens=groundcua.DEFAULT_MAX_NEW_TOKENS)
generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]

response = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(response)
# Expected output: <tool_call>{"name": "computer_use", "arguments": {"action": "left_click", "coordinate": [x, y]}}</tool_call>
```


## Updates

- **[Nov 11 2025]** 🎉 We released our [project webpage](https://groundcua.github.io), the [GroundCUA dataset](https://huggingface.co/datasets/ServiceNow/GroundCUA), and the [GroundNext-7B model](https://huggingface.co/ServiceNow/GroundNext-7B-V0)!

## Performance

### Desktop Grounding Benchmarks

<div align="center">

| **Model** | **ScreenSpot-Pro** | **OSWorld-G** | **UI-Vision** | **Avg** |
|-----------|:------------------:|:-------------:|:-------------:|:-----------------:|
| Qwen2.5-VL-7B | 29.7 | 42.7 | 16.5 | 29.6 |
| UI-TARS-72B | 38.1 | 57.1 | 25.5 | 40.2 |
| **GroundNext-3B** | **49.8** | **64.2** | **62.1** | **58.7** |
| **GroundNext-7B** | **52.9** | **67.7** | **60.3** | **60.3** |

</div>

### Cross-Platform Generalization

<div align="center">

| **Model** | **MMBench-GUI** | **ScreenSpot-v2** | **Avg** |
|-----------|:---------------:|:-----------------:|:--------------------:|
| Qwen2.5-VL-7B | 33.9 | 88.8 | 61.4 |
| UI-TARS-72B | 74.3 | 90.3 | 82.3 |
| **GroundNext-3B** | **77.1** | **88.5** | **82.8** |
| **GroundNext-7B** | **81.1** | **90.4** | **85.8** |

</div>

*Performance numbers demonstrate strong cross-domain (desktop, mobile and web) generalization despite training only on desktop data.*

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

## 🎓 Training

<div style="border-left: 6px solid #3b82f6; background: #eff6ff; padding: 12px 16px; margin: 16px 0;">
  <strong>🚧 Coming Soon:</strong> We are currently refining the training documentation and code. Complete training instructions, including supervised fine-tuning and reinforcement learning recipes, will be released in the <code>training/</code> folder soon. Stay tuned!
</div>

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
huggingface-cli download ServiceNow/GroundCUA --repo-type dataset --local-dir ./GroundCUA
```

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


## Project Structure

```
GroundCUA/
├── README.md                    # This file
├── pyproject.toml              # Package configuration
├── PUBLISHING.md               # Guide for publishing to PyPI
├── assets/                      # Images and resources
├── groundcua/                  # Main package (pip installable)
│   ├── __init__.py             # Package initialization and utilities
│   └── version.py              # Version information
├── eval/                        # Evaluation framework
│   ├── eval.py                 # Main evaluation script
│   ├── data.py                 # Data loading utilities
│   ├── prompts.py              # Prompt processing
│   └── models/                 # Model implementations
└── training/                   # Training pipeline (documentation coming soon)
```


## Acknowledgements

<p>
We thank the following projects and teams for their contributions to the open-source community:
</p>

- [InfiGUI-G1](https://github.com/InfiXAI/InfiGUI-G1) for the evaluation framework foundation
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for the excellent SFT training framework
- [verl](https://github.com/volcengine/verl) for the robust RL infrastructure
- [Qwen-2.5-VL](https://huggingface.co/collections/Qwen/qwen25-vl) for the foundation vision-language models
- [OpenCUA](https://github.com/xlang-ai/OpenCUA) for design inspiration of repository
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

