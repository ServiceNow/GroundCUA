# Grounding Computer Use Agents on Human Demonstrations

<div align="center">

<a href="#" target="_blank">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-2501.XXXXX-red" height="35" />
</a>
<a href="https://uivision.github.io/suite.html" target="_blank">
    <img alt="Website" src="https://img.shields.io/badge/%F0%9F%8C%8E%20Website-UI Vision Suite-blue" height="35" />
</a>

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

<img src="./assets/groundcua-hq.png"/>

---

## Introduction

Building reliable computer-use agents requires **grounding**: accurately connecting natural language instructions to the correct on-screen elements. While large datasets exist for web and mobile interactions, high-quality resources for desktop environments are limited. 

To address this gap, we introduce **GroundCUA**, a large-scale desktop grounding dataset built from expert human demonstrations, and **GroundNext**, a family of vision-language models designed for precise grounding across desktop applications.

* **GroundCUA Dataset**: A large-scale, human-annotated dataset covering **87 applications** across **12 categories** with **56K screenshots** and **3.56M+ human-verified annotations**. The dataset provides dense, high-resolution supervision with fine-grained category information and includes multiple variants of related applications.

* **GroundNext Models**: Vision-language models at **3B and 7B scales** that achieve **state-of-the-art results** across five benchmarks using supervised fine-tuning, while requiring **less than one-tenth the training data** of prior work.

* **Two-Stage Training**: Our approach combines supervised fine-tuning (SFT) on 700K curated datapoints from GroundCUA, followed by reinforcement learning (RL) to further refine performance without complex reward strategies.

* **Cross-Platform Generalization**: Despite training only on desktop data, GroundNext excels across desktop, mobile, and web environments, demonstrating robust generalization capabilities.

### Key Contributions

1. **High-Quality Desktop Dataset**: GroundCUA provides dense, expert-annotated supervision with maximum annotation density, covering almost every visible element including small icons and controls across diverse desktop applications.

2. **Efficient Model Training**: GroundNext achieves state-of-the-art performance with significantly fewer datapoints than prior models (700K vs 9M), demonstrating that high-quality, well-curated data outperforms larger, less precise datasets.

3. **Comprehensive Evaluation**: Strong performance across desktop benchmarks (ScreenSpotPro, OSWorld-G, UI-Vision) and cross-domain generalization to mobile and web platforms (MMBench-GUI, ScreenSpot-v2).

### GroundCUA Dataset

GroundCUA represents a significant advancement in desktop grounding datasets:

- **Scale**: 56K annotated screenshots and 3.56 million element annotations
- **Resolution & Density**: High-resolution images (500K to 7M pixels) with maximum annotation density, covering almost every visible element
- **Expert Quality**: Human-verified annotations from trained annotators for high accuracy
- **Application Diversity**: 87 desktop applications across 12 categories for broad real-world coverage
- **Fine-grained Categories**: 50% of UI elements include detailed category information (menus, buttons, etc.)
- **Application Variants**: Multiple variants of related applications (e.g., LibreOffice and OnlyOffice) for robust learning

### GroundNext Models

From GroundCUA, we construct a 700K image-instruction pair dataset that mimics real-world semantic interactions. The GroundNext series includes:

1. **Two-Stage Training**: First, supervised fine-tuning (SFT) on curated GroundCUA data, then reinforcement learning for performance refinement
2. **Efficient Architecture**: Models at 3B and 7B scales offering balance between efficiency and accuracy  
3. **State-of-the-art Performance**: Outperforms existing models while using significantly fewer training datapoints

---

## Results

We evaluate GroundNext against state-of-the-art computer use agents across multiple benchmarks. GroundNext achieves state-of-the-art results on key desktop benchmarks and demonstrates strong cross-platform generalization.

### Benchmark Performance

**Desktop Grounding Benchmarks:**
- **[ScreenSpotPro](https://github.com/likaixin2000/ScreenSpot-Pro-GUI-Grounding)**: State-of-the-art performance on desktop element grounding
- **[OSWorld-G](https://arxiv.org/abs/2505.13227)**: Leading results on operating system grounding tasks  
- **[UI-Vision](https://arxiv.org/abs/2503.15661)**: Top performance across diverse desktop applications

**Cross-Platform Generalization:**
- **[MMBench-GUI](https://arxiv.org/abs/2507.19478)**: Strong performance on GUI understanding tasks
- **[ScreenSpot-v2](https://arxiv.org/abs/2410.23218)**: Excellent generalization to web and mobile interfaces

### Key Results:

* **Data Efficiency**: GroundNext-3B and 7B achieve state-of-the-art performance using only 700K training examples, compared to 9M+ datapoints used by prior models like Jedi
* **Cross-Domain Generalization**: Despite training only on desktop data, models excel across desktop, mobile, and web environments  
* **Fine-Grained Grounding**: Superior performance on small UI elements and complex multi-window desktop workflows
* **Training Efficiency**: High-quality expert demonstrations enable faster convergence and better final performance than large-scale synthetic data

---

## Setup

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

---

## Dataset

### GroundCUA Dataset Overview

GroundCUA is a large-scale, human-annotated desktop grounding dataset that provides dense supervision for training computer-use agents:

- **56K Screenshots**: High-resolution desktop screenshots across diverse applications
- **3.56M+ Annotations**: Human-verified element annotations with fine-grained categories  
- **87 Applications**: Spanning 12 categories including productivity, development, media, and utilities
- **Dense Labeling**: Maximum annotation density covering almost every visible UI element
- **Resolution Diversity**: Images ranging from 500K to 7M pixels reflecting real desktop environments
- **Expert Quality**: Annotations from trained human annotators ensuring high accuracy

The dataset includes fine-grained category information (menus, buttons, text fields, etc.) for 50% of UI elements and covers multiple variants of related applications to enable robust, application-specific grounding strategies.

## 🛠️ Data Generation

We provide a complete data pipeline for creating human-grounded datasets compatible with our training frameworks.

### 1. Data Format

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
    "images": [
      "./GroundCUA/96513__Mjs9hv1Dm5OXYjDAAiXz_96513_before_action_8_1735102393068.png"
    ],
    "tool": "[{\"name\": \"computer_use\", \"description\": \"Use a mouse and keyboard to interact with a computer, and take screenshots...\",...}]"
  }
]
```

**RL Data Format** (compatible with verl):
```python
{
  "system": "You are a helpful assistant.\n\n# Tools\n\nYou may call one or more functions to assist with the user query...",
  "instruction": "Click on 'All Deleted Documents' under the Trash section to view the list of deleted documents.",
  "images": [
    "./GroundCUA/54817_QLPUtLiY6J_R0CliD5rWS_54817_before_action_15.png"
  ],
  "gt_response": "{\"name\": \"computer_use\", \"arguments\": {\"action\": \"mouse_move\", \"coordinate\": [101, 444]}}",
  "gt_bbox": [181, 456, 20, 432]
}
```

### 2. Data Preparation

Place your training data in the appropriate directories:
- SFT data: `LLaMA-Factory/data/`
- RL data: `rl/data/`

---

## Training

### Supervised Fine-tuning (SFT)

We use [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for initial supervised fine-tuning on human demonstrations.

Training configurations are located in `sft/config/sft/` directory.

Example SFT configuration (`sft/config/sft/groundnext-3b.json`):

```json
{
    "stage": "sft",
    "do_train": true,
    "model_name_or_path": "Qwen/Qwen2.5-VL-3B-Instruct",
    "dataset": "groundcua-sft",
    "dataset_dir": "/path/to/LLaMA-Factory/data/",
    "template": "qwen2_vl",
    "output_dir": "/path/to/sft/outputs/GroundNext-sft-3b",
    "learning_rate": 3e-6,
    "num_train_epochs": 1.0,
    "bf16": true,
    "flash_attn": "fa2"
}
```

**Run SFT Training**:

```bash
cd LLaMA-Factory/
python main.py train ../sft/config/sft/groundnext-3b.json
python main.py train ../sft/config/sft/groundnext-7b.json
```

### Reinforcement Learning (RLOO) with verl

We use the [verl](https://github.com/volcengine/verl) framework for reinforcement learning with the RLOO algorithm. This builds upon the SFT checkpoints to further optimize the policy using reward-based learning.

Training scripts are located in `rl/recipe/groundnext/` directory.

**Run RL Training**:

```bash
# For 3B model
./rl/recipe/groundnext/groundnext-3b.sh

# For 7B model  
./rl/recipe/groundnext/groundnext-7b.sh
```

**Key RL Configuration Parameters**:
- **Algorithm**: RLOO (Reward Learning with Likelihood Optimization)
- **Reward Function**: Custom GUI reward function (`reward_clipped.py`)
- **Base Model**: SFT checkpoint from previous stage
- **Batch Size**: 64 for training, 8 for rollout
- **Learning Rate**: 1e-6

---

## 📊 Evaluation

Our evaluation framework builds upon [InfiGUI-G1](https://github.com/InfiXAI/InfiGUI-G1/tree/main/eval) and provides comprehensive evaluation across multiple benchmarks. Use the `eval/` directory for comprehensive evaluation across multiple benchmarks.

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

# Evaluate on multiple benchmarks
python eval.py \
    --model_type qwen25vl \
    --model_name_or_path /path/to/trained/model \
    --benchmark all \
    --task all \
    --language en
```

### Supported Benchmarks

The evaluation framework supports multiple GUI automation benchmarks:
- ScreenSpot-Pro
- ScreenSpot-v2
- MMBench-GUI-Bench
- OSWorld-G
- 

### Evaluation Metrics

- **Success Rate**: Percentage of tasks completed successfully
- **Accuracy**: Precision of GUI element localization
- **Efficiency**: Number of steps required for task completion
- **Generalization**: Performance on unseen applications/layouts

---

## Project Structure

```
GroundCUA/
├── README.md                    # This file
├── eval/                        # Evaluation framework
│   ├── eval.py                 # Main evaluation script
│   ├── data.py                 # Data loading utilities
│   ├── prompts.py              # Prompt processing
│   └── models/                 # Model implementations
├── sft/                        # Supervised Fine-tuning
│   └── config/                 # Training configurations
│       ├── sft/               # SFT-specific configs
│       └── deepspeed/         # DeepSpeed configurations
├── LLaMA-Factory/              # LLaMA-Factory framework
│   └── data/                  # SFT training data
├── rl/                         # Reinforcement Learning
│   ├── recipe/                # Training recipes
│   │   └── groundnext/       # GroundNext-specific scripts
│   └── data/                  # RL training data
└── verl/                       # verl framework
```

---

## Quick Start

```bash
# Setup environment
conda create -n groundcua python=3.11.3 -y
conda activate groundcua
pip install -r requirements.txt

# Run SFT training
cd LLaMA-Factory/
python main.py train ../sft/config/sft/groundnext-3b.json

# Run RL training
./rl/recipe/groundnext/groundnext-3b.sh

# Evaluate model
cd eval/
python eval.py --model_name_or_path /path/to/trained/model
```

---

## Advanced Usage

### Custom Data Preparation

1. **Format your demonstration data** according to the schemas above
2. **Place data files** in appropriate directories:
   - SFT: `LLaMA-Factory/data/`
   - RL: `rl/data/`
3. **Update dataset configurations** in config files
4. **Run training** with custom configurations

### Model Customization

- **Architecture**: Modify model configurations in training scripts
- **Hyperparameters**: Adjust learning rates, batch sizes, and training epochs
- **Reward Functions**: Implement custom reward functions for RL training
- **Evaluation Metrics**: Add custom evaluation benchmarks in `eval/`

---

## Conclusion

GroundCUA represents a significant step forward in desktop grounding research. Through our human-annotated dataset spanning 87 applications (56K screenshots, 3.56M+ elements) with dense keyframe labels, we demonstrate that **high-quality data drives reliable desktop grounding more effectively than sheer data volume**.

The GroundNext family of models achieves state-of-the-art results across five challenging benchmarks despite using substantially less SFT training data than many prior works. This validates our core thesis that expert-driven, densely annotated datasets enable more efficient and effective model training than large-scale synthetic alternatives.

### Key Takeaways

- **Quality over Quantity**: 700K high-quality examples outperform 9M+ synthetic datapoints
- **Cross-Platform Generalization**: Desktop-trained models excel across mobile and web environments  
- **Dense Annotation Value**: Maximum annotation density provides superior supervision signals
- **Expert Demonstrations**: Human-verified annotations enable robust grounding capabilities

### Future Opportunities

By releasing both the GroundCUA dataset and GroundNext models, we aim to unlock grounding as a core capability for end-to-end computer-use agents. The dense annotations enable development of precise reward signals for RL, while the platform metadata supports research on continual learning and cross-domain adaptation.

This work lays the foundation for reliable, adaptable computer-use agents that can perform complex tasks across diverse desktop applications and generalize to new interaction paradigms as they emerge.

---

## Acknowledgements

We thank the following projects and teams:

* [InfiGUI-G1](https://github.com/InfiXAI/InfiGUI-G1) for the evaluation framework foundation
* [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for the excellent SFT training framework
* [verl](https://github.com/volcengine/verl) for the robust RL infrastructure
* [Qwen-2.5-VL](https://huggingface.co/collections/Qwen/qwen25-vl) for the foundation vision-language models
* The computer use and GUI automation research community

---

## Citation

```bibtex
@article{groundcua2025,
  title={Grounding Computer Use Agents on Human Demonstrations},
  author={Aarash Feizi and Shravan Nayak and Xiangru Jian and Kevin Qinghong Lin and Kaixin Li and Rabiul Awal and Xing Han Lü and Johan Obando-Ceron and Juan A. Rodriguez and Nicolas Chapados and David Vazquez and Adriana Romero-Soriano and Reihaneh Rabbany and Perouz Taslakian and Christopher Pal and Spandana Gella and Sai Rajeswar},
  journal={arXiv preprint arXiv:2501.XXXXX},
  year={2025}
}
```

