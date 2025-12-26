# PRS-Med: Position Reasoning Segmentation with Vision-Language Model

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

This repository implements **PRS-Med** (Position Reasoning Segmentation with Vision-Language Model) as described in the [paper](https://arxiv.org/pdf/2505.11872). The implementation supports both TinySAM and SAM-Med2D vision encoders.

A complete PyTorch implementation of **PRS-Med** for medical image segmentation. This implementation combines TinySAM or SAM-Med2D vision encoder with LLaVA-Med multimodal language model to perform position-aware medical image segmentation.

## 📋 Table of Contents

- [Features](#-features)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Dataset](#-dataset)
- [Training](#-training)
- [Inference](#-inference)
- [Evaluation](#-evaluation)
- [Project Structure](#-project-structure)
- [Model Architecture](#-model-architecture)
- [Memory Optimization](#-memory-optimization)
- [Supported Modalities](#-supported-modalities)
- [Configuration](#-configuration)
- [Troubleshooting](#-troubleshooting)
- [References](#-references)

## ✨ Features

- ✅ **Position Reasoning**: Understands spatial relationships in medical images
- ✅ **Multi-modal**: Works across 7+ medical imaging modalities
- ✅ **Efficient Training**: LoRA adaptation for efficient fine-tuning
- ✅ **Memory Optimized**: Mixed precision training (AMP) with bfloat16
- ✅ **Comprehensive Evaluation**: Dice, IoU, Hausdorff distance, and position accuracy metrics

## 🚀 Installation

### Prerequisites

- Python 3.11+
- CUDA-capable GPU (recommended) or CPU
- [uv](https://github.com/astral-sh/uv) package manager

### Install Dependencies

```bash
# Clone the repository
git clone <repository-url>
cd prs_med_mmrs

# Install dependencies using uv
uv sync

# Or install manually
pip install -r requirements.txt
```

### Download Model Weights

Download the vision encoder checkpoints and place them in the `weights/` directory:

```bash
mkdir -p weights
# Download tinysam_42.3.pth to weights/tinysam_42.3.pth
# Download sam-med2d_b.pth to weights/sam-med2d_b.pth
```

## 🏃 Quick Start

### ⚙️ Setup Environment

Run the `setup.sh` script to automatically prepare the project environment:

```shell
bash <(curl -Ls https://raw.githubusercontent.com/seanhuvaya/prs_med_mmrs/refs/heads/code-refactor-with-sammed/setup.sh) <data_download_dir> <project_repo_dir>
```

#### What the script does

**Downloads the dataset**
- Downloads the training dataset from a public Amazon S3 bucket used by the models in this project
- If the AWS CLI is not already installed, the script installs it automatically before downloading the data

**Clones the project repository**
- Clones the full project source code into the specified repository directory

**Installs and syncs dependencies**
- Installs [uv](https://github.com/astral-sh/uv) and synchronizes all project dependencies to ensure a reproducible environment

### Implementation Details

This repository implements the exact PRS-Med architecture from the paper:
- **Model**: `LLMSeg` with `PromptedMaskDecoder`
- **Loss**: `structure_loss` (weighted BCE + IoU) + classification loss + LLM loss
- **Vision Encoders**: Supports both TinySAM and SAM-Med2D
- **MLLM**: LLaVA-Med with LoRA adaptation

**Vision Encoder Options:**
- `--encoder_type tinysam`: Use TinySAM encoder (default, lightweight)
- `--encoder_type sam_med2d`: Use SAM-Med2D encoder (better for medical images)

## 📦 Dataset

### Dataset (Public S3)

The dataset is publicly available at:
```shell
s3://prs-med-experiments/data/
```

or via HTTPS:
```
https://prs-med-experiments.s3.us-east-1.amazonaws.com/data/
```

### Dataset Structure

```text
data/
├── our-dataset/
│   ├── annotations/
│   │   ├── head_and_neck.csv
│   │   └── prostate.csv
│   ├── head_and_neck/
│   │   ├── test_images/
│   │   │   ├── 0204_channel_08.png
│   │   │   └── ...
│   │   ├── test_masks/
│   │   │   ├── 0204_channel_08.png
│   │   │   └── ...
│   │   ├── train_images/
│   │   ├── train_masks/
│   │   ├── val_images/
│   │   └── val_masks/
│   └── prostate/
│       └── ...
├── prs-med-dataset/
│   ├── annotations/
│   │   ├── brain_tumors_ct_scan.csv
│   │   └── ...
│   ├── brain_tumors_ct_scan/
│   │   ├── test_images/
│   │   │   ├── 100.png
│   │   │   └── ...
│   │   ├── test_masks/
│   │   │   ├── 100.png
│   │   │   └── ...
│   │   ├── train_images/
│   │   ├── train_masks/
│   │   ├── val_images/
│   │   └── val_masks/
│   └── ...
```

**Data Format:**
- CSV columns: `image_path` (or `image_name`), `question`, `answer`, `position` (optional), `split`
- Supports directory structure: `{task}/{split}_images/` and `{task}/{split}_masks/`
- If using `image_name`, the script will construct paths automatically

## 🎓 Training

This project supports training on both the **PRS-Med dataset** (to reproduce reported results) and **custom research datasets** using the same training pipeline.

### Training on PRS-Med (Reproducibility)

To reproduce results on the PRS-Med benchmark, run:

```shell
uv run python -m train \
    --data_root /path/to/data/prs-med-dataset \
    --ann_paths "/path/to/data/prs-med-dataset/annotations/brain_tumors_ct_scan.csv,/path/to/data/prs-med-dataset/annotations/breast_tumors_ct_scan.csv,/path/to/data/prs-med-dataset/annotations/lung_CT.csv,/path/to/data/prs-med-dataset/annotations/lung_Xray.csv,/path/to/data/prs-med-dataset/annotations/polyp_endoscopy.csv,/path/to/data/prs-med-dataset/annotations/skin_rgbimage.csv" \
    --vlm_path microsoft/llava-med-v1.5-mistral-7b \
    --sam_ckpt weights/tinysam_42.3.pth \
    --encoder_type tinysam \
    --batch_size 8 \
    --epochs 20
```

This configuration uses the official PRS-Med annotations and the LLaVA-Med v1.5 (Mistral-7B) vision–language backbone.

### Training on a Custom Research Dataset

To train the model on a custom research dataset, specify the corresponding data root and annotation files:

```shell
uv run python -m train \
    --data_root /path/to/data/our-dataset \
    --ann_paths "/path/to/data/our-dataset/annotations/head_and_neck.csv,/path/to/data/our-dataset/annotations/prostate.csv" \
    --vlm_path microsoft/llava-med-v1.5-mistral-7b \
    --sam_ckpt weights/tinysam_42.3.pth \
    --encoder_type tinysam \
    --batch_size 8 \
    --epochs 20
```

### Training with SAM-Med2D

For better performance on medical images, use the SAM-Med2D encoder:

```shell
uv run python -m train \
    --data_root /path/to/data/prs-med-dataset \
    --ann_paths "/path/to/data/prs-med-dataset/annotations/brain_tumors_ct_scan.csv" \
    --vlm_path microsoft/llava-med-v1.5-mistral-7b \
    --sam_ckpt weights/sam-med2d_b.pth \
    --encoder_type sam_med2d \
    --batch_size 8 \
    --epochs 20 \
    --save_dir ./checkpoints
```

### Training Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_root` | Required | Path to data directory |
| `--ann_paths` | Required | Comma-separated paths to annotation CSV files |
| `--vlm_path` | Required | Path to LLaVA-Med model (HF ID or local path) |
| `--sam_ckpt` | Required | Path to vision encoder checkpoint |
| `--encoder_type` | `tinysam` | Vision encoder: `tinysam` or `sam_med2d` |
| `--sam_model_type` | `vit_t` | TinySAM model type (ignored for SAM-Med2D) |
| `--batch_size` | 8 | Batch size |
| `--learning_rate` | 1e-4 | Learning rate |
| `--epochs` | 20 | Number of training epochs |
| `--image_size` | 1024 | Input image size |
| `--device` | `cuda:0` | Device to use |
| `--save_dir` | `./checkpoints` | Directory to save checkpoints |
| `--cls_loss_weight` | 0.5 | Classification loss weight |
| `--cls_loss_epochs` | 5 | Number of epochs to use classification loss |
| `--load_8bit` | False | Load model in 8-bit |
| `--load_4bit` | False | Load model in 4-bit |
| `--num_workers` | 4 | Number of data loader workers |
| `--log_file` | `logs/training.log` | Log file path |

### Notes

- `--ann_paths` accepts a comma-separated list of annotation CSV files
- All datasets must follow the directory structure described in the Dataset Structure section
- Hyperparameters such as `batch_size` and `epochs` can be adjusted based on available GPU memory and dataset size

### Checkpoint Saving

Checkpoints are automatically saved:
- **Per epoch**: Saved after each epoch as `llm_seg_{epoch+1}`
- **Best model**: When validation Dice score improves, saved to `best_model` directory

Checkpoints include:
- LoRA adapter weights
- Tokenizer
- Image encoder state dict
- Mask decoder state dict
- Classification head state dict

## 🔮 Inference

Run inference on test data:

```shell
uv run python -m infer \
    --checkpoint /workspace/checkpoints/llm_seg_best_model_epoch_20 \
    --vlm_path microsoft/llava-med-v1.5-mistral-7b \
    --sam_ckpt weights/tinysam_42.3.pth \
    --encoder_type tinysam \
    --data_root /workspace/data/prs-med-dataset \
    --ann_paths "/workspace/data/prs-med-dataset/annotations/brain_tumors_ct_scan.csv,/workspace/data/prs-med-dataset/annotations/breast_tumors_ct_scan.csv,/workspace/data/prs-med-dataset/annotations/lung_CT.csv,/workspace/data/prs-med-dataset/annotations/lung_Xray.csv,/workspace/data/prs-med-dataset/annotations/polyp_endoscopy.csv,/workspace/data/prs-med-dataset/annotations/skin_rgbimage.csv" \
    --split test
```

## 📊 Evaluation

**Note:** You need a HuggingFace token on an account with access to the meta-llama model.

Run evaluation on inference results:

```shell
python -m evaluation.eval \
    --results_csv inference_outputs_original/inference_20251219_192029/results_test_20251219_192029.csv \
    --qwen_model_name Qwen/Qwen2-7B-Instruct \
    --llama_model_name meta-llama/Llama-3-8B-Instruct \
    --use_auth_token
```

Or with a custom output path:

```shell
python -m evaluation.eval \
    --results_csv /path/to/inference_results.csv \
    --output_path /path/to/eval_results \
    --use_auth_token
```

### Evaluation Metrics

- **Dice Coefficient**: Segmentation overlap
- **IoU Score**: Intersection over Union
- **Hausdorff Distance**: Boundary accuracy
- **Position Accuracy**: Position reasoning correctness

## 📁 Project Structure

```
prs_med_mmrs/
├── train.py                      # Main training script
├── infer.py                      # Inference script
├── evaluation/
│   └── eval.py                   # Evaluation script
├── models/
│   ├── llm_seg.py                # LLMSeg model
│   ├── vision_backbone/          # Vision encoders
│   │   ├── sam_med2d_encoder.py  # SAM-Med2D encoder
│   │   └── tiny_sam_encoder.py   # TinySAM encoder
│   ├── decoder/                  # Mask decoders
│   │   ├── mask_decoder.py
│   │   ├── mask_prediction_module.py
│   │   └── fusion_module.py
│   ├── mllm/                     # Multimodal language model
│   │   ├── llava_med_mllm.py
│   │   └── llava_med_lora_adapter.py
│   └── loss/
│       └── objective_function.py # Loss functions
├── data/
│   └── dataset.py                 # Dataset and data loaders
├── data_utils/
│   └── utils.py                   # Data utilities
├── llava/                        # LLaVA-Med implementation
├── weights/                       # Model checkpoints
│   ├── tinysam_42.3.pth
│   └── sam-med2d_b.pth
└── checkpoints/                  # Training checkpoints
```

## 🧠 Model Architecture

### Components

1. **Vision Encoder**: Supports both TinySAM and SAM-Med2D
   - Input: 1024×1024 medical images
   - Output: 256-channel feature maps (16×16)
   - TinySAM: Lightweight, efficient
   - SAM-Med2D: Medical-domain pretrained, better performance

2. **LLaVA-Med MLLM**: Multimodal language model
   - Base: Mistral-7B with LoRA adaptation
   - Processes images and text prompts
   - Output: Multimodal embeddings (hidden states)

3. **PromptedMaskDecoder**: Transformer-based decoder
   - Cross-attention between image features and prompt embeddings
   - Transformer encoder layers for feature refinement
   - Upsamples from 16×16 to 1024×1024
   - Generates binary segmentation masks

### Forward Pass

```
Image (1024×1024) → Vision Encoder → Visual Features (256×16×16)
                                 ↓
Text Prompt → LLaVA-Med → Prompt Embeddings
                                 ↓
                    PromptedMaskDecoder → Mask (1024×1024)
```

## 💾 Memory Optimization

The implementation includes memory optimizations:

1. **Mixed Precision Training (AMP)**: Uses `autocast` with `bfloat16` for reduced memory usage during training

For GPUs with limited memory, reduce batch size:

```bash
uv run python -m train \
    --data_root ./data \
    --batch_size 2 \
    --epochs 20
```

You can also use 8-bit or 4-bit quantization when loading the model:

```bash
uv run python -m train \
    --data_root ./data \
    --load_8bit  # or --load_4bit
```

## 🏥 Supported Modalities

The model supports multiple medical imaging modalities:

1. **Brain Tumors CT Scan** (`brain_tumors_ct_scan`)
2. **Breast Tumors CT Scan** (`breast_tumors_ct_scan`)
3. **Dental X-ray** (`dental_xray`)
4. **Lung CT** (`lung_CT`)
5. **Lung X-ray** (`lung_Xray`)
6. **Polyp Endoscopy** (`polyp_endoscopy`)
7. **Skin RGB Image** (`skin_rgbimage`)

## ⚙️ Configuration

### Hyperparameters

Default hyperparameters (from paper):

```python
batch_size = 8
learning_rate = 1e-4
num_epochs = 20
image_size = 1024
lora_rank = 16
lora_alpha = 16
lora_dropout = 0.05
lambda_seg = 1.0      # Segmentation loss weight
lambda_txt = 0.5      # Text loss weight
weight_decay = 0.01
max_grad_norm = 1.0
```

### Position Reasoning

The model uses template-based question-answer pairs:

**Questions:**
- "Where is the lesion located in this {modality}?"
- "What is the anatomical position of the tumour?"
- "Can you identify the tumour's location?"

**Answers:**
- Position descriptions: "top-left", "top-right", "bottom-left", "bottom-right"
- Contextual: "near-center", "upper region", etc.

## 🐛 Troubleshooting

### Out of Memory (OOM) Errors

1. **Reduce batch size**:
   ```bash
   --batch_size 2
   ```

2. **Use quantization**:
   ```bash
   --load_8bit  # or --load_4bit
   ```

3. **Check disk space**: Ensure at least 10GB free for checkpoints

### Checkpoint Saving Issues

- **Disk space**: Check available disk space
- **Permissions**: Ensure write permissions on checkpoint directory
- **File system**: Network mounts may cause issues

The implementation includes automatic retry logic and atomic writes.

## 📚 References

- **Paper**: [PRS-Med: Position Reasoning Segmentation with Vision-Language Model in Medical Imaging](https://arxiv.org/pdf/2505.11872)
- **TinySAM**: [TinySAM: Segment Anything Model in Less Than 50MB](https://github.com/xinghaochen/TinySAM)
- **LLaVA-Med**: [LLaVA-Med: Training a Large Language-and-Vision Assistant for Biomedicine](https://github.com/microsoft/LLaVA-Med)

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- TinySAM team for the efficient vision encoder
- LLaVA-Med team for the multimodal language model
- PyTorch team for the excellent deep learning framework

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or issues, please open an issue on GitHub.

---

**Ready to train your PRS-Med model!** 🎉
