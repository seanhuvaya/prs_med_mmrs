# PRS-Med: Position Reasoning Segmentation with Vision-Language Model

This repository implements **PRS-Med** (Position Reasoning Segmentation with Vision-Language Model) as described in the [paper](https://arxiv.org/pdf/2505.11872). The implementation supports both TinySAM and SAM-Med2D vision encoders.

## Quick Start

### Training

```bash
python train.py \
    --data_root /path/to/data_v2 \
    --ann_paths /path/to/annotations/head_and_neck.csv,/path/to/annotations/prostate.csv \
    --vlm_path microsoft/llava-med-v1.5-mistral-7b \
    --sam_ckpt weights/tinysam_42.3.pth \
    --encoder_type tinysam \
    --batch_size 4 \
    --epochs 20 \
    --device cuda:0
```

### Training with SAM-Med2D

```bash
python train.py \
    --data_root /path/to/data_v2 \
    --ann_paths /path/to/annotations/head_and_neck.csv \
    --vlm_path microsoft/llava-med-v1.5-mistral-7b \
    --sam_ckpt weights/sam-med2d_b.pth \
    --encoder_type sam_med2d \
    --batch_size 4 \
    --epochs 20 \
    --device cuda:0
```

### Inference

```bash
python infer_original.py \
    --checkpoint checkpoints/llm_seg_10 \
    --vlm_path microsoft/llava-med-v1.5-mistral-7b \
    --sam_ckpt weights/tinysam_42.3.pth \
    --encoder_type tinysam \
    --data_root /path/to/data_v2 \
    --ann_paths /path/to/annotations/head_and_neck.csv \
    --split test \
    --num_samples 10
```

```shell
uv run python -m infer_original \
    --checkpoint /workspace/checkpoints/llm_seg_best_model_epoch_20 \
    --vlm_path microsoft/llava-med-v1.5-mistral-7b \
    --sam_ckpt weights/tinysam_42.3.pth \
    --data_root /workspace/data/prs-med-dataset \
    --ann_paths "/workspace/data/prs-med-dataset/annotations/brain_tumors_ct_scan.csv,/workspace/data/prs-med-dataset/annotations/breast_tumors_ct_scan.csv,/workspace/data/prs-med-dataset/annotations/lung_CT.csv,/workspace/data/prs-med-dataset/annotations/lung_Xray.csv,/workspace/data/prs-med-dataset/annotations/polyp_endoscopy.csv,/workspace/data/prs-med-dataset/annotations/skin_rgbimage.csv" \
    --split test
```

```bash
curl -fsSL --no-buffer -H "Cache-Control: no-cache" -H "Pragma: no-cache" https://raw.githubusercontent.com/seanhuvaya/prs_med_mmrs/refs/heads/master/train.sh \
  | bash -s -- \
  YOUR_AWS_ACCESS_KEY_ID \
  YOUR_AWS_SECRET_ACCESS_KEY \
  us-east-1 \
  prs-med \
  s3://prs-med-dataset/new_data \
  /workspace/data \
  /workspace/checkpoints \
  s3://prs-med-dataset/checkpoints \
  sam_med2d \
  weights/sam2.1_hiera_tiny.pt

```

```bash
curl -fsSL --no-buffer -H "Cache-Control: no-cache" -H "Pragma: no-cache" https://raw.githubusercontent.com/seanhuvaya/prs_med_mmrs/refs/heads/master/evaluate.sh \
  | bash -s -- \
  YOUR_AWS_ACCESS_KEY_ID \
  YOUR_AWS_SECRET_ACCESS_KEY \
  us-east-1 \
  prs-med \
  s3://prs-med-dataset/new_data \
  /workspace/data \
  s3://prs-med-dataset/checkpoints/training_20251203_165217/best_model_epoch_20.pth \
  /workspace/checkpoints/best_model.pth \
  sam_med2d \
  weights/sam2.1_hiera_tiny.pt \
  prostate
```
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A complete PyTorch implementation of **PRS-Med** (Position Reasoning Segmentation with Vision-Language Model) for medical image segmentation. This implementation combines TinySAM vision encoder with LLaVA-Med multimodal language model to perform position-aware medical image segmentation.

## 📋 Table of Contents

- [Features](#-features)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Training](#-training)
- [Evaluation](#-evaluation)
- [Model Architecture](#-model-architecture)
- [Memory Optimization](#-memory-optimization)
- [Multi-GPU Training](#-multi-gpu-training)
- [Supported Modalities](#-supported-modalities)
- [Configuration](#-configuration)
- [Troubleshooting](#-troubleshooting)
- [References](#-references)

## ✨ Features

- ✅ **Position Reasoning**: Understands spatial relationships in medical images
- ✅ **Multi-modal**: Works across 7+ medical imaging modalities
- ✅ **Efficient Training**: LoRA adaptation for efficient fine-tuning
- ✅ **Memory Optimized**: Mixed precision training, gradient checkpointing, and accumulation
- ✅ **Distributed Training**: Multi-GPU support with DDP
- ✅ **Comprehensive Evaluation**: Dice, IoU, Hausdorff distance, and position accuracy metrics
- ✅ **Production Ready**: Robust checkpoint saving with atomic writes and retry logic

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

Download the TinySAM checkpoint and place it in the `weights/` directory:

```bash
mkdir -p weights
# Download tinysam_42.3.pth to weights/tinysam_42.3.pth
```

## 🏃 Quick Start

### Implementation Details

This repository implements the exact PRS-Med architecture from the paper:
- **Model**: `LLMSeg` with `PromptedMaskDecoder` (from `mask_decoder_v5.py`)
- **Loss**: `structure_loss` (weighted BCE + IoU) + classification loss + LLM loss
- **Vision Encoders**: Supports both TinySAM and SAM-Med2D
- **MLLM**: LLaVA-Med with LoRA adaptation

**Data Format:**
- CSV columns: `image_path` (or `image_name`), `question`, `answer`, `position` (optional), `split`
- Supports `data_v2/` structure: `{task}/{split}_images/` and `{task}/{split}_masks/`
- If using `image_name`, the script will construct paths automatically

**Vision Encoder Options:**
- `--encoder_type tinysam`: Use TinySAM encoder (default, lightweight)
- `--encoder_type sam_med2d`: Use SAM-Med2D encoder (better for medical images)

### 1. Prepare Your Data

Organize your data in the MMRS format:

```
data/
├── images_and_masks/
│   ├── image_001.png
│   ├── mask_001.png
│   └── ...
└── csv/
    ├── train.csv
    ├── val.csv
    └── test.csv
```

CSV format should include columns: `image_path`, `mask_path`, `question`, `answer`, `modality`

### 2. Train the Model

**Training with TinySAM (default):**
```bash
python train.py \
    --data_root /path/to/data_v2 \
    --ann_paths /path/to/annotations/head_and_neck.csv,/path/to/annotations/prostate.csv \
    --vlm_path microsoft/llava-med-v1.5-mistral-7b \
    --sam_ckpt weights/tinysam_42.3.pth \
    --encoder_type tinysam \
    --batch_size 4 \
    --epochs 20 \
    --device cuda:0
```

**Training with SAM-Med2D:**
```bash
python train.py \
    --data_root /path/to/data_v2 \
    --ann_paths /path/to/annotations/head_and_neck.csv \
    --vlm_path microsoft/llava-med-v1.5-mistral-7b \
    --sam_ckpt weights/sam-med2d_b.pth \
    --encoder_type sam_med2d \
    --batch_size 4 \
    --epochs 20 \
    --device cuda:0
```

### 3. Evaluate the Model

```bash
python evaluation/benchmark_prs_med.py \
    --checkpoint ./checkpoints/training_*/best_model_epoch_*.pth \
    --data_root ./data
```

## 📁 Project Structure

```
prs_med_mmrs/
├── train.py                      # Main training script (original PRS-Med implementation)
├── infer_original.py             # Inference script (original PRS-Med implementation)
├── evaluation/
│   └── benchmark_prs_med.py      # Evaluation and benchmarking
├── models/
│   ├── llm_seg_original.py       # LLMSeg model (original PRS-Med architecture)
│   ├── vision_backbone/          # Vision encoders
│   │   ├── sam_med2d_encoder.py  # SAM-Med2D encoder
│   │   └── tiny_sam_encoder.py  # TinySAM encoder
│   ├── decoder/
│   │   └── mask_decoder_original.py  # PromptedMaskDecoder (from paper)
│   └── loss/
│       └── original_loss.py      # Loss functions (structure_loss, dice_score, etc.)
├── data/
│   └── dataset.py                # Dataset and data loaders
├── data_utils/
│   └── utils.py                  # Data utilities (load_image, load_annotation, etc.)
├── llava/                        # LLaVA-Med implementation
├── weights/                      # Model checkpoints
│   ├── tinysam_42.3.pth
│   └── sam-med2d_b.pth
└── checkpoints/                   # Training checkpoints
```

**Note:** `train_prs_med.py` is an alternative implementation with a different architecture. Use `train.py` for the original PRS-Med paper implementation.

## 🎓 Training

### Basic Training

```bash
python train.py \
    --data_root /path/to/data_v2 \
    --ann_paths /path/to/annotations/head_and_neck.csv \
    --vlm_path microsoft/llava-med-v1.5-mistral-7b \
    --sam_ckpt weights/tinysam_42.3.pth \
    --encoder_type tinysam \
    --batch_size 4 \
    --epochs 20 \
    --device cuda:0
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
| `--batch_size` | 4 | Batch size per GPU |
| `--learning_rate` | 1e-4 | Learning rate |
| `--epochs` | 20 | Number of training epochs |
| `--image_size` | 1024 | Input image size |
| `--device` | `cuda:0` | Device to use |
| `--save_dir` | `./checkpoints` | Directory to save checkpoints |
| `--cls_loss_weight` | 0.5 | Classification loss weight |
| `--cls_loss_epochs` | 5 | Number of epochs to use classification loss |

### Memory Optimization

For GPUs with limited memory, use gradient accumulation:

```bash
python train_prs_med.py \
    --data_root ./data \
    --batch_size 2 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing \
    --use_amp
```

See [MEMORY_OPTIMIZATION.md](MEMORY_OPTIMIZATION.md) for detailed guidance.

### Checkpoint Saving

Checkpoints are automatically saved:
- **Best model**: When validation loss improves
- **Periodic**: Every 5 epochs
- **Initial**: After epoch 1 (for verification)
- **Final**: At the end of training

Checkpoints include:
- Model state dict
- Optimizer state
- Epoch number
- Timestamp

## 📊 Evaluation

Run evaluation on test data:

```bash
python evaluation/benchmark_prs_med.py \
    --checkpoint ./checkpoints/training_*/best_model_epoch_*.pth \
    --data_root ./data \
    --batch_size 8
```

### Evaluation Metrics

- **Dice Coefficient**: Segmentation overlap
- **IoU Score**: Intersection over Union
- **Hausdorff Distance**: Boundary accuracy
- **Position Accuracy**: Position reasoning correctness

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

The implementation includes several memory optimizations:

1. **Mixed Precision Training (AMP)**: ~50% memory reduction
2. **Gradient Accumulation**: Effective larger batch sizes
3. **Gradient Checkpointing**: Trade compute for memory
4. **Model Compilation**: Additional memory savings

See [MEMORY_OPTIMIZATION.md](MEMORY_OPTIMIZATION.md) for:
- Configuration recommendations by GPU size
- Memory usage breakdown
- Troubleshooting guide

## 🔄 Multi-GPU Training

Train on multiple GPUs using distributed data parallel (DDP):

```bash
# Using torchrun
torchrun --nproc_per_node=4 train_prs_med.py \
    --data_root ./data \
    --batch_size 8

# Using provided script
./run_multi_gpu_train.sh ./data 4
```

**Features:**
- Automatic distributed setup
- Gradient synchronization across GPUs
- Checkpoint saving only on rank 0
- Proper data shuffling with DistributedSampler

See [MULTI_GPU_TRAINING.md](MULTI_GPU_TRAINING.md) for detailed guide.

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

See `configs/hyperparameters.py` for all configuration options.

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
   --batch_size 2 --gradient_accumulation_steps 4
   ```

2. **Enable gradient checkpointing**:
   ```bash
   --gradient_checkpointing
   ```

3. **Check disk space**: Ensure at least 10GB free for checkpoints

See [MEMORY_OPTIMIZATION.md](MEMORY_OPTIMIZATION.md) for detailed solutions.

### Checkpoint Saving Issues

- **Disk space**: Check available disk space
- **Permissions**: Ensure write permissions on checkpoint directory
- **File system**: Network mounts may cause issues

The implementation includes automatic retry logic and atomic writes.

### Distributed Training Issues

- **Port conflicts**: Change `--master_port` if using multiple jobs
- **NCCL errors**: Ensure GPUs are visible with `nvidia-smi`
- **Hanging**: Check network connectivity between nodes

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


