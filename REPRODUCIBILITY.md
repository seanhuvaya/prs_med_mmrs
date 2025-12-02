# Reproducibility Guide for PRS-Med

This document describes the reproducibility features implemented to help reproduce the results from the PRS-Med paper.

## ✅ Implemented Features

### 1. Reproducibility Settings

**Location:** `train_prs_med.py`

- **Random Seed Control**: Added `set_seed()` function that sets seeds for:
  - Python `random` module
  - NumPy
  - PyTorch (CPU and all CUDA devices)
  - Python hash seed (via environment variable)
  
- **Deterministic Training**: 
  - `torch.backends.cudnn.deterministic = True`
  - `torch.backends.cudnn.benchmark = False`
  - Optional full deterministic mode via `--deterministic` flag

**Usage:**
```bash
python train_prs_med.py --data_root /path/to/data --seed 42 --deterministic
```

### 2. Hyperparameters Configuration

**Location:** `configs/hyperparameters.py`

- **Documented Paper Values**: All hyperparameters from the PRS-Med paper are documented
- **Predefined Configurations**:
  - `PAPER_CONFIG`: Exact values from the paper
  - `FAST_TEST_CONFIG`: Quick test configuration
  - `LARGE_BATCH_CONFIG`: For GPUs with more memory

**Key Hyperparameters (from paper):**
- Batch size: 8
- Learning rate: 1e-4
- Lambda_seg: 1.0
- Lambda_txt: 0.5
- LoRA rank: 16
- LoRA alpha: 16
- LoRA dropout: 0.05
- Weight decay: 0.01
- Max gradient norm: 1.0

**Usage:**
```python
from configs.hyperparameters import get_config, PAPER_CONFIG

# Use paper configuration
config = get_config("paper")
# or
config = PAPER_CONFIG
```

### 3. Enhanced Evaluation Metrics

**Location:** `evaluation/benchmark_prs_med.py`

#### Segmentation Metrics:
- ✅ **Dice Coefficient (mDice)**: Already implemented
- ✅ **IoU Score (mIoU)**: Already implemented  
- ✅ **Hausdorff Distance (HD95)**: **NEW** - 95th percentile Hausdorff distance for boundary accuracy

#### Position Reasoning Metrics:
- ✅ **Exact Match Accuracy**: **NEW** - Exact string matching (case-insensitive)
- ✅ **Keyword Match Accuracy**: **NEW** - Position keyword extraction and matching
- ✅ **LLM-based Evaluation**: Enhanced with fallback to keyword matching

**Usage:**
```python
from evaluation.benchmark_prs_med import evaluate_prs_med

metrics = evaluate_prs_med(model, test_loader, device, use_llm_eval=True)
print(f"mDice: {metrics['mDice']:.4f}")
print(f"mIoU: {metrics['mIoU']:.4f}")
print(f"mHD95: {metrics['mHD95']:.2f}")
print(f"Exact Match: {metrics['exact_match_acc']:.4f}")
print(f"Keyword Match: {metrics['keyword_match_acc']:.4f}")
```

## 📋 Reproducibility Checklist

To ensure reproducible results:

1. ✅ **Set Random Seed**: Use `--seed 42` (or your preferred seed)
2. ✅ **Use Paper Hyperparameters**: Import from `configs.hyperparameters.PAPER_CONFIG`
3. ✅ **Enable Deterministic Mode**: Use `--deterministic` flag (may be slower)
4. ✅ **Fixed Data Splits**: Ensure your dataset has fixed train/val/test splits
5. ✅ **Same Environment**: Use same PyTorch version, CUDA version, etc.

## 🔧 Training with Reproducibility

```bash
# Reproducible training with paper hyperparameters
python train_prs_med.py \
    --data_root /path/to/data \
    --seed 42 \
    --deterministic \
    --batch_size 8 \
    --learning_rate 1e-4 \
    --lambda_seg 1.0 \
    --lambda_txt 0.5 \
    --lora_rank 16 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --num_epochs 20
```

## 📊 Evaluation

```bash
# Run comprehensive evaluation
python -c "
from train_prs_med import PRSMedModel
from data.dataset import PRSMedDataLoader
from evaluation.benchmark_prs_med import evaluate_prs_med
import torch

# Load model
model = PRSMedModel(args, device)
model.load_state_dict(torch.load('checkpoints/best_model.pth'))

# Load test data
data_loader = PRSMedDataLoader(data_root='/path/to/data')
test_loader = data_loader.get_dataloader('test')

# Evaluate
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
metrics = evaluate_prs_med(model, test_loader, device)
"
```

## 📝 Notes

- **Deterministic Mode**: Enabling `--deterministic` may reduce performance but ensures full reproducibility
- **LLM Evaluation**: Requires `HF_TOKEN` environment variable for LLM-based position reasoning evaluation
- **Hausdorff Distance**: Computes HD95 (95th percentile) which is more robust than standard HD
- **Keyword Matching**: Works without external dependencies, always available for position reasoning evaluation

## 🐛 Known Limitations

1. **CUDA Determinism**: Some CUDA operations may still have non-deterministic behavior even with deterministic flags
2. **DataLoader Workers**: Multiple workers may introduce non-determinism; consider `num_workers=0` for full reproducibility
3. **Floating Point**: Different hardware may produce slightly different results due to floating-point precision

## 📚 References

- PRS-Med Paper: https://arxiv.org/pdf/2505.11872
- PyTorch Reproducibility: https://pytorch.org/docs/stable/notes/randomness.html

