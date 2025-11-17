# Memory Optimization Guide

## Problem: Out of Memory (OOM) Errors

When training with batch size 8 (as mentioned in the paper), you may encounter OOM errors due to:

1. **Large model size**: LLaVA-Med (Mistral-7B) + TinySAM encoder
2. **High resolution images**: 1024x1024 input images
3. **Large mask outputs**: 1024x1024 segmentation masks
4. **No memory optimizations**: Original code uses float32 everywhere

## Solutions Implemented

### 1. Mixed Precision Training (AMP) - **Enabled by Default**

**Memory Savings**: ~50% reduction in memory usage

Mixed precision training uses float16 for forward passes while keeping float32 for loss computation and gradient accumulation. This is enabled by default.

```bash
# AMP is enabled by default, but you can disable it:
python train_prs_med.py --data_root /path/to/data --no-use_amp
```

### 2. Gradient Accumulation

**Memory Savings**: Allows effective batch size 8 with smaller physical batch sizes

Instead of processing batch_size=8 at once, you can use a smaller batch size and accumulate gradients:

```bash
# Use batch_size=2 with gradient_accumulation_steps=4 to get effective batch size of 8
python train_prs_med.py \
    --data_root /path/to/data \
    --batch_size 2 \
    --gradient_accumulation_steps 4
```

**Benefits**:
- Physical batch size: 2 (uses less memory)
- Effective batch size: 2 × 4 = 8 (same as paper)
- Same training dynamics as batch_size=8

### 3. Gradient Checkpointing

**Memory Savings**: ~30-50% reduction (trades compute for memory)

Gradient checkpointing recomputes activations during backward pass instead of storing them:

```bash
python train_prs_med.py \
    --data_root /path/to/data \
    --gradient_checkpointing
```

**Note**: This makes training slower (~20-30% slower) but uses significantly less memory.

### 4. Model Compilation (PyTorch 2.0+)

**Memory Savings**: ~10-15% reduction + faster training

Compiles the model for better memory efficiency:

```bash
python train_prs_med.py \
    --data_root /path/to/data \
    --compile_model
```

## Recommended Configurations

### Configuration 0: 80GB GPU (A100/H100) - **Optimal for Large GPUs**

With an 80GB GPU, you have plenty of memory. Use this configuration for maximum speed:

```bash
python train_prs_med.py \
    --data_root /path/to/data \
    --batch_size 8 \
    --use_amp \
    --compile_model \
    --num_workers 8
```

**Memory Usage**: ~12-15 GB (plenty of headroom)
**Training Speed**: Maximum (AMP + compilation for speed)

**Optional: Larger Batch Sizes for Faster Training**

If you want to train faster, you can increase batch size:

```bash
# Batch size 16 (2x faster per epoch, may need to scale LR)
python train_prs_med.py \
    --data_root /path/to/data \
    --batch_size 16 \
    --learning_rate 2e-4 \
    --use_amp \
    --compile_model \
    --num_workers 8

# Batch size 32 (4x faster per epoch, scale LR accordingly)
python train_prs_med.py \
    --data_root /path/to/data \
    --batch_size 32 \
    --learning_rate 4e-4 \
    --use_amp \
    --compile_model \
    --num_workers 8
```

**Note**: When increasing batch size, you may want to scale learning rate proportionally (e.g., batch_size 16 → LR 2e-4, batch_size 32 → LR 4e-4). However, some research suggests square root scaling (batch_size 16 → LR 1.41e-4) or keeping LR the same. Experiment to find what works best.

### Configuration 1: Maximum Memory Savings (for limited GPU memory)

```bash
python train_prs_med.py \
    --data_root /path/to/data \
    --batch_size 2 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing \
    --use_amp
```

**Memory Usage**: ~40-50% of original
**Training Speed**: ~60-70% of original (due to checkpointing)

### Configuration 2: Balanced (Recommended)

```bash
python train_prs_med.py \
    --data_root /path/to/data \
    --batch_size 4 \
    --gradient_accumulation_steps 2 \
    --use_amp
```

**Memory Usage**: ~50-60% of original
**Training Speed**: ~90-95% of original

### Configuration 3: Paper Settings (if you have enough memory)

```bash
python train_prs_med.py \
    --data_root /path/to/data \
    --batch_size 8 \
    --use_amp
```

**Memory Usage**: ~50% of original (due to AMP)
**Training Speed**: ~95-100% of original

## Understanding Memory Usage

### Memory Breakdown (Batch Size 8, 1024x1024 images):

1. **Input Images**: ~100 MB (8 × 3 × 1024 × 1024 × 4 bytes)
2. **Vision Backbone (TinySAM)**: ~500-800 MB (model + activations)
3. **MLLM (LLaVA-Med)**: ~14-16 GB (Mistral-7B base model, even when frozen)
4. **Mask Prediction**: ~200-300 MB (intermediate activations)
5. **Gradients**: ~1-2 GB (for trainable parameters)
6. **Optimizer States**: ~2-4 GB (AdamW momentum buffers)

**Total**: ~18-24 GB for batch_size=8

### With Optimizations:

- **AMP**: Reduces to ~9-12 GB
- **Gradient Accumulation (batch_size=2)**: Reduces to ~4.5-6 GB
- **Gradient Checkpointing**: Further reduces by 30-50%

## Troubleshooting

### Still Getting OOM Errors?

1. **Reduce batch size further**:
   ```bash
   --batch_size 1 --gradient_accumulation_steps 8
   ```

2. **Enable all optimizations**:
   ```bash
   --batch_size 1 \
   --gradient_accumulation_steps 8 \
   --gradient_checkpointing \
   --use_amp
   ```

3. **Check GPU memory**:
   ```python
   import torch
   print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
   print(f"Available: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
   ```

4. **Reduce image size** (if acceptable):
   ```bash
   --image_size 512  # Instead of 1024
   ```

### Performance Tips

1. **Use `pin_memory=True`** (already enabled) for faster data transfer
2. **Use `non_blocking=True`** (already enabled) for async data transfer
3. **Clear cache periodically** (already implemented every 10 batches)
4. **Use multiple GPUs** with distributed training:
   ```bash
   torchrun --nproc_per_node=4 train_prs_med.py --data_root /path/to/data --batch_size 2
   ```

## Example: Training with Limited Memory (16GB GPU)

```bash
python train_prs_med.py \
    --data_root /path/to/data \
    --batch_size 1 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --use_amp \
    --learning_rate 1e-4 \
    --num_epochs 20
```

This gives you:
- Effective batch size: 8 (same as paper)
- Memory usage: ~6-8 GB
- Training speed: ~50-60% of original (due to checkpointing)

## Monitoring Memory Usage

Add this to your training script to monitor memory:

```python
if torch.cuda.is_available():
    print(f"GPU Memory Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    print(f"GPU Memory Reserved: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
    print(f"GPU Memory Cached: {torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
```

## Summary

The key optimizations are:
1. ✅ **Mixed Precision (AMP)** - Enabled by default, ~50% memory savings + faster training
2. ✅ **Gradient Accumulation** - Use smaller batches, accumulate gradients
3. ✅ **Gradient Checkpointing** - Trade compute for memory (optional, not needed for 80GB GPUs)
4. ✅ **Model Compilation** - Additional optimization for speed (recommended for 80GB GPUs)

### Quick Start Guide by GPU Size:

- **80GB GPU (A100/H100)**: Use Configuration 0 - batch_size 8-32, AMP + compilation
- **24GB GPU (RTX 3090/4090)**: Use Configuration 2 - batch_size 4-8, AMP
- **16GB GPU (RTX 3080/4080)**: Use Configuration 1 - batch_size 2-4, AMP + gradient accumulation
- **8GB GPU**: Use Configuration 1 with batch_size 1, all optimizations enabled

Start with the configuration matching your GPU and adjust based on your specific setup.

