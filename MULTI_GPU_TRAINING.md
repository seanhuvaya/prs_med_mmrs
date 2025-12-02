# Multi-GPU Training Guide

This guide explains how to run PRS-Med training on multiple GPUs using PyTorch's DistributedDataParallel (DDP).

## Quick Start

### Option 1: Using the provided script (Recommended)

```bash
# Run with all available GPUs
./run_multi_gpu_train.sh /path/to/data

# Run with specific number of GPUs
./run_multi_gpu_train.sh /path/to/data 4
```

### Option 2: Using torchrun directly

```bash
# Train on 4 GPUs
torchrun --nproc_per_node=4 train_prs_med.py \
    --data_root /path/to/data \
    --batch_size 8 \
    --learning_rate 1e-4 \
    --num_epochs 20
```

### Option 3: Using torch.distributed.launch (legacy)

```bash
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    train_prs_med.py \
    --data_root /path/to/data \
    --batch_size 8
```

## Key Features

### Automatic Multi-GPU Detection
- The script automatically detects if distributed training is enabled
- Falls back to single GPU/CPU if not running in distributed mode
- No code changes needed - works with existing single-GPU code

### Distributed Data Parallel (DDP)
- Each GPU processes a different subset of the data
- Gradients are synchronized across all GPUs
- Effective batch size = `batch_size * num_gpus`

### Learning Rate Scaling
- Learning rate is automatically scaled by the number of GPUs
- Formula: `effective_lr = base_lr * world_size`
- This maintains the same effective learning rate as single-GPU training

### Data Loading
- Uses `DistributedSampler` to ensure each GPU sees different data
- No data duplication across GPUs
- Proper shuffling maintained across epochs

### Checkpoint Saving
- Only rank 0 (main process) saves checkpoints
- Prevents duplicate checkpoint files
- All checkpoints are saved in the same directory

## Configuration

### Batch Size
- `--batch_size` is the batch size **per GPU**
- Total effective batch size = `batch_size * num_gpus`
- Example: `--batch_size 8` with 4 GPUs = effective batch size of 32

### Learning Rate
- Base learning rate is automatically scaled by `world_size`
- If you set `--learning_rate 1e-4` with 4 GPUs, effective LR is `4e-4`
- This is the standard practice for distributed training

### Number of Workers
- Each GPU process uses `--num_workers` data loading workers
- Total workers = `num_workers * num_gpus`
- Adjust based on your system's CPU cores

## Example Commands

### Training on 2 GPUs
```bash
torchrun --nproc_per_node=2 train_prs_med.py \
    --data_root data \
    --batch_size 8 \
    --num_epochs 20
```

### Training on 4 GPUs with custom settings
```bash
torchrun --nproc_per_node=4 train_prs_med.py \
    --data_root data \
    --batch_size 4 \
    --learning_rate 5e-5 \
    --num_epochs 50 \
    --num_workers 4 \
    --checkpoint_dir ./checkpoints
```

### Training on 8 GPUs (multi-node)
```bash
# Node 0
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 \
    --master_addr=<node0_ip> --master_port=29500 \
    train_prs_med.py --data_root data

# Node 1
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 \
    --master_addr=<node0_ip> --master_port=29500 \
    train_prs_med.py --data_root data
```

## Performance Tips

1. **Batch Size**: Start with the same per-GPU batch size as single-GPU training
2. **Learning Rate**: The script automatically scales LR, but you can adjust if needed
3. **Gradient Accumulation**: For very large models, consider gradient accumulation instead of increasing batch size
4. **Mixed Precision**: Consider using `torch.cuda.amp` for faster training (not yet implemented)

## Troubleshooting

### Issue: "NCCL error" or "connection timeout"
- **Solution**: Increase `--master_port` or use a different port
- Check firewall settings if using multi-node

### Issue: "CUDA out of memory"
- **Solution**: Reduce `--batch_size` per GPU
- The effective batch size will still be larger due to multiple GPUs

### Issue: "Address already in use"
- **Solution**: Change the `--master_port` in torchrun
- Or wait for previous training to finish

### Issue: Slow data loading
- **Solution**: Increase `--num_workers` (but not too high - typically 2-4 per GPU)
- Ensure data is on fast storage (SSD, not network drive)

## Single GPU Fallback

The code automatically falls back to single GPU training if:
- `torchrun` is not used
- `--local_rank` is not set
- Only one GPU is available

Simply run:
```bash
python train_prs_med.py --data_root data
```

This maintains backward compatibility with existing single-GPU workflows.

## Monitoring

- All print statements are only shown on rank 0 to avoid clutter
- Check GPU utilization: `nvidia-smi -l 1`
- Monitor training logs for loss values and checkpoint saves

## Notes

- Validation loss is aggregated across all GPUs for accurate metrics
- Model checkpoints contain the full model state (not DDP-wrapped)
- The code handles both distributed and non-distributed training seamlessly

