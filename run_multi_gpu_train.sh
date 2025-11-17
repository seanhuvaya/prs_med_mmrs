#!/bin/bash

# Multi-GPU Training Script for PRS-Med
# Usage: ./run_multi_gpu_train.sh <data_root> [num_gpus]

# Set default values
DATA_ROOT=${1:-"data"}
NUM_GPUS=${2:-$(nvidia-smi --list-gpus | wc -l)}

# Check if CUDA is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "Error: nvidia-smi not found. CUDA is required for multi-GPU training."
    exit 1
fi

# Check if torchrun is available
if ! command -v torchrun &> /dev/null; then
    echo "Error: torchrun not found. Please install PyTorch with distributed support."
    exit 1
fi

echo "Starting multi-GPU training with $NUM_GPUS GPUs"
echo "Data root: $DATA_ROOT"
echo ""

# Run training with torchrun
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=29500 \
    train_prs_med.py \
    --data_root "$DATA_ROOT" \
    --batch_size 8 \
    --learning_rate 1e-4 \
    --num_epochs 20 \
    --checkpoint_dir ./checkpoints \
    --num_workers 4 \
    --seed 42

echo ""
echo "Training completed!"

