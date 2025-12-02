#!/bin/bash

# Optimized Training Script for 80GB GPU (A100/H100)
# This configuration maximizes training speed while using available memory efficiently

# Set default values
DATA_ROOT=${1:-"data"}
NUM_GPUS=${2:-1}

echo "=========================================="
echo "PRS-Med Training - 80GB GPU Configuration"
echo "=========================================="
echo "Data root: $DATA_ROOT"
echo "Number of GPUs: $NUM_GPUS"
echo ""

# Check if CUDA is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "Error: nvidia-smi not found. CUDA is required."
    exit 1
fi

# Check if data root exists
if [ ! -d "$DATA_ROOT" ]; then
    echo "Error: Data root directory '$DATA_ROOT' does not exist."
    exit 1
fi

# Display GPU information
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

# Create logs directory
mkdir -p ./logs

# Single GPU training
if [ "$NUM_GPUS" -eq 1 ]; then
    echo "Starting single GPU training..."
    echo ""
    
    python train_prs_med.py \
        --data_root "$DATA_ROOT" \
        --batch_size 8 \
        --learning_rate 1e-4 \
        --num_epochs 20 \
        --checkpoint_dir ./checkpoints \
        --num_workers 8 \
        --seed 42 \
        --use_amp \
        --compile_model \
        --image_size 1024
    
    echo ""
    echo "Training completed!"
else
    # Multi-GPU training
    echo "Starting multi-GPU training with $NUM_GPUS GPUs..."
    echo ""
    
    # Check if uv is available
    if ! command -v uv &> /dev/null; then
        echo "Error: uv not found. Please install uv: https://github.com/astral-sh/uv"
        exit 1
    fi
    
    uv run python -m torch.distributed.run \
        --nproc_per_node=$NUM_GPUS \
        --nnodes=1 \
        --node_rank=0 \
        --master_addr=localhost \
        --master_port=29500 \
        --log_dir=./logs \
        train_prs_med.py \
        --data_root "$DATA_ROOT" \
        --batch_size 8 \
        --learning_rate 1e-4 \
        --num_epochs 20 \
        --checkpoint_dir ./checkpoints \
        --num_workers 8 \
        --seed 42 \
        --use_amp \
        --compile_model \
        --image_size 1024
    
    echo ""
    echo "Training completed!"
fi

