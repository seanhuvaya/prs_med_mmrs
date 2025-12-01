#!/usr/bin/env bash

# Install AWS CLI
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# Setting credentials for profile


# Download dataset
aws s3 sync s3://prs-med-dataset/data/ /workspace/data/

# Clone git repo
git clone https://github.com/seanhuvaya/prs_med_mmrs.git /workspace/prs_med_mmrs/
cd /workspace/prs_mde_mmrs

# Sync dependencies
uv sync

# Train model
uv run python -m train_prs_med --batch_size 8 --data_root /workspace/data/ --checkpoint_dir /workspace/checkpoints