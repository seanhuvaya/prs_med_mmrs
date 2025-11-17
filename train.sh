#!/usr/bin/env bash
set -e

# === CONFIGURATION ===
GPU_INSTANCE_IP="<YOUR_LAMBDA_INSTANCE_IP>"
SSH_KEY_PATH="/path/to/your/lambda-key.pem"
REPO_URL="<YOUR_PRS_MED_REPO_URL>"
DATA_ROOT_REMOTE="/home/ubuntu/data/prs_med_mmrs"
BATCH_SIZE=8
EPOCHS=20
IMG_SIZE=1024
LR=1e-4
USE_LORA_FLAG="--use_lora"

# === STEP 1: SSH and setup environment ===
ssh -i ${SSH_KEY_PATH} ubuntu@${GPU_INSTANCE_IP} << 'EOF'
set -e

# Navigate to home
cd ~
# Clone repo
if [ ! -d prs_med ]; then
  git clone ${REPO_URL} prs_med
fi
cd prs_med

# Create virtualenv if not exists
if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
EOF

# === STEP 2: Upload or sync dataset ===
# (choose one method)

# Method A: SCP local → remote (uncomment and modify local path)
# scp -r -i ${SSH_KEY_PATH} /local/path/to/dataset ubuntu@${GPU_INSTANCE_IP}:${DATA_ROOT_REMOTE}

# Method B: Use AWS S3 sync (remote) — requires AWS CLI configured on remote
ssh -i ${SSH_KEY_PATH} ubuntu@${GPU_INSTANCE_IP} << 'EOF'
mkdir -p ${DATA_ROOT_REMOTE}
aws s3 sync s3://your-bucket/prs_med_mmrs ${DATA_ROOT_REMOTE}
EOF

# === STEP 3: Launch training ===
ssh -i ${SSH_KEY_PATH} ubuntu@${GPU_INSTANCE_IP} << 'EOF'
cd ~/prs_med
source .venv/bin/activate

python train_prs_med.py \
  --data_root ${DATA_ROOT_REMOTE} \
  --train_csv ${DATA_ROOT_REMOTE}/csv/mmrs_train.csv \
  --val_csv   ${DATA_ROOT_REMOTE}/csv/mmrs_val.csv \
  --img_size ${IMG_SIZE} \
  --batch_size ${BATCH_SIZE} \
  --epochs ${EPOCHS} \
  --lr ${LR} \
  ${USE_LORA_FLAG}

# After training: sync outputs to S3 (optional)
aws s3 sync outputs/ s3://your-bucket/prs_med_outputs/
EOF

# === STEP 4: Clean up (optional) ===
echo "Training script executed. Remember to shut down your Lambda instance when done!"
