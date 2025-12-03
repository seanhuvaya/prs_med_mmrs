#!/usr/bin/env bash

set -euo pipefail

BLUE='\033[0;34m'
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

usage() {
    cat <<EOF
Usage: $0 AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_REGION AWS_PROFILE S3_DATA_URI LOCAL_DATA_DIR CHECKPOINTS_DIR S3_CHECKPOINTS_URI

Arguments:
  1) AWS_ACCESS_KEY_ID
  2) AWS_SECRET_ACCESS_KEY
  3) AWS_REGION                (e.g. us-east-1)
  4) AWS_PROFILE               (e.g. prs-med)
  5) S3_DATA_URI               (e.g. s3://prs-med-dataset/Data/)
  6) LOCAL_DATA_DIR            (e.g. /workspace/data)
  7) CHECKPOINTS_DIR           (e.g. /workspace/checkpoints)
  8) S3_CHECKPOINTS_URI        (e.g. s3://prs-med-dataset/Checkpoints)
  9) VISION_ENCODER_TYPE       (e.g. sam_med2d or tinysam)
  10) VISION_ENCODER_CHECKPOINT (e.g. weights/sam2.1_hiera_tiny.pt or weights/tinysam_42.3.pth)
EOF
    exit 1
}

log() {
    printf "${BLUE}%s${NC}\n" "$1"
}

ok() {
    printf "${GREEN}%s${NC}\n" "$1"
}

err() {
    printf "${RED}%s${NC}\n" "$1" >&2
}

# ---- Argument parsing ----
if [ "$#" -ne 8 ]; then
    err "Error: Expected 8 arguments, got $#."
    usage
fi

AWS_ACCESS_KEY_ID="$1"
AWS_SECRET_ACCESS_KEY="$2"
AWS_DEFAULT_REGION="$3"
AWS_PROFILE="$4"
AWS_S3_BUCKET_DATA_URI="$5"
LOCAL_DATA_URI="$6"
CHECKPOINTS_DIR="$7"
AWS_S3_BUCKET_CHECKPOINTS_URI="$8"
VISION_ENCODER_TYPE="$9"
VISION_ENCODER_CHECKPOINT="$10"

REPO_URL="https://github.com/seanhuvaya/prs_med_mmrs.git"
REPO_DIR="/workspace/prs_med_mmrs"

# ---- Basic tool checks ----
for cmd in curl unzip git; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
        err "Required command '$cmd' not found in PATH."
        exit 1
    fi
done

if ! command -v uv >/dev/null 2>&1; then
    err "Required command 'uv' not found in PATH. Install uv before running this script."
    exit 1
fi

# ---- Install AWS CLI if needed ----
if ! command -v aws >/dev/null 2>&1; then
    log "AWS CLI not found. Installing AWS CLI v2..."
    curl -sSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
    unzip -q awscliv2.zip
    # Usually running as root in containers; no sudo needed
    ./aws/install
    rm -rf aws awscliv2.zip
    ok "AWS CLI installed."
else
    ok "AWS CLI already installed. Skipping installation."
fi

# ---- Configure AWS credentials ----
log "Setting credentials for profile: ${AWS_PROFILE}..."

aws configure set aws_access_key_id        "$AWS_ACCESS_KEY_ID"        --profile "$AWS_PROFILE"
aws configure set aws_secret_access_key    "$AWS_SECRET_ACCESS_KEY"    --profile "$AWS_PROFILE"
aws configure set region                   "$AWS_DEFAULT_REGION"       --profile "$AWS_PROFILE"
aws configure set output                   "json"                      --profile "$AWS_PROFILE"

ok "AWS credentials configured for profile '$AWS_PROFILE'."

# Ensure we use the configured profile by default in this script
export AWS_PROFILE

# ---- Download dataset from S3 ----
log "Downloading dataset from ${AWS_S3_BUCKET_DATA_URI} to ${LOCAL_DATA_URI} ..."
mkdir -p "$LOCAL_DATA_URI"
aws s3 sync "$AWS_S3_BUCKET_DATA_URI" "$LOCAL_DATA_URI"
ok "Dataset sync complete."

# ---- Clone or update git repo ----
if [ -d "$REPO_DIR/.git" ]; then
    log "Repository already exists at ${REPO_DIR}. Pulling latest changes..."
    git -C "$REPO_DIR" pull --rebase
else
    log "Cloning git repo into ${REPO_DIR}..."
    mkdir -p "$(dirname "$REPO_DIR")"
    git clone "$REPO_URL" "$REPO_DIR"
fi

cd "$REPO_DIR"

# ---- Sync dependencies with uv ----
log "Syncing dependencies with uv..."
uv sync
ok "Dependencies synced."

# ---- Train model ----
log "Training model..."
mkdir -p "$CHECKPOINTS_DIR"

echo "${BLUE}Downloading SAM2.1 Hiera Tiny checkpoint...${NC}"
wget -O "weights/sam2.1_hiera_tiny.pt" "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt" 


uv run python -m train_prs_med \
  --data_root "$LOCAL_DATA_URI" \
  --vision_encoder_type "$VISION_ENCODER_TYPE" \
  --vision_encoder_checkpoint "$VISION_ENCODER_CHECKPOINT" \
  --checkpoint_dir "$CHECKPOINTS_DIR" \

ok "Training finished."

# ---- Sync checkpoints back to S3 ----
log "Syncing checkpoints from ${CHECKPOINTS_DIR} to ${AWS_S3_BUCKET_CHECKPOINTS_URI} ..."
aws s3 sync "$CHECKPOINTS_DIR" "$AWS_S3_BUCKET_CHECKPOINTS_URI"
ok "Checkpoint sync complete."
