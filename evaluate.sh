set -euo pipefail

BLUE='\033[0;34m'
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

usage() {
    cat <<EOF
Usage: $0 AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_REGION AWS_PROFILE S3_DATA_URI LOCAL_DATA_DIR CHECKPOINTS_DIR S3_CHECKPOINTS_URI VISION_ENCODER_TYPE VISION_ENCODER_CHECKPOINT

Arguments:
  1) AWS_ACCESS_KEY_ID
  2) AWS_SECRET_ACCESS_KEY
  3) AWS_REGION                 (e.g. us-east-1)
  4) AWS_PROFILE                (e.g. prs-med)
  5) S3_DATA_URI                (e.g. s3://prs-med-dataset/Data/)
  6) LOCAL_DATA_DIR             (e.g. /workspace/data)
  7) AWS_S3_BUCKET_CHECKPOINT_URI            (e.g. /workspace/checkpoints)
  8) LOCAL_CHECKPOINT_URI         (e.g. s3://prs-med-dataset/Checkpoints)
  9) VISION_ENCODER_TYPE        (e.g. sam_med2d or tinysam)
  10) VISION_ENCODER_CHECKPOINT (e.g. weights/sam2.1_hiera_tiny.pt or weights/tinysam_42.3.pth)
  11) DATASET_NAME (e.g prostate, head_and_neck e.t.c)
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
#if [ "$#" -ne 11 ]; then
#    err "Error: Expected 10 arguments, got $#."
#    usage
#fi

AWS_ACCESS_KEY_ID="$1"
AWS_SECRET_ACCESS_KEY="$2"
AWS_DEFAULT_REGION="$3"
AWS_PROFILE="$4"
AWS_S3_BUCKET_DATA_URI="$5"
LOCAL_DATA_URI="$6"
AWS_S3_BUCKET_CHECKPOINT_URI="$7"
LOCAL_CHECKPOINT_URI="$8"
VISION_ENCODER_TYPE="$9"
VISION_ENCODER_CHECKPOINT="${10}"
DATASET_NAME="${11}"
LOCAL_OUTPUT_DIR="${12}"


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
    curl -fsSL --retry 5 --retry-delay 2 \
        "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" \
        -o "awscliv2.zip"
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

aws configure set aws_access_key_id     "$AWS_ACCESS_KEY_ID"     --profile "$AWS_PROFILE"
aws configure set aws_secret_access_key "$AWS_SECRET_ACCESS_KEY" --profile "$AWS_PROFILE"
aws configure set region                "$AWS_DEFAULT_REGION"    --profile "$AWS_PROFILE"
aws configure set output                "json"                   --profile "$AWS_PROFILE"

ok "AWS credentials configured for profile '$AWS_PROFILE'."

# Ensure we use the configured profile by default in this script
export AWS_PROFILE

# ---- Download dataset from S3 ----
log "Downloading dataset from ${AWS_S3_BUCKET_DATA_URI} to ${LOCAL_DATA_URI} ..."
mkdir -p "$LOCAL_DATA_URI"
aws s3 sync "$AWS_S3_BUCKET_DATA_URI" "$LOCAL_DATA_URI"
ok "Dataset sync complete."

# ---- Download model checkpoint from s3 ---
log "Downloading model checkpoint from ${AWS_S3_BUCKET_CHECKPOINT_URI} to ${LOCAL_CHECKPOINT_URI}"

if [ ! -d "/workspace/checkpoints" ]; then
    mkdir -p "/workspace/checkpoints"
    echo "Directory '/workspace/checkpoints' created."
else
    echo "Directory '/workspace/checkpoints' already exists."
fi

if [ ! -d "$LOCAL_OUTPUT_DIR" ]; then
  mkdir -p "$LOCAL_OUTPUT_DIR"
  echo "Directory '$LOCAL_OUTPUT_DIR' created"
fi

aws s3 cp "$AWS_S3_BUCKET_CHECKPOINT_URI" "$LOCAL_CHECKPOINT_URI"

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

# ---- Ensure vision encoder checkpoint exists (download if needed) ----
# Only auto-download if using SAM-Med2D and the file does not already exist.
if [ "$VISION_ENCODER_TYPE" = "sam_med2d" ] && [ ! -f "$VISION_ENCODER_CHECKPOINT" ]; then
    log "VISION_ENCODER_TYPE is 'sam_med2d' and checkpoint '$VISION_ENCODER_CHECKPOINT' not found."
    log "Downloading SAM2.1 Hiera Tiny checkpoint..."

    mkdir -p "$(dirname "$VISION_ENCODER_CHECKPOINT")"

    curl -fsSL --retry 5 --retry-delay 2 \
        -o "$VISION_ENCODER_CHECKPOINT" \
        "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt"

    ok "SAM2.1 Hiera Tiny checkpoint downloaded to $VISION_ENCODER_CHECKPOINT."
fi

log "Evaluating model..."
uv run python -m evaluation.benchmark_prs_med \
    --checkpoint "$LOCAL_CHECKPOINT_URI" \
    --data_root "$LOCAL_DATA_URI" \
    --output_dir "$LOCAL_OUTPUT_DIR" \
    --split "test" \
    --vision_encoder_type "$VISION_ENCODER_TYPE" \
    --vision_encoder_checkpoint "$VISION_ENCODER_CHECKPOINT" \
    --specific_dataset "$DATASET_NAME"


ok "Evaluation finished."