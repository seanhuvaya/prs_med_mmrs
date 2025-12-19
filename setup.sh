#!/usr/bin/env bash
set -euo pipefail

# -----------------------------
# Colors
# -----------------------------
BLUE='\033[0;34m'
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

# -----------------------------
# Usage
# -----------------------------
usage() {
    cat <<EOF
Usage: $0 [OPTIONS] <DATA_DOWNLOAD_DIR> <PROJECT_REPO_DIR>

Set up the PRS-Med project by downloading data, cloning the repository,
and installing dependencies using uv.

Arguments:
  DATA_DOWNLOAD_DIR    Directory where the dataset will be downloaded
  PROJECT_REPO_DIR     Directory where the project repository will be cloned
  WEIGHTS_DIR          Directory where the weights will be downloaded
Options:
  -h, --help           Show this help message and exit

Example:
  $0 /workspace/data /workspace/prs_med_mmrs /workspace/weights
EOF
}

# -----------------------------
# Parse args
# -----------------------------
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

if [[ $# -ne 3 ]]; then
    echo -e "${RED}Error: Expected 3 arguments.${NC}\n"
    usage
    exit 1
fi

DATA_DOWNLOAD_DIR="$1"
PROJECT_REPO_DIR="$2"
WEIGHTS_DIR="$3"
SAM_MED2D_WEIGHT="sam2.1_hiera_large.pt"

# -----------------------------
# Install system dependencies
# -----------------------------
echo -e "${BLUE}Updating system and installing system dependencies...${NC}"
apt update
apt install -y unzip curl git tmux

# -----------------------------
# Install AWS CLI (if missing)
# -----------------------------
if ! command -v aws >/dev/null 2>&1; then
    echo -e "${BLUE}AWS CLI not found. Installing...${NC}"
    curl -s "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
    unzip -q awscliv2.zip
    ./aws/install
else
    echo -e "${GREEN}AWS CLI already installed. Skipping.${NC}"
fi

# -----------------------------
# Download dataset (public S3)
# -----------------------------
echo -e "${BLUE}Downloading dataset from public S3 bucket...${NC}"
mkdir -p "$DATA_DOWNLOAD_DIR"
aws s3 sync s3://prs-med-experiments/data/ "$DATA_DOWNLOAD_DIR" --no-sign-request

# -----------------------------
# Download SAM-Med2D weights (public S3)
# -----------------------------
echo -e "${BLUE}Downloading SAM-Med2D weights...${NC}"
mkdir -p "$WEIGHTS_DIR"

if [[ ! -f "$WEIGHTS_DIR/$SAM_MED2D_WEIGHT" ]]; then
    aws s3 cp \
        s3://prs-med-experiments/weights/$SAM_MED2D_WEIGHT \
        "$WEIGHTS_DIR/$SAM_MED2D_WEIGHT" \
        --no-sign-request
    echo -e "${GREEN}Weights downloaded successfully.${NC}"
else
    echo -e "${GREEN}Weights already exist. Skipping download.${NC}"
fi

# -----------------------------
# Clone repository
# -----------------------------
echo -e "${BLUE}Cloning project repository...${NC}"
git clone https://github.com/seanhuvaya/prs_med_mmrs.git "$PROJECT_REPO_DIR"

# -----------------------------
# Install uv
# -----------------------------
echo -e "${BLUE}Installing uv...${NC}"
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"

# -----------------------------
# Sync dependencies
# -----------------------------
echo -e "${BLUE}Syncing project dependencies...${NC}"
cd "$PROJECT_REPO_DIR"
uv sync

echo -e "${GREEN}Setup completed successfully!${NC}"
