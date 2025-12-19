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

Options:
  -h, --help           Show this help message and exit

Example:
  $0 /workspace/data /workspace/prs_med_mmrs
EOF
}

# -----------------------------
# Parse args
# -----------------------------
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

if [[ $# -ne 2 ]]; then
    echo -e "${RED}Error: Expected 2 arguments.${NC}\n"
    usage
    exit 1
fi

DATA_DOWNLOAD_DIR="$1"
PROJECT_REPO_DIR="$2"

# -----------------------------
# Install system dependencies
# -----------------------------
echo -e "${BLUE}Updating system and installing unzip...${NC}"
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
# Download dataset
# -----------------------------
echo -e "${BLUE}Downloading dataset from public S3 bucket...${NC}"
mkdir -p "$DATA_DOWNLOAD_DIR"
aws s3 sync s3://prs-med-experiments/data/ "$DATA_DOWNLOAD_DIR" --no-sign-request

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


