#!/usr/bin/env bash

set -euo pipefail

DATA_DOWNLOAD_DIR="$1"
PROJECT_REPO_DIR="$2"

apt update
apt install unzip

curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
./aws/install

aws s3 sync s3://prs-med-experiments/data/ "$DATA_DOWNLOAD_DIR"

git clone https://github.com/seanhuvaya/prs_med_mmrs.git "$PROJECT_REPO_DIR"

curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"

cd "$PROJECT_REPO_DIR"
uv sync


