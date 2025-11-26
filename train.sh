# Install awscli
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# Sync data from S3
aws s3 sync s3://prs-med-dataset/new_data/ /workspace/new_data

# Clone repository
git clone https://github.com/seanhuvaya/prs_med_mmrs.git /workspace/prs_med_mmrs

# Sync dependencies
cd /workspace/prs_med_mmrs/
uv sync

# Train the model
git checkout v3 # TODO: remove this
uv run python -m train --data-dir /workspace/new_data --ckptt-dir /workspace/checkpoints


# TODO: add evaluation