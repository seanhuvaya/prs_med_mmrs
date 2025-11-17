#!/bin/bash
# Cloud Platform Setup Script for PRS-Med Training
# Run this script on your cloud GPU instance to set up the environment

set -e  # Exit on error

echo "🚀 Setting up PRS-Med training environment on cloud instance..."

# Check for GPU
if command -v nvidia-smi &> /dev/null; then
    echo "✅ GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "⚠️  Warning: nvidia-smi not found. GPU may not be available."
fi

# Update system packages
echo "📦 Updating system packages..."
sudo apt-get update -qq
sudo apt-get install -y python3.11 python3.11-venv python3-pip git curl

# Install uv (if not already installed)
if ! command -v uv &> /dev/null; then
    echo "📥 Installing uv package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

# Clone repository (if not already present)
if [ ! -d "prs_med_mmrs" ]; then
    echo "📥 Cloning repository..."
    # Replace with your actual repo URL
    # git clone <your-repo-url> prs_med_mmrs
    echo "⚠️  Please clone your repository manually or update this script with your repo URL"
fi

cd prs_med_mmrs || exit

# Create virtual environment
echo "🐍 Creating Python virtual environment..."
python3.11 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install dependencies using uv or pip
if command -v uv &> /dev/null; then
    echo "📦 Installing dependencies with uv..."
    uv pip install -e .
else
    echo "📦 Installing dependencies with pip..."
    pip install torch torchvision transformers peft pandas scipy timm
    pip install python-dotenv
fi

# Verify PyTorch CUDA availability
echo "🔍 Verifying PyTorch CUDA setup..."
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')
else:
    print('⚠️  CUDA not available - training will be slow on CPU')
"

# Create directories
echo "📁 Creating necessary directories..."
mkdir -p checkpoints
mkdir -p weights
mkdir -p data

# Check if TinySAM weights exist
if [ ! -f "weights/tinysam_42.3.pth" ]; then
    echo "⚠️  Warning: TinySAM weights not found at weights/tinysam_42.3.pth"
    echo "   Please download from: https://github.com/xinghaochen/TinySAM"
fi

# Create a simple test script
cat > test_setup.py << 'EOF'
"""Quick test to verify setup"""
import torch
from models.vision_backbone.tiny_sam_encoder import TinySAMVisionBackbone
from models.mllm.llava_med_lora_adapter import LLavaMedWithLoRA

print("Testing model components...")

# Test GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Test vision backbone (if weights available)
try:
    vision = TinySAMVisionBackbone(
        checkpoint_path="weights/tinysam_42.3.pth",
        image_size=1024,
        device=str(device)
    )
    print("✅ Vision backbone loaded successfully")
except Exception as e:
    print(f"⚠️  Vision backbone test failed: {e}")

# Test MLLM
try:
    mllm = LLavaMedWithLoRA(
        rank=16,
        alpha=16,
        dropout=0.05,
        freeze_llm=True,
        device=str(device)
    )
    print("✅ MLLM loaded successfully")
except Exception as e:
    print(f"⚠️  MLLM test failed: {e}")

print("\n✅ Setup test complete!")
EOF

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Test setup: python test_setup.py"
echo "3. Prepare your data in the 'data' directory"
echo "4. Start training: python train_prs_med.py --data_root data --seed 42"
echo ""
echo "💡 Tip: Use 'screen' or 'tmux' to keep training running after SSH disconnect:"
echo "   screen -S training"
echo "   # Run your training command"
echo "   # Press Ctrl+A then D to detach"
echo "   # Reattach with: screen -r training"

