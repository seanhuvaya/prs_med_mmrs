# Cloud Platform Recommendations for PRS-Med Training

## Hardware Requirements

Based on the PRS-Med model architecture:

- **Model**: LLaVA-Med (Mistral-7B) + TinySAM + Fusion + Decoder
- **Training Method**: LoRA fine-tuning (memory efficient)
- **Batch Size**: 8 (from paper)
- **Image Size**: 1024×1024
- **Minimum VRAM**: 16GB (with batch size 4-6, mixed precision)
- **Recommended VRAM**: 24GB+ (for batch size 8, comfortable training)
- **GPU Type**: Modern GPU with good compute (A100, V100, RTX 3090/4090, A6000, etc.)

## 💰 Recommended Cloud Platforms (Ranked by Cost-Effectiveness)

### 1. **Vast.ai** ⭐ Best Value for Money
**Best for**: Cost-conscious users, flexible scheduling

**Pros:**
- ✅ **Cheapest option** - Often 50-70% cheaper than major cloud providers
- ✅ Wide variety of GPUs (RTX 3090, RTX 4090, A100, V100)
- ✅ Pay-per-hour with no minimum commitment
- ✅ Easy setup with pre-configured Docker images
- ✅ Community marketplace with competitive pricing

**Cons:**
- ⚠️ Less reliable than major providers (hosted on user machines)
- ⚠️ May need to restart if host goes offline
- ⚠️ Less customer support

**Pricing Examples:**
- RTX 3090 (24GB): ~$0.30-0.50/hour
- RTX 4090 (24GB): ~$0.50-0.80/hour
- A100 (40GB): ~$1.00-1.50/hour

**Setup:**
```bash
# 1. Sign up at vast.ai
# 2. Search for GPU (filter: 24GB+ VRAM, CUDA available)
# 3. Create instance with PyTorch Docker image
# 4. SSH into instance and clone your repo
```

**Estimated Cost for 20 Epochs**: $15-30 (depending on dataset size)

---

### 2. **RunPod** ⭐ Great Balance
**Best for**: Reliable training with good pricing

**Pros:**
- ✅ **Good pricing** - Competitive with Vast.ai but more reliable
- ✅ Pre-configured PyTorch templates
- ✅ Persistent storage available
- ✅ Community templates for ML workflows
- ✅ Better uptime than Vast.ai

**Cons:**
- ⚠️ Slightly more expensive than Vast.ai
- ⚠️ Smaller GPU selection than Vast.ai

**Pricing Examples:**
- RTX 3090 (24GB): ~$0.40-0.60/hour
- RTX 4090 (24GB): ~$0.60-0.90/hour
- A100 (40GB): ~$1.20-1.80/hour

**Setup:**
```bash
# 1. Sign up at runpod.io
# 2. Create Pod with PyTorch template
# 3. Attach network volume for data
# 4. SSH and start training
```

**Estimated Cost for 20 Epochs**: $20-40

---

### 3. **Lambda Labs** ⭐ Developer-Friendly
**Best for**: Easy setup, good documentation

**Pros:**
- ✅ **Excellent documentation** and tutorials
- ✅ Pre-configured ML environments
- ✅ Good for beginners
- ✅ Reliable infrastructure
- ✅ Free credits for new users ($10-50)

**Cons:**
- ⚠️ More expensive than Vast.ai/RunPod
- ⚠️ Limited GPU types

**Pricing Examples:**
- RTX 3090 (24GB): ~$0.50/hour
- RTX 6000 Ada (48GB): ~$1.10/hour
- A100 (40GB): ~$1.50/hour

**Setup:**
```bash
# 1. Sign up at lambdalabs.com
# 2. Launch instance (choose PyTorch template)
# 3. Use their CLI or web interface
```

**Estimated Cost for 20 Epochs**: $30-50

---

### 4. **Google Colab Pro/Pro+** ⭐ Convenient but Limited
**Best for**: Quick experiments, small datasets

**Pros:**
- ✅ Very easy to use (Jupyter notebook interface)
- ✅ Free tier available (limited)
- ✅ No setup required
- ✅ Good for testing

**Cons:**
- ⚠️ **Limited VRAM** (Pro: 16GB, Pro+: 24GB)
- ⚠️ Session timeouts (90 min - 24 hours)
- ⚠️ Not ideal for long training runs
- ⚠️ Can't guarantee GPU availability

**Pricing:**
- Colab Pro: $10/month (16GB VRAM, may not be enough)
- Colab Pro+: $50/month (24GB VRAM, better but still limited)

**Best for**: Testing code, not full training

---

### 5. **AWS/GCP/Azure** ⭐ Enterprise-Grade
**Best for**: Production workloads, enterprise needs

**Pros:**
- ✅ Most reliable and scalable
- ✅ Excellent documentation
- ✅ Enterprise support
- ✅ Spot instances for cost savings

**Cons:**
- ⚠️ **Most expensive** option
- ⚠️ Complex setup
- ⚠️ Need to manage infrastructure

**Pricing Examples (Spot Instances):**
- p3.2xlarge (V100 16GB): ~$0.50-0.80/hour
- p4d.24xlarge (A100 40GB): ~$2.00-3.00/hour
- g4dn.xlarge (T4 16GB): ~$0.30-0.50/hour

**Estimated Cost for 20 Epochs**: $50-150 (with spot instances)

---

## 🎯 Recommended Setup Strategy

### For Budget-Conscious Users:
1. **Start with Vast.ai** - Test your code on a cheap RTX 3090
2. **Use RunPod** if you need more reliability
3. **Monitor costs** - Stop instances when not training

### For Reliability:
1. **RunPod** or **Lambda Labs** - Better uptime than Vast.ai
2. **AWS Spot Instances** - If you need enterprise reliability

### For Quick Testing:
1. **Google Colab Pro+** - Test code quickly
2. **Vast.ai** - For actual training

---

## 💡 Cost Optimization Tips

1. **Use Spot/Preemptible Instances**: 50-70% cheaper
2. **Reduce Batch Size**: If VRAM is tight, use batch_size=4 instead of 8
3. **Use Mixed Precision**: Enable FP16 to reduce memory usage
4. **Monitor Training**: Stop early if model converges
5. **Use Gradient Checkpointing**: Trade compute for memory
6. **Schedule Training**: Train during off-peak hours (if available)

---

## 🔧 Recommended Instance Configuration

### Minimum (Budget):
- **GPU**: RTX 3090 (24GB) or RTX 4090 (24GB)
- **CPU**: 4-8 cores
- **RAM**: 32GB
- **Storage**: 100GB SSD
- **Cost**: ~$0.30-0.60/hour
- **Batch Size**: 4-6 (may need to reduce from 8)

### Recommended:
- **GPU**: A100 (40GB) or RTX 6000 Ada (48GB)
- **CPU**: 8-16 cores
- **RAM**: 64GB
- **Storage**: 200GB SSD
- **Cost**: ~$1.00-1.50/hour
- **Batch Size**: 8 (as per paper)

---

## 📋 Setup Checklist

Before starting training on cloud:

1. ✅ **Test locally first** - Ensure code runs without errors
2. ✅ **Prepare data** - Upload dataset to cloud storage or instance
3. ✅ **Create requirements.txt** - List all dependencies
4. ✅ **Set up monitoring** - Use wandb/tensorboard for remote monitoring
5. ✅ **Configure checkpoints** - Save frequently to avoid losing progress
6. ✅ **Set up SSH** - For easy access to instance
7. ✅ **Enable auto-shutdown** - Stop instance when training completes

---

## 🚀 Quick Start Script for Cloud Training

Create a setup script for your cloud instance:

```bash
#!/bin/bash
# setup_cloud.sh

# Update system
sudo apt-get update

# Install Python 3.11
sudo apt-get install -y python3.11 python3.11-venv python3-pip

# Install CUDA toolkit (if not pre-installed)
# Check: nvidia-smi

# Clone your repository
git clone <your-repo-url>
cd prs_med_mmrs

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt  # Create this from pyproject.toml

# Or use uv
curl -LsSf https://astral.sh/uv/install.sh | sh
uv pip install -e .

# Download model weights
# (TinySAM checkpoint, etc.)

# Start training
python train_prs_med.py \
    --data_root /path/to/data \
    --seed 42 \
    --batch_size 8 \
    --learning_rate 1e-4 \
    --num_epochs 20
```

---

## 📊 Cost Comparison Table

| Platform | GPU | VRAM | Price/Hour | Est. 20 Epochs | Reliability |
|----------|-----|------|------------|----------------|-------------|
| **Vast.ai** | RTX 3090 | 24GB | $0.30-0.50 | $15-30 | ⭐⭐⭐ |
| **RunPod** | RTX 3090 | 24GB | $0.40-0.60 | $20-40 | ⭐⭐⭐⭐ |
| **Lambda Labs** | RTX 3090 | 24GB | $0.50 | $30-50 | ⭐⭐⭐⭐ |
| **Vast.ai** | A100 | 40GB | $1.00-1.50 | $50-75 | ⭐⭐⭐ |
| **AWS Spot** | A100 | 40GB | $2.00-3.00 | $100-150 | ⭐⭐⭐⭐⭐ |
| **Colab Pro+** | T4/V100 | 16-24GB | $50/mo | $50/month | ⭐⭐ |

*Note: Costs vary based on availability, region, and time of day*

---

## 🎓 Learning Resources

- **Vast.ai Docs**: https://docs.vast.ai/
- **RunPod Docs**: https://docs.runpod.io/
- **Lambda Labs Guide**: https://lambdalabs.com/service/gpu-cloud
- **AWS EC2 GPU Guide**: https://aws.amazon.com/ec2/instance-types/

---

## ⚠️ Important Notes

1. **Always backup checkpoints** - Cloud instances can be terminated
2. **Monitor costs** - Set up billing alerts
3. **Test with small dataset first** - Validate setup before full training
4. **Use persistent storage** - For data and checkpoints
5. **Enable auto-shutdown** - Stop instance when idle to save costs

---

## 🏆 Top Recommendation

**For most users**: Start with **Vast.ai** or **RunPod** with an RTX 3090/4090 (24GB). This provides the best balance of cost, performance, and reliability for training PRS-Med.

**For production**: Use **AWS/GCP/Azure** with spot instances for maximum reliability and scalability.

