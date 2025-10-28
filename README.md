# PRS-Med: Position Reasoning Segmentation with Vision-Language Model

This is a complete implementation of the [PRS-Med paper](https://arxiv.org/pdf/2505.11872) for Position Reasoning Segmentation in Medical Imaging.

## 🚀 **Quick Start**

### 1. **Prepare Data**
```bash
# Organize raw data into MMRS format
uv run python main.py prepare
```

### 2. **Train Model**
```bash
# Train on all modalities
uv run python main.py train --epochs 50 --batch_size 8

# Train on specific modalities only
uv run python main.py train --modalities brain_tumors_ct_scan lung_CT --epochs 30

# Fast test run
uv run python main.py train --epochs 5 --batch_size 2
```

### 3. **Run Inference**
```bash
# Run inference on test data
uv run python main.py infer --checkpoint outputs/final_model.pth
```

## 📁 **Project Structure**

```
prs_med_mmrs/
├── models/                 # Core model components
│   ├── image_encoder.py   # TinyVisionEncoder (TinySAM-based)
│   ├── text_encoder.py    # MultimodalTextEncoder (DialoGPT + LoRA)
│   ├── fusion_module.py   # CrossAttentionFusion
│   ├── mask_decoder.py    # MaskDecoder (with interpolation)
│   └── prs_med_model.py   # Main PRSMedModel
├── data_pipeline/         # Data processing
│   ├── dataset_mmrs.py    # MMRSDataset with position reasoning
│   ├── transforms.py      # Image preprocessing
│   └── templates/         # Question-answer templates
├── training/              # Training components
│   ├── train_loop.py      # Trainer class
│   ├── losses.py          # Segmentation + Text losses
│   ├── optimizer.py      # Optimizer configuration
│   └── evaluation.py      # Evaluation metrics
├── configs/               # Configuration system
│   └── training_config.py # Training configurations
├── scripts/               # Utility scripts
│   ├── prepare_data.py    # Data preprocessing
│   ├── train.py          # Training script
│   ├── infer.py          # Inference script
│   └── visualize.py       # Visualization utilities
├── tests/                 # Test suite
└── data/                  # Data directory
    ├── raw/              # Raw datasets (6 modalities)
    └── mmrs/             # Processed MMRS format
```

## 🏥 **Supported Modalities**

The implementation supports all 6 modalities from the paper:

1. **Brain Tumors CT Scan** (`brain_tumors_ct_scan`)
2. **Breast Tumors CT Scan** (`breast_tumors_ct_scan`) 
3. **Dental X-ray** (`dental_xray`)
4. **Lung CT** (`lung_CT`)
5. **Lung X-ray** (`lung_Xray`)
6. **Polyp Endoscopy** (`polyp_endoscopy`)
7. **Skin RGB Image** (`skin_rgbimage`)

## 🧠 **Model Architecture**

### **PRS-Med Components:**

1. **TinyVisionEncoder**: Lightweight image encoder based on TinySAM
2. **MultimodalTextEncoder**: Text encoder with LoRA adaptation
3. **CrossAttentionFusion**: Fuses visual and textual features
4. **MaskDecoder**: Generates segmentation masks with position reasoning

### **Key Features:**
- ✅ **Position Reasoning**: Understands spatial relationships
- ✅ **Multi-modal**: Works across 6+ imaging modalities  
- ✅ **Template-based QA**: Uses 50+ question-answer templates
- ✅ **LoRA Adaptation**: Efficient fine-tuning
- ✅ **Comprehensive Evaluation**: Dice, IoU, Hausdorff, Position Accuracy

## 📊 **Training Configuration**

### **Available Configurations:**
- `single_modality`: Train on one modality (fast)
- `multi_modality`: Train on 3 modalities (balanced)
- `all_modalities`: Train on all 6 modalities (full)
- `fast_test`: Quick test run (5 epochs)

### **Training Parameters:**
```python
# Default training config
batch_size = 8
epochs = 50
learning_rate = 1e-4
lambda_seg = 1.0      # Segmentation loss weight
lambda_text = 1.0     # Text loss weight
device = "mps"        # or "cuda" or "cpu"
```

## 🎯 **Position Reasoning**

The model generates position descriptions like:
- `"top-left"`, `"top-right"`, `"bottom-left"`, `"bottom-right"`
- `"near-center"` (within threshold distance)
- Contextual descriptions based on image type

### **Question Templates:**
- "Where is the lesion located in this {image_type}?"
- "What is the anatomical position of the tumour?"
- "Can you identify the tumour's location?"

## 📈 **Evaluation Metrics**

### **Segmentation Metrics:**
- **Dice Coefficient**: Overlap between predicted and ground truth masks
- **IoU Score**: Intersection over Union
- **Hausdorff Distance**: Boundary accuracy

### **Position Reasoning Metrics:**
- **Exact Match**: Perfect position description match
- **Keyword Match**: Position keyword accuracy

## 🔧 **Advanced Usage**

### **Custom Training:**
```python
from configs.training_config import get_config, TrainingConfig

# Get predefined config
config = get_config("multi_modality")

# Or create custom config
custom_config = TrainingConfig(
    modalities=["brain_tumors_ct_scan", "lung_CT"],
    epochs=100,
    batch_size=4,
    learning_rate=5e-5
)
```

### **Data Pipeline:**
```python
from data_pipeline.dataset_mmrs import MMRSDataset

# Create dataset
dataset = MMRSDataset(
    root="data/mmrs/brain_tumors_ct_scan/train",
    split="train",
    img_size=224,
    tokenizer=tokenizer
)

# Get sample
sample = dataset[0]
print(sample["question"])  # "Where is the lesion located in this Brain Ct Scan?"
print(sample["answer"])    # "The tumour is located in the top-left region."
```

## 🧪 **Testing**

Run the complete test suite:
```bash
uv run python -m pytest tests/ -v
```

Tests cover:
- ✅ Model components (image encoder, fusion, mask decoder)
- ✅ Training pipeline (losses, optimizer, trainer)
- ✅ Data pipeline (centroid computation, position labeling)

## 📚 **Paper Implementation Status**

### **✅ Completed:**
- [x] Core model architecture (TinyVisionEncoder + MultimodalTextEncoder)
- [x] Cross-attention fusion module
- [x] Mask decoder with interpolation
- [x] Position reasoning dataset (MMRS)
- [x] Template-based question generation
- [x] Multi-modal training pipeline
- [x] Comprehensive evaluation metrics
- [x] Configuration system

### **🔄 Remaining (Optional Enhancements):**
- [ ] Advanced visualization tools
- [ ] Model ensemble support
- [ ] Hyperparameter optimization
- [ ] Cross-validation pipeline
- [ ] Model compression/quantization

## 🚀 **Next Steps**

1. **Prepare your data**: Run `uv run python main.py prepare`
2. **Start training**: Run `uv run python main.py train`
3. **Monitor progress**: Check outputs in `outputs/` directory
4. **Evaluate results**: Use evaluation metrics in `training/evaluation.py`

## 📖 **References**

- [PRS-Med Paper](https://arxiv.org/pdf/2505.11872): Position Reasoning Segmentation with Vision-Language Model in Medical Imaging
- **Datasets**: BUSI, LungCT, LungXray, BrainMRI, Kvasir-SEG, ClinicDB, CVC300, ColonDB, ETIS-Polyric
- **Base Models**: TinySAM, DialoGPT, LoRA

---

**Ready to train your PRS-Med model!** 🎉