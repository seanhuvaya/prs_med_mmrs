"""
Configuration system for PRS-Med training.
Supports different modalities and training parameters.
"""

import json
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass, asdict
from .modality_config import MODALITY_MAPPINGS, get_datasets_by_modality, get_modality_stats

@dataclass
class ModalityConfig:
    """Configuration for a specific imaging modality."""
    name: str
    image_type: str
    img_size: int = 224
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    epochs: int = 50

@dataclass
class TrainingConfig:
    """Main training configuration."""
    # Data
    data_dir: str = "data/mmrs"
    modalities: List[str] = None
    
    # Model
    base_model: str = "microsoft/DialoGPT-medium"
    image_encoder: str = "vit_tiny_patch16_224"
    
    # Training
    batch_size: int = 8
    epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    
    # Loss weights
    lambda_seg: float = 1.0
    lambda_text: float = 1.0
    
    # Device
    device: str = "mps"
    
    # Output
    output_dir: str = "outputs"
    save_every: int = 10
    
    # Evaluation
    eval_every: int = 5
    
    def __post_init__(self):
        if self.modalities is None:
            # Use all available datasets from modality mappings
            self.modalities = list(MODALITY_MAPPINGS.keys())

# Predefined configurations for different scenarios
CONFIGS = {
    "single_modality": TrainingConfig(
        modalities=["brain_tumors_ct_scan"],
        epochs=30,
        batch_size=16
    ),
    
    "multi_modality": TrainingConfig(
        modalities=[
            "brain_tumors_ct_scan",
            "breast_tumors_ct_scan",
            "lung_CT"
        ],
        epochs=50,
        batch_size=8
    ),
    
    "all_modalities": TrainingConfig(
        epochs=100,
        batch_size=4,
        learning_rate=5e-5
    ),
    
    "fast_test": TrainingConfig(
        modalities=["brain_tumors_ct_scan"],
        epochs=5,
        batch_size=2,
        save_every=1,
        eval_every=1
    ),
    
    # Modality-specific configurations
    "ct_only": TrainingConfig(
        modalities=get_datasets_by_modality("CT"),
        epochs=50,
        batch_size=8,
        learning_rate=1e-4
    ),
    
    "xray_only": TrainingConfig(
        modalities=get_datasets_by_modality("X-ray"),
        epochs=40,
        batch_size=12,
        learning_rate=1e-4
    ),
    
    "endoscopy_only": TrainingConfig(
        modalities=get_datasets_by_modality("Endoscopy"),
        epochs=30,
        batch_size=16,
        learning_rate=1e-4
    ),
    
    "rgb_only": TrainingConfig(
        modalities=get_datasets_by_modality("RGB Image"),
        epochs=25,
        batch_size=20,
        learning_rate=1e-4
    )
}

def load_config(config_path: str) -> TrainingConfig:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    return TrainingConfig(**config_dict)

def save_config(config: TrainingConfig, config_path: str):
    """Save configuration to JSON file."""
    with open(config_path, 'w') as f:
        json.dump(asdict(config), f, indent=2)

def get_config(config_name: str = "multi_modality") -> TrainingConfig:
    """Get predefined configuration."""
    if config_name in CONFIGS:
        return CONFIGS[config_name]
    else:
        raise ValueError(f"Unknown config: {config_name}. Available: {list(CONFIGS.keys())}")

# Modality-specific configurations (deprecated - use MODALITY_MAPPINGS instead)
MODALITY_CONFIGS = {
    "brain_tumors_ct_scan": ModalityConfig(
        name="brain_tumors_ct_scan",
        image_type="Brain CT Scan",
        img_size=224,
        batch_size=8
    ),
    "breast_tumors_ct_scan": ModalityConfig(
        name="breast_tumors_ct_scan", 
        image_type="Breast CT Scan",
        img_size=224,
        batch_size=8
    ),
    "dental_xray": ModalityConfig(
        name="dental_xray",
        image_type="Dental X-ray",
        img_size=224,
        batch_size=16
    ),
    "lung_CT": ModalityConfig(
        name="lung_CT",
        image_type="Lung CT Scan", 
        img_size=224,
        batch_size=8
    ),
    "lung_Xray": ModalityConfig(
        name="lung_Xray",
        image_type="Lung X-ray",
        img_size=224,
        batch_size=16
    ),
    "polyp_endoscopy": ModalityConfig(
        name="polyp_endoscopy",
        image_type="Endoscopy",
        img_size=224,
        batch_size=16
    ),
    "skin_rgbimage": ModalityConfig(
        name="skin_rgbimage",
        image_type="Skin RGB Image",
        img_size=224,
        batch_size=16
    )
}
