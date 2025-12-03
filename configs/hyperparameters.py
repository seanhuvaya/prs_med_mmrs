"""
Hyperparameters configuration for PRS-Med training.

This file documents the hyperparameters used in the PRS-Med paper and provides
default values for training. Values marked with [PAPER] are from the paper,
others are reasonable defaults.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class PRSMedHyperparameters:
    """
    Hyperparameters for PRS-Med model training.
    
    Reference: PRS-Med paper (https://arxiv.org/pdf/2505.11872)
    """
    
    # ========== Model Architecture ==========
    # Vision Backbone (TinySAM or SAM-Med2D)
    vision_encoder_type: str = "tinysam"  # Options: "tinysam", "sam_med2d"
    vision_encoder_checkpoint: str = "weights/tinysam_42.3.pth"  # Path to encoder checkpoint
    tinysam_checkpoint: str = "weights/tinysam_42.3.pth"  # [Deprecated] Use vision_encoder_checkpoint instead
    image_size: int = 1024  # [PAPER] Image resolution
    
    # MLLM (LLaVA-Med)
    mllm_model_name: str = "chaoyinshe/llava-med-v1.5-mistral-7b-hf"
    freeze_llm: bool = True  # [PAPER] Freeze base LLM, only train LoRA
    
    # LoRA Configuration
    lora_rank: int = 16  # [PAPER] LoRA rank
    lora_alpha: int = 16  # [PAPER] LoRA alpha (scaling factor)
    lora_dropout: float = 0.05  # [PAPER] LoRA dropout rate
    
    # ========== Training Hyperparameters ==========
    batch_size: int = 8  # [PAPER] Batch size
    learning_rate: float = 1e-4  # [PAPER] Learning rate
    num_epochs: int = 20  # Training epochs (adjust based on convergence)
    
    # Optimizer (AdamW)
    optimizer: str = "AdamW"
    weight_decay: float = 0.01  # [PAPER] Weight decay for AdamW
    betas: tuple = (0.9, 0.999)  # AdamW betas
    
    # Learning Rate Schedule
    lr_scheduler: Optional[str] = None  # Optional: "cosine", "step", etc.
    warmup_epochs: int = 0  # Warmup epochs
    
    # Gradient Clipping
    max_grad_norm: float = 1.0  # [PAPER] Gradient clipping norm
    
    # ========== Loss Function ==========
    lambda_seg: float = 1.0  # [PAPER] Segmentation loss weight
    lambda_txt: float = 0.5  # [PAPER] Text reasoning loss weight
    
    # Segmentation Loss Components
    dice_smooth: float = 1e-6  # Smoothing factor for Dice loss
    bce_weight: float = 1.0  # BCE loss weight (combined with Dice)
    
    # ========== Data Loading ==========
    num_workers: int = 2  # DataLoader workers
    pin_memory: bool = True  # Pin memory for faster GPU transfer
    
    # ========== Reproducibility ==========
    seed: int = 42  # Random seed for reproducibility
    deterministic: bool = False  # Fully deterministic (slower but reproducible)
    
    # ========== Evaluation ==========
    eval_interval: int = 1  # Evaluate every N epochs
    save_interval: int = 5  # Save checkpoint every N epochs
    
    # ========== Device ==========
    device: Optional[str] = None  # Auto-detect if None
    
    def to_dict(self):
        """Convert to dictionary for logging/saving."""
        return {
            "model": {
                "vision_encoder_type": self.vision_encoder_type,
                "vision_encoder_checkpoint": self.vision_encoder_checkpoint,
                "tinysam_checkpoint": self.tinysam_checkpoint,  # Deprecated, kept for compatibility
                "image_size": self.image_size,
                "mllm_model_name": self.mllm_model_name,
                "freeze_llm": self.freeze_llm,
                "lora_rank": self.lora_rank,
                "lora_alpha": self.lora_alpha,
                "lora_dropout": self.lora_dropout,
            },
            "training": {
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
                "num_epochs": self.num_epochs,
                "optimizer": self.optimizer,
                "weight_decay": self.weight_decay,
                "betas": self.betas,
                "max_grad_norm": self.max_grad_norm,
            },
            "loss": {
                "lambda_seg": self.lambda_seg,
                "lambda_txt": self.lambda_txt,
                "dice_smooth": self.dice_smooth,
            },
            "reproducibility": {
                "seed": self.seed,
                "deterministic": self.deterministic,
            }
        }


# Predefined configurations
PAPER_CONFIG = PRSMedHyperparameters(
    # Paper values (as documented)
    batch_size=8,
    learning_rate=1e-4,
    lambda_seg=1.0,
    lambda_txt=0.5,
    lora_rank=16,
    lora_alpha=16,
    lora_dropout=0.05,
    max_grad_norm=1.0,
    weight_decay=0.01,
    seed=42,
)

FAST_TEST_CONFIG = PRSMedHyperparameters(
    # Quick test configuration
    batch_size=2,
    num_epochs=5,
    num_workers=0,  # Faster for small tests
)

LARGE_BATCH_CONFIG = PRSMedHyperparameters(
    # For GPUs with more memory
    batch_size=16,
    learning_rate=2e-4,  # Scale LR with batch size
)


def get_config(config_name: str = "paper") -> PRSMedHyperparameters:
    """
    Get a predefined configuration.
    
    Args:
        config_name: One of "paper", "fast_test", "large_batch"
    
    Returns:
        PRSMedHyperparameters instance
    """
    configs = {
        "paper": PAPER_CONFIG,
        "fast_test": FAST_TEST_CONFIG,
        "large_batch": LARGE_BATCH_CONFIG,
    }
    
    if config_name not in configs:
        raise ValueError(f"Unknown config: {config_name}. Available: {list(configs.keys())}")
    
    return configs[config_name]


if __name__ == "__main__":
    # Print paper configuration
    config = PAPER_CONFIG
    print("PRS-Med Paper Hyperparameters:")
    print("=" * 50)
    for key, value in config.to_dict().items():
        print(f"\n{key.upper()}:")
        for k, v in value.items():
            print(f"  {k}: {v}")

