import argparse
from dataclasses import dataclass

@dataclass
class PRSMedConfig:
    # Paths
    data_dir: str
    ckpt_dir: str = "../checkpoints"
    tinysam_ckpt: str = "weights/tinysam_42.3.pth"

    # LoRA
    lora_rank: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.05

    # Loss weights
    lambda_seg: float = 1.0
    lambda_txt: float = 0.5

    # Training params
    batch_size: int = 8
    learning_rate: float = 1e-4
    num_epochs: int = 20
    image_size: int = 1024
    num_workers: int = 2

    # Misc
    seed: int = 42
    deterministic: bool = False

    # AMP
    amp: bool = True

def get_config() -> PRSMedConfig:
    parser = argparse.ArgumentParser(description="PRS-Med Training")

    # Paths
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--ckpt-dir", type=str, default="./checkpoints")
    parser.add_argument("--tinysam-ckpt", type=str, default="weights/tinysam_42.3.pth", help='Path to TinySAM checkpoint')

    # LoRA
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)

    # Loss weights
    parser.add_argument("--lambda-seg", type=float, default=1.0)
    parser.add_argument("--lambda-txt", type=float, default=0.5)

    # Training params
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-epochs", type=int, default=20)
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=2)

    # Misc
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--deterministic", action="store_true", help="Enable fully deterministic training (may be slower)")

    # AMP flags
    parser.add_argument("--amp", dest="amp", action="store_true", help="Use Automatic Mixed Precision (AMP) to reduce memory usage")
    parser.add_argument("--no-amp", dest="amp", action="store_false", help="Disable Automatic Mixed Precision (AMP)")
    parser.set_defaults(amp=True)

    args = parser.parse_args()

    return PRSMedConfig(**vars(args))