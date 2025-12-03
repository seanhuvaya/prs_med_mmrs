"""
Vision backbone modules for PRS-Med.
Supports multiple vision encoders: TinySAM and SAM-Med2D.
"""

from typing import Optional
import torch.nn as nn

from models.vision_backbone.tiny_sam_encoder import TinySAMVisionBackbone
from models.vision_backbone.sam_med2d_encoder import SAMMed2DVisionBackbone


def create_vision_backbone(
    encoder_type: str = "tinysam",
    checkpoint_path: str = "weights/tinysam_42.3.pth",
    image_size: int = 1024,
    device: Optional[str] = None,
) -> nn.Module:
    """
    Factory function to create a vision backbone encoder.
    
    Args:
        encoder_type: Type of encoder to use. Options: "tinysam" or "sam_med2d"
        checkpoint_path: Path to the encoder checkpoint file
        image_size: Input image size (default: 1024)
        device: Device to load the model on (auto-detected if None)
    
    Returns:
        Vision backbone encoder module
        
    Raises:
        ValueError: If encoder_type is not supported
    """
    encoder_type = encoder_type.lower()
    
    if encoder_type == "tinysam":
        return TinySAMVisionBackbone(
            checkpoint_path=checkpoint_path,
            image_size=image_size,
            device=device,
        )
    elif encoder_type == "sam_med2d" or encoder_type == "sammed2d":
        return SAMMed2DVisionBackbone(
            checkpoint_path=checkpoint_path,
            image_size=image_size,
            device=device,
        )
    else:
        raise ValueError(
            f"Unsupported encoder_type: {encoder_type}. "
            f"Supported options: 'tinysam', 'sam_med2d'"
        )


__all__ = [
    "TinySAMVisionBackbone",
    "SAMMed2DVisionBackbone",
    "create_vision_backbone",
]

