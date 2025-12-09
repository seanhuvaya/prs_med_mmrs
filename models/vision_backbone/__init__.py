from typing import Optional
import torch.nn as nn

from models.vision_backbone.tiny_sam_encoder import TinySAMVisionBackbone
from models.vision_backbone.sam_med2d_encoder import SAMMed2DVisionBackbone


def create_vision_backbone(
    encoder_type: str = "tinysam",
    checkpoint_path: Optional[str] = None,
    image_size: int = 1024,
    device: Optional[str] = None,
) -> nn.Module:
    """
    Factory function to create a vision backbone encoder.

    Args:
        encoder_type: "tinysam" or "sam_med2d"
        checkpoint_path: Path to the encoder checkpoint file.
                         If None, a sensible default is chosen per encoder type.
        image_size: Input image size (default: 1024)
        device: Device to load the model on (auto-detected if None)
    """
    encoder_type = encoder_type.lower()

    if encoder_type == "tinysam":
        # Default TinySAM checkpoint
        if checkpoint_path is None:
            checkpoint_path = "weights/tinysam_42.3.pth"

        return TinySAMVisionBackbone(
            checkpoint_path=checkpoint_path,
            image_size=image_size,
            device=device,
        )

    elif encoder_type in ("sam_med2d", "sammed2d"):
        # Default SAM-Med2D / SAM2 checkpoint
        if checkpoint_path is None:
            # TODO: replace with your actual SAM-Med2D checkpoint
            checkpoint_path = "weights/sam2.1_hiera_tiny.pt"

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
    "create_vision_backbone",
]