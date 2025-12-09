import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import warnings
import os


class TinySAMVisionBackbone(nn.Module):
    """
    TinySAM Vision Backbone EXACTLY as described in PRS-Med paper Section 4.

    Paper: "TinySAM image encoder based on TinyViT architecture"
    Output: z_image ∈ ℝ^(b×256×16×16) where input is (b×3×W×H)
    """

    def __init__(self, checkpoint_path: str, image_size: int = 1024, device: str = None):
        """
        Args:
            checkpoint_path: Path to TinySAM checkpoint
            image_size: Input image size (default 1024 as in paper)
            device: Target device
        """
        super().__init__()

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.image_size = image_size

        # Check if checkpoint exists BEFORE trying to load it
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"TinySAM checkpoint not found: {checkpoint_path}\n"
                "Download from: https://github.com/xingxing-zhang/tinysam"
            )

        print(f"[TinySAM] Loading from {checkpoint_path}")
        print(f"[TinySAM] Target device: {self.device}")

        # ------------------------------------------------------------
        # 1. Suppress warnings during TinySAM import
        # ------------------------------------------------------------
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            warnings.filterwarnings("ignore", category=UserWarning)

            try:
                from tinysam import sam_model_registry
                print("[TinySAM] Successfully imported TinySAM library")

                # Build TinySAM model (vit_tiny as specified in paper)
                # Note: sam_model_registry["vit_t"] with checkpoint=None
                # creates an uninitialized model
                sam_model = sam_model_registry["vit_t"](checkpoint=checkpoint_path)

                # Extract the image encoder (TinyViT backbone)
                self.encoder = sam_model.image_encoder

                print(f"[TinySAM] Encoder type: {type(self.encoder).__name__}")

            except ImportError as e:
                raise ImportError(
                    "TinySAM library not found. Install with:\n"
                    "pip install 'git+https://github.com/xingxing-zhang/tinysam.git'"
                ) from e
            except FileNotFoundError as e:
                # Re-raise FileNotFoundError for clarity
                raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}") from e
            except Exception as e:
                # Catch any other loading errors
                raise RuntimeError(f"Failed to load TinySAM from {checkpoint_path}: {e}") from e

        # ------------------------------------------------------------
        # 2. Initialize channel projection (for ensuring 256 channels)
        # ------------------------------------------------------------
        self.channel_proj = None  # Will be created dynamically if needed

        # ------------------------------------------------------------
        # 3. ImageNet normalization constants
        # ------------------------------------------------------------
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )

        # ------------------------------------------------------------
        # 4. Move everything to target device
        # ------------------------------------------------------------
        self._move_to_device()

        # ------------------------------------------------------------
        # 5. Set trainable parameters (encoder unfrozen as per paper)
        # ------------------------------------------------------------
        self._set_trainable_params()

    def _move_to_device(self):
        """Move encoder and buffers to target device."""
        # Move encoder
        self.encoder = self.encoder.to(self.device)

        # Move buffers
        self.mean = self.mean.to(self.device)
        self.std = self.std.to(self.device)

    def _set_trainable_params(self):
        """Set parameters as trainable (encoder unfrozen as per paper)."""
        # Keep encoder unfrozen for medical fine-tuning
        for param in self.encoder.parameters():
            param.requires_grad = True

        print(f"[TinySAM] Encoder parameters: {sum(p.numel() for p in self.encoder.parameters()):,}")
        print(
            f"[TinySAM] Trainable parameters: {sum(p.numel() for p in self.encoder.parameters() if p.requires_grad):,}")

    def _normalize_image(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply ImageNet normalization to input tensor.
        Handles both [0, 1] and [0, 255] input ranges.
        """
        # Check if values are likely in [0, 255] range
        if x.max() > 1.5:  # Conservative threshold
            x = x / 255.0

        # Apply normalization
        x = (x - self.mean) / self.std

        return x

    def _resize_to_target(self, x: torch.Tensor, target_size: Tuple[int, int] = (1024, 1024)):
        """
        Resize input to target size if needed.
        Paper uses 1024x1024 input.
        """
        if x.shape[-2:] != target_size:
            x = F.interpolate(
                x,
                size=target_size,
                mode='bilinear',
                align_corners=False,
            )
        return x

    def _extract_and_format_features(self, features):
        """
        Extract and format features from TinySAM encoder.
        Ensures output is (B, 256, 16, 16) as per paper.
        """
        # Handle different output formats
        if isinstance(features, dict):
            # Try common keys used by SAM/TinySAM
            key_priority = ['image_embeddings', 'image_embedding', 'feat', 'features', '0']
            for key in key_priority:
                if key in features and isinstance(features[key], torch.Tensor):
                    features = features[key]
                    break
            else:
                # Take first tensor value
                for val in features.values():
                    if isinstance(val, torch.Tensor):
                        features = val
                        break

        elif isinstance(features, (tuple, list)):
            # Take first tensor element (main output)
            for item in features:
                if isinstance(item, torch.Tensor):
                    features = item
                    break

        # Ensure we have a tensor
        if not isinstance(features, torch.Tensor):
            raise TypeError(f"TinySAM encoder returned non-tensor: {type(features)}")

        # ------------------------------------------------------------
        # CRITICAL: Ensure output is (B, 256, 16, 16) as per paper
        # ------------------------------------------------------------
        B = features.shape[0]

        # 1. Ensure 256 channels
        if features.shape[1] != 256:
            # Paper specifies 256 channels, so we need to project if different
            if self.channel_proj is None:
                self.channel_proj = nn.Conv2d(features.shape[1], 256, 1).to(features.device)
                # Make projection trainable
                for param in self.channel_proj.parameters():
                    param.requires_grad = True

            features = self.channel_proj(features)

        # 2. Ensure 16x16 spatial dimensions
        if features.shape[-2:] != (16, 16):
            # Paper specifically states output is 16x16
            features = F.adaptive_avg_pool2d(features, (16, 16))

        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass exactly as described in PRS-Med paper.

        Args:
            x: Input image tensor [B, 3, H, W]

        Returns:
            z_image: Image embeddings [B, 256, 16, 16]
        """
        # ------------------------------------------------------------
        # Step 1: Move input to correct device
        # ------------------------------------------------------------
        x = x.to(self.device)

        # ------------------------------------------------------------
        # Step 2: Resize to paper-specified input size (1024x1024)
        # ------------------------------------------------------------
        x = self._resize_to_target(x, (self.image_size, self.image_size))

        # ------------------------------------------------------------
        # Step 3: Apply ImageNet normalization
        # ------------------------------------------------------------
        x = self._normalize_image(x)

        # ------------------------------------------------------------
        # Step 4: Forward through TinySAM encoder
        # ------------------------------------------------------------
        with torch.set_grad_enabled(self.training):
            # The paper states encoder is kept unfrozen
            features = self.encoder(x)

        # ------------------------------------------------------------
        # Step 5: Extract and format features to paper specification
        # ------------------------------------------------------------
        z_image = self._extract_and_format_features(features)

        return z_image