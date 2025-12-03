import torch
import torch.nn as nn
from torchvision import transforms

# Try to import SAM2 (for SAM-Med2D based on SAM2)
try:
    from sam2.build_sam import build_sam2
    SAM2_AVAILABLE = True
except ImportError:
    SAM2_AVAILABLE = False


class SAMMed2DVisionBackbone(nn.Module):
    """
    Extracts dense, pixel-level features from medical images using SAM2 / SAM-Med2D encoder.

    This implementation is tailored for the official SAM2.1 hiera tiny checkpoint:
      - config: configs/sam2.1/sam2.1_hiera_t.yaml
      - weights: sam2.1_hiera_tiny.pt

    It produces features shaped (B, 256, 16, 16) to match TinySAM for the fusion module.
    """

    def __init__(self, checkpoint_path: str, image_size: int = 1024, device: str = None):
        super().__init__()
        self.device = device or (
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available()
            else "cpu"
        )

        print(f"[INFO] Loading SAM2/SAM-Med2D checkpoint from {checkpoint_path}")

        if not SAM2_AVAILABLE:
            raise ImportError(
                "SAM2 is required for SAMMed2DVisionBackbone.\n"
                "Install with: pip install 'git+https://github.com/facebookresearch/segment-anything-2.git'"
            )

        # Build SAM2 model using the official config for hiera tiny
        config_file = "configs/sam2.1/sam2.1_hiera_t.yaml"
        try:
            print(f"[INFO] Building SAM2 model with config '{config_file}'...")
            sam_model = build_sam2(
                config_file=config_file,
                ckpt_path=checkpoint_path,
                device=self.device,
            )
            encoder = sam_model.image_encoder
            print("[INFO] Successfully loaded SAM2 hiera_tiny image encoder")
        except Exception as e:
            raise RuntimeError(
                f"Failed to build SAM2 model from config '{config_file}' and checkpoint '{checkpoint_path}': {e}"
            )

        self.encoder = encoder.to(self.device)

        # Detect output channels by running a dummy forward pass
        encoder_channels = 1280  # sensible default
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, image_size, image_size).to(self.device)
            dummy_output = self.encoder(dummy_input)

            # Some SAM/SAM2 variants return dicts; extract the main image embedding
            if isinstance(dummy_output, dict):
                for k in ["image_embeddings", "image_embedding", "feat", "features"]:
                    if k in dummy_output and isinstance(dummy_output[k], torch.Tensor):
                        dummy_output = dummy_output[k]
                        break
                else:
                    # Fallback: first tensor value in dict
                    for v in dummy_output.values():
                        if isinstance(v, torch.Tensor):
                            dummy_output = v
                            break

            if isinstance(dummy_output, torch.Tensor):
                encoder_channels = dummy_output.shape[1]
            elif isinstance(dummy_output, (list, tuple)) and len(dummy_output) > 0:
                encoder_channels = dummy_output[0].shape[1]

            print(f"[INFO] Detected SAM2 encoder output channels: {encoder_channels}")

        # Keep encoder unfrozen for medical fine-tuning
        for param in self.encoder.parameters():
            param.requires_grad = True

        # Define preprocessing consistent with SAM training (ImageNet stats)
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        # Project to 256 channels to match TinySAM output
        self.output_proj = nn.Conv2d(encoder_channels, 256, kernel_size=1).to(self.device)

    def forward(self, x):
        if isinstance(x, list):
            x = torch.stack([self.transform(img) for img in x])
        elif isinstance(x, torch.Tensor):
            # If tensor is already ImageNet-normalized (can have values outside [0,1]),
            # assume it's ready. Otherwise, if values are in [0, 255] range, normalize.
            if x.max() > 2.0:
                # Likely unnormalized [0, 255] range
                x = x / 255.0
                mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
                x = (x - mean) / std

        x = x.to(self.device)

        with torch.set_grad_enabled(self.training):
            z_image = self.encoder(x)

        # Handle dict outputs from SAM/SAM2 encoders
        if isinstance(z_image, dict):
            for k in ["image_embeddings", "image_embedding", "feat", "features"]:
                if k in z_image and isinstance(z_image[k], torch.Tensor):
                    z_image = z_image[k]
                    break
            else:
                for v in z_image.values():
                    if isinstance(v, torch.Tensor):
                        z_image = v
                        break

        # Project to 256 channels
        z_image = self.output_proj(z_image)

        # Ensure output is 16x16 to match TinySAM
        if z_image.shape[-2:] != (16, 16):
            z_image = torch.nn.functional.adaptive_avg_pool2d(z_image, (16, 16))

        return z_image


if __name__ == "__main__":
    # Test with your local SAM2 hiera tiny checkpoint
    try:
        model = SAMMed2DVisionBackbone(checkpoint_path="weights/sam2.1_hiera_tiny.pt")
        dummy = torch.randn(2, 3, 1024, 1024)
        features = model(dummy)
        print("Output shape:", features.shape)
        print("Expected: (2, 256, 16, 16)")
    except Exception as e:
        print(f"Error: {e}")
        print("Note: This requires SAM2 to be installed and a valid SAM2.1 hiera tiny checkpoint")

