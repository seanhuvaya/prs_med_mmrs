import torch
import torch.nn as nn

from torchvision import transforms
from models.vision_backbone.tinysam.build_sam import build_sam_vit_t

class TinySAMVisionBackbone(nn.Module):
    """
    Extracts dense, pixel-level features from medical images using TinyViT-based TinySAM encoder.
    """
    def __init__(self, checkpoint_path: str, image_size: int = 1024, device: str = None):
        super().__init__()
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"[INFO] Loading TinySAM checkpoint from {checkpoint_path}")
        sam_model = build_sam_vit_t(checkpoint=checkpoint_path, map_location=self.device)

        # Extracting TinyViT image encorder
        self.encoder = sam_model.image_encoder.to(self.device)

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

    def forward(self, x):
        if isinstance(x, list):
            x = torch.stack([self.transform(img) for img in x])
        elif isinstance(x, torch.Tensor):
            # If tensor is already ImageNet-normalized (can have values outside [0,1]),
            # assume it's ready. Otherwise, if values are in [0, 255] range, normalize.
            # Simple heuristic: if max > 2.0, likely unnormalized (divide by 255)
            # If already normalized, values typically in range roughly [-2, 2]
            if x.max() > 2.0:
                # Likely unnormalized [0, 255] range
                x = x / 255.0
                # Apply ImageNet normalization
                mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
                x = (x - mean) / std
            # If already normalized (max <= 2.0), assume it's ImageNet-normalized and use as-is
        
        x = x.to(self.device)

        with torch.set_grad_enabled(self.training):
            z_image = self.encoder(x)

        if z_image.shape[-2:] == (64, 64):
            # 64x64 → 16x16 (4x downsampling)
            z_image = torch.nn.functional.adaptive_avg_pool2d(z_image, (16, 16))
        elif z_image.shape[-2:] != (16, 16):
            # Fallback: ensure we get 16x16
            z_image = torch.nn.functional.adaptive_avg_pool2d(z_image, (16, 16))

        return z_image

if __name__ == "__main__":
    model = TinySAMVisionBackbone(checkpoint_path="weights/tinysam_42.3.pth")
    dummy = torch.randn(2, 3, 1024, 1024)
    features = model(dummy)
    print("Output shape:", features.shape)