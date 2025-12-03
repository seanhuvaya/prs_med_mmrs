import torch
import torch.nn as nn
from torchvision import transforms
import os

# Try to import SAM2 (for SAM-Med2D based on SAM2)
try:
    from sam2.build_sam import build_sam2
    SAM2_AVAILABLE = True
except ImportError:
    SAM2_AVAILABLE = False

# Try to import official SAM-Med2D if available
try:
    from sam_med_2d import build_sam_med2d
    SAM_MED2D_AVAILABLE = True
except ImportError:
    SAM_MED2D_AVAILABLE = False


class SAMMed2DVisionBackbone(nn.Module):
    """
    Extracts dense, pixel-level features from medical images using SAM-Med2D encoder.
    SAM-Med2D is based on SAM's ViT encoder with adapter layers, fine-tuned for medical images.
    
    Supports multiple checkpoint formats:
    1. Official SAM-Med2D checkpoint (if sam_med_2d package is installed)
    2. SAM2-based checkpoint (if sam2 package is installed)
    3. Direct PyTorch checkpoint with image_encoder key
    """
    def __init__(self, checkpoint_path: str, image_size: int = 1024, device: str = None):
        super().__init__()
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"[INFO] Loading SAM-Med2D checkpoint from {checkpoint_path}")
        
        # Try different loading methods
        encoder = None
        encoder_channels = 1280  # Default for ViT-H
        
        # Method 1: Try official SAM-Med2D package
        if SAM_MED2D_AVAILABLE:
            try:
                print("[INFO] Attempting to load with official SAM-Med2D package...")
                sam_model = build_sam_med2d(checkpoint_path, device=self.device)
                encoder = sam_model.image_encoder
                print("[INFO] Successfully loaded with SAM-Med2D package")
            except Exception as e:
                print(f"[WARNING] Failed to load with SAM-Med2D package: {e}")
        
        # Method 2: Try SAM2 package
        if encoder is None and SAM2_AVAILABLE:
            try:
                print("[INFO] Attempting to load with SAM2 package...")
                # Try different model sizes
                for model_cfg in ["sam2_h.yaml", "sam2_l.yaml", "sam2_b.yaml"]:
                    try:
                        sam_model = build_sam2(
                            model_cfg=model_cfg,
                            ckpt_path=checkpoint_path,
                            device=self.device
                        )
                        encoder = sam_model.image_encoder
                        print(f"[INFO] Successfully loaded with SAM2 ({model_cfg})")
                        break
                    except Exception:
                        continue
            except Exception as e:
                print(f"[WARNING] Failed to load with SAM2 package: {e}")
        
        # Method 3: Try direct checkpoint loading (if checkpoint has encoder state dict)
        if encoder is None:
            try:
                print("[INFO] Attempting to load checkpoint directly...")
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                
                # Try different checkpoint structures
                if 'model' in checkpoint:
                    if hasattr(checkpoint['model'], 'image_encoder'):
                        encoder = checkpoint['model'].image_encoder
                    elif 'image_encoder' in checkpoint['model']:
                        encoder = checkpoint['model']['image_encoder']
                elif 'image_encoder' in checkpoint:
                    # If checkpoint contains encoder state dict, we'd need to build architecture first
                    # This requires knowing the exact architecture, so we skip this method
                    print("[WARNING] Checkpoint contains image_encoder but architecture unknown")
            except Exception as e:
                print(f"[WARNING] Failed to load checkpoint directly: {e}")
        
        if encoder is None:
            raise RuntimeError(
                f"Failed to load SAM-Med2D encoder from {checkpoint_path}. "
                f"Please ensure:\n"
                f"  1. The checkpoint file exists and is valid\n"
                f"  2. Install SAM-Med2D: pip install sam-med-2d\n"
                f"  3. Or install SAM2: pip install git+https://github.com/facebookresearch/segment-anything-2.git"
            )
        
        self.encoder = encoder.to(self.device)
        
        # Detect output channels by running a dummy forward pass
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, image_size, image_size).to(self.device)
            dummy_output = self.encoder(dummy_input)
            if isinstance(dummy_output, torch.Tensor):
                encoder_channels = dummy_output.shape[1]
            elif isinstance(dummy_output, (list, tuple)):
                encoder_channels = dummy_output[0].shape[1] if len(dummy_output) > 0 else 1280
            print(f"[INFO] Detected encoder output channels: {encoder_channels}")
        
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
        
        # SAM2 ViT encoder outputs features at different resolution
        # Typically outputs at 64x64 for 1024x1024 input
        # Project to 256 channels and downsample to 16x16 to match TinySAM output
        if hasattr(self, 'output_proj'):
            z_image = self.output_proj(z_image)
        
        # Ensure output is 16x16 to match TinySAM
        if z_image.shape[-2:] != (16, 16):
            z_image = torch.nn.functional.adaptive_avg_pool2d(z_image, (16, 16))

        return z_image


# Alternative implementation using direct checkpoint loading if SAM-Med2D has custom structure
class SAMMed2DVisionBackboneDirect(nn.Module):
    """
    Alternative implementation that loads SAM-Med2D checkpoint directly.
    Use this if SAM-Med2D has a different checkpoint format.
    """
    def __init__(self, checkpoint_path: str, image_size: int = 1024, device: str = None):
        super().__init__()
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"[INFO] Loading SAM-Med2D checkpoint from {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Try to extract image encoder from checkpoint
        # Adjust these keys based on actual SAM-Med2D checkpoint structure
        if 'image_encoder' in checkpoint:
            self.encoder = checkpoint['image_encoder']
        elif 'model' in checkpoint and hasattr(checkpoint['model'], 'image_encoder'):
            self.encoder = checkpoint['model'].image_encoder
        else:
            # If checkpoint is just the encoder state dict
            # You'll need to build the encoder architecture first
            # This is a placeholder - adjust based on actual SAM-Med2D architecture
            raise NotImplementedError(
                "Direct checkpoint loading requires knowing the encoder architecture. "
                "Please use SAMMed2DVisionBackbone with SAM2 build_sam2 function instead."
            )
        
        self.encoder = self.encoder.to(self.device)
        
        # Keep encoder unfrozen for medical fine-tuning
        for param in self.encoder.parameters():
            param.requires_grad = True
            
        # Define preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            ),
        ])
        
        # Project to 256 channels to match TinySAM
        # Adjust input channels based on actual encoder output
        self.output_proj = nn.Conv2d(1280, 256, kernel_size=1).to(self.device)

    def forward(self, x):
        if isinstance(x, list):
            x = torch.stack([self.transform(img) for img in x])
        elif isinstance(x, torch.Tensor):
            if x.max() > 2.0:
                x = x / 255.0
                mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
                x = (x - mean) / std
        
        x = x.to(self.device)

        with torch.set_grad_enabled(self.training):
            z_image = self.encoder(x)
        
        # Project and resize to match TinySAM output
        if hasattr(self, 'output_proj'):
            z_image = self.output_proj(z_image)
        
        if z_image.shape[-2:] != (16, 16):
            z_image = torch.nn.functional.adaptive_avg_pool2d(z_image, (16, 16))

        return z_image


if __name__ == "__main__":
    # Test with dummy checkpoint path
    try:
        model = SAMMed2DVisionBackbone(checkpoint_path="weights/sam_med2d.pth")
        dummy = torch.randn(2, 3, 1024, 1024)
        features = model(dummy)
        print("Output shape:", features.shape)
        print("Expected: (2, 256, 16, 16)")
    except Exception as e:
        print(f"Error: {e}")
        print("Note: This requires SAM2 to be installed and a valid SAM-Med2D checkpoint")

