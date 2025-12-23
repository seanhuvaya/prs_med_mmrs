import torch
import torch.nn as nn
from torchvision import transforms
import functools

# Patch torch.load before importing SAM-Med2D to ensure compatibility with PyTorch 2.6+
# This allows loading checkpoints that contain optimizer states
_original_torch_load = torch.load

@functools.wraps(_original_torch_load)
def _patched_torch_load(f, *args, **kwargs):
    """Patched torch.load that defaults to weights_only=False for PyTorch 2.6+ compatibility."""
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(f, *args, **kwargs)

# Apply patch globally
torch.load = _patched_torch_load

# Import SAM-Med2D (installed as sam-med2d package via uv)
# The package exposes segment_anything as top-level module
try:
    from segment_anything import sam_model_registry
except (ImportError, ModuleNotFoundError) as e:
    error_str = str(e).lower()
    error_msg = str(e)
    
    # Check if it's a missing modeling module error (incomplete package installation)
    if "modeling" in error_str or ("no module named" in error_str and "segment_anything" in error_str):
        raise ImportError(
            "SAM-Med2D package is installed but incomplete. The 'modeling' module is missing.\n\n"
            "This usually means the SAM-Med2D package's pyproject.toml doesn't include all necessary files.\n"
            "Please check your forked SAM-Med2D repository and ensure the pyproject.toml includes:\n"
            "  - segment_anything/modeling/ directory\n"
            "  - All Python files in the segment_anything package\n\n"
            "Example pyproject.toml configuration:\n"
            "  [tool.setuptools]\n"
            "  packages = find_packages()\n"
            "  # OR explicitly:\n"
            "  # packages = [\"segment_anything\", \"segment_anything.modeling\", ...]\n\n"
            "After updating, reinstall with: uv sync\n"
            f"Original error: {error_msg}"
        ) from e
    
    # Try alternative import paths as fallback
    try:
        from sam_med2d.segment_anything import sam_model_registry
    except (ImportError, ModuleNotFoundError):
        raise ImportError(
            "SAM-Med2D (sam-med2d) is required but not found.\n"
            "Install it by running: uv sync\n"
            "Make sure sam-med2d is in your pyproject.toml dependencies.\n"
            f"Import error: {error_msg}"
        ) from e


class SAMMed2DVisionBackbone(nn.Module):
    """
    Extracts dense, pixel-level features from medical images using SAM-Med2D encoder.
    
    This implementation uses the official SAM-Med2D from OpenGVLab:
      - Repository: https://github.com/OpenGVLab/SAM-Med2D
      - Uses segment_anything module (original SAM architecture)
      - Supports model types: vit_b, vit_l, vit_h
      - Checkpoint format: sam-med2d_b.pth, sam-med2d_l.pth, etc.

    It produces features shaped (B, 256, 16, 16) to match TinySAM for the fusion module.
    """

    def __init__(self, checkpoint_path: str, image_size: int = 1024, device: str = None, model_type: str = None):
        super().__init__()
        self.device = device or (
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available()
            else "cpu"
        )

        print(f"[INFO] Loading SAM-Med2D checkpoint from {checkpoint_path}")

        # Determine model type from checkpoint if not specified
        # Priority: 1) Checkpoint inspection, 2) Filename, 3) Default
        if model_type is None:
            # First, try to determine from checkpoint dimensions (most reliable)
            try:
                print(f"[INFO] Inspecting checkpoint to determine model type...")
                checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                
                # Handle training checkpoint format (has "model" key)
                if isinstance(checkpoint, dict) and "model" in checkpoint:
                    checkpoint = checkpoint["model"]
                    # Check if nested further
                    if isinstance(checkpoint, dict) and not any(
                        key.startswith("image_encoder") or 
                        key.startswith("prompt_encoder") or 
                        key.startswith("mask_decoder")
                        for key in checkpoint.keys()
                    ):
                        # Try common nested keys
                        for key in ["state_dict", "model_state_dict", "model", "sam_model"]:
                            if key in checkpoint and isinstance(checkpoint[key], dict):
                                checkpoint = checkpoint[key]
                                break
                
                # Check for image_encoder.trunk dimensions (SAM-Med2D structure)
                if 'image_encoder.trunk.patch_embed.proj.weight' in checkpoint:
                    embed_dim = checkpoint['image_encoder.trunk.patch_embed.proj.weight'].shape[0]
                    if embed_dim == 96:
                        model_type = "vit_b"
                        print(f"[INFO] Detected vit_b from checkpoint (embed_dim={embed_dim})")
                    elif embed_dim == 144:
                        model_type = "vit_l"
                        print(f"[INFO] Detected vit_l from checkpoint (embed_dim={embed_dim})")
                    elif embed_dim == 128:
                        model_type = "vit_h"
                        print(f"[INFO] Detected vit_h from checkpoint (embed_dim={embed_dim})")
                    else:
                        print(f"[WARNING] Unknown embed_dim {embed_dim}, will try to infer from filename")
                        model_type = None  # Will try filename next
                # Also check for standard SAM structure (without trunk)
                elif 'image_encoder.patch_embed.proj.weight' in checkpoint:
                    embed_dim = checkpoint['image_encoder.patch_embed.proj.weight'].shape[0]
                    if embed_dim == 768:
                        model_type = "vit_b"
                        print(f"[INFO] Detected vit_b from checkpoint (embed_dim={embed_dim})")
                    elif embed_dim == 1024:
                        model_type = "vit_l"
                        print(f"[INFO] Detected vit_l from checkpoint (embed_dim={embed_dim})")
                    elif embed_dim == 1280:
                        model_type = "vit_h"
                        print(f"[INFO] Detected vit_h from checkpoint (embed_dim={embed_dim})")
                    else:
                        print(f"[WARNING] Unknown embed_dim {embed_dim}, will try to infer from filename")
                        model_type = None  # Will try filename next
                else:
                    print(f"[WARNING] Could not find image_encoder in checkpoint, will try filename")
                    model_type = None  # Will try filename next
            except Exception as e:
                print(f"[WARNING] Could not inspect checkpoint: {e}, will try filename")
                model_type = None  # Will try filename next
            
            # If checkpoint inspection didn't work, try filename
            if model_type is None:
                checkpoint_lower = checkpoint_path.lower()
                if "vit_h" in checkpoint_lower or "_h.pth" in checkpoint_lower or "huge" in checkpoint_lower:
                    model_type = "vit_h"
                    print(f"[INFO] Detected vit_h from filename")
                elif "vit_l" in checkpoint_lower or "_l.pth" in checkpoint_lower or "large" in checkpoint_lower:
                    model_type = "vit_l"
                    print(f"[INFO] Detected vit_l from filename")
                elif "vit_b" in checkpoint_lower or "_b.pth" in checkpoint_lower or "base" in checkpoint_lower:
                    model_type = "vit_b"
                    print(f"[INFO] Detected vit_b from filename")
                else:
                    # Default to vit_b
                    model_type = "vit_b"
                    print(f"[WARNING] Could not determine model type, defaulting to 'vit_b'")

        print(f"[INFO] Model type: {model_type}, Image size: {image_size}")

        # Check if checkpoint is a training checkpoint (has "model" key) or direct state_dict
        # Load checkpoint to inspect its structure
        import tempfile
        import os
        
        checkpoint_to_use = checkpoint_path
        temp_checkpoint_path = None
        
        try:
            checkpoint_data = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            
            # Check if this is a training checkpoint with "model" key
            if isinstance(checkpoint_data, dict) and "model" in checkpoint_data:
                print(f"[INFO] Detected training checkpoint format (has 'model' key), extracting model state_dict...")
                model_state_dict = checkpoint_data["model"]
                
                # Check if model_state_dict contains SAM model keys directly
                # SAM model keys typically start with: image_encoder, prompt_encoder, mask_decoder
                has_sam_keys = any(
                    key.startswith("image_encoder") or 
                    key.startswith("prompt_encoder") or 
                    key.startswith("mask_decoder")
                    for key in model_state_dict.keys()
                )
                
                if has_sam_keys:
                    # Direct SAM model state_dict - use it directly
                    actual_state_dict = model_state_dict
                else:
                    # Check if it's a nested structure (e.g., model wrapped in another layer)
                    # Try common nested structures
                    if isinstance(model_state_dict, dict):
                        # Check for common wrapper keys
                        for key in ["state_dict", "model_state_dict", "model", "sam_model"]:
                            if key in model_state_dict:
                                nested = model_state_dict[key]
                                if isinstance(nested, dict) and any(
                                    k.startswith("image_encoder") or 
                                    k.startswith("prompt_encoder") or 
                                    k.startswith("mask_decoder")
                                    for k in nested.keys()
                                ):
                                    actual_state_dict = nested
                                    print(f"[INFO] Found SAM model state_dict nested under '{key}' key")
                                    break
                        else:
                            # If no nested structure found, use the model_state_dict as-is
                            # It might be the actual state_dict but with different key structure
                            actual_state_dict = model_state_dict
                            print(f"[WARNING] Could not find standard SAM keys, using model state_dict as-is")
                    else:
                        actual_state_dict = model_state_dict
                
                # Save extracted state_dict to temporary file for SAM to load
                temp_checkpoint = tempfile.NamedTemporaryFile(delete=False, suffix='.pth')
                temp_checkpoint_path = temp_checkpoint.name
                temp_checkpoint.close()
                torch.save(actual_state_dict, temp_checkpoint_path)
                print(f"[INFO] Saved extracted model state_dict to temporary file: {temp_checkpoint_path}")
                checkpoint_to_use = temp_checkpoint_path
            elif isinstance(checkpoint_data, dict) and any(
                key.startswith("image_encoder") or 
                key.startswith("prompt_encoder") or 
                key.startswith("mask_decoder")
                for key in checkpoint_data.keys()
            ):
                # Direct SAM model state_dict - use as is
                print(f"[INFO] Checkpoint appears to be a direct SAM model state_dict")
                checkpoint_to_use = checkpoint_path
            else:
                # Unknown structure - try using as-is
                print(f"[INFO] Checkpoint structure unclear, attempting to load as-is")
                checkpoint_to_use = checkpoint_path
        except Exception as e:
            print(f"[WARNING] Could not inspect checkpoint structure: {e}, using checkpoint as-is")
            checkpoint_to_use = checkpoint_path

        # SAM-Med2D registry expects checkpoint as a keyword argument
        # torch.load is already patched at module level for PyTorch 2.6+ compatibility
        # This allows loading checkpoints that contain optimizer states
        sam_model = None
        encoder = None
        last_error = None
        
        # Try loading with the detected/specified model type
        model_types_to_try = [model_type]
        
        # If auto-detection failed or we're unsure, try other variants
        if model_type == "vit_b":
            # Checkpoint might be misnamed - try vit_l if vit_b fails
            model_types_to_try.extend(["vit_l", "vit_h"])
        elif model_type == "vit_l":
            model_types_to_try.extend(["vit_h", "vit_b"])
        elif model_type == "vit_h":
            model_types_to_try.extend(["vit_l", "vit_b"])
        
        try:
            for try_model_type in model_types_to_try:
                try:
                    print(f"[INFO] Trying to build SAM-Med2D model with type '{try_model_type}'...")
                    sam_model = sam_model_registry[try_model_type](checkpoint=checkpoint_to_use)
                    encoder = sam_model.image_encoder
                    print(f"[INFO] Successfully loaded SAM-Med2D {try_model_type} image encoder")
                    model_type = try_model_type  # Update to the successful type
                    break
                except KeyError:
                    raise ValueError(
                        f"Invalid model type '{try_model_type}'. "
                        f"Supported types: {list(sam_model_registry.keys())}"
                    )
                except (RuntimeError, ValueError) as e:
                    # Size mismatch or other loading error - try next type
                    last_error = e
                    error_msg = str(e).lower()
                    # Check if it's a size mismatch or shape error
                    if ("size mismatch" in error_msg or "shape" in error_msg or "copying a param" in error_msg or "missing key" in error_msg):
                        if try_model_type != model_types_to_try[-1]:
                            print(f"[WARNING] Size/shape/key mismatch with '{try_model_type}', trying next model type...")
                            continue
                    
                    # If it's the last attempt, raise the error
                    if try_model_type == model_types_to_try[-1]:
                        raise RuntimeError(
                            f"Failed to load SAM-Med2D checkpoint '{checkpoint_path}' with any model type.\n"
                            f"Tried: {model_types_to_try}\n"
                            f"Last error: {e}\n"
                            f"Please check that the checkpoint matches one of the model types: {list(sam_model_registry.keys())}"
                        )
                    else:
                        # Re-raise if it's not a size mismatch error
                        raise
        finally:
            # Clean up temporary file if we created one
            if temp_checkpoint_path is not None and os.path.exists(temp_checkpoint_path):
                try:
                    os.unlink(temp_checkpoint_path)
                    print(f"[INFO] Cleaned up temporary checkpoint file")
                except:
                    pass
        
        if encoder is None:
            raise RuntimeError(
                f"Failed to load SAM-Med2D model. Checkpoint: {checkpoint_path}, "
                f"Last error: {last_error}"
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

            print(f"[INFO] Detected SAM-Med2D encoder output channels: {encoder_channels}")

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
    # Test with your local SAM-Med2D checkpoint
    try:
        model = SAMMed2DVisionBackbone(
            checkpoint_path="weights/sam-med2d_b.pth",
            model_type="vit_b",  # Can be vit_b, vit_l, or vit_h
            image_size=1024
        )
        dummy = torch.randn(2, 3, 1024, 1024)
        features = model(dummy)
        print("Output shape:", features.shape)
        print("Expected: (2, 256, 16, 16)")
    except Exception as e:
        print(f"Error: {e}")
        print("Note: This requires SAM-Med2D to be installed from https://github.com/OpenGVLab/SAM-Med2D")

