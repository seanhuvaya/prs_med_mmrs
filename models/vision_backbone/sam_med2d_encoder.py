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

        # Extract checkpoint first if it's a training checkpoint, then detect model type from extracted state_dict
        import tempfile
        import os
        
        checkpoint_to_use = checkpoint_path
        temp_checkpoint_path = None
        extracted_state_dict = None
        
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
                    extracted_state_dict = model_state_dict
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
                                    extracted_state_dict = nested
                                    print(f"[INFO] Found SAM model state_dict nested under '{key}' key")
                                    break
                        else:
                            # If no nested structure found, use the model_state_dict as-is
                            # It might be the actual state_dict but with different key structure
                            extracted_state_dict = model_state_dict
                            print(f"[WARNING] Could not find standard SAM keys, using model state_dict as-is")
                    else:
                        extracted_state_dict = model_state_dict
                
                # Always re-detect model type from extracted state_dict to ensure accuracy
                print(f"[INFO] Detecting model type from extracted state_dict...")
                detected_type = None
                # Check for image_encoder.trunk dimensions (SAM-Med2D structure)
                if 'image_encoder.trunk.patch_embed.proj.weight' in extracted_state_dict:
                    embed_dim = extracted_state_dict['image_encoder.trunk.patch_embed.proj.weight'].shape[0]
                    if embed_dim == 96:
                        detected_type = "vit_b"
                        print(f"[INFO] Detected vit_b from extracted checkpoint (embed_dim={embed_dim})")
                    elif embed_dim == 144:
                        detected_type = "vit_l"
                        print(f"[INFO] Detected vit_l from extracted checkpoint (embed_dim={embed_dim})")
                    elif embed_dim == 128:
                        detected_type = "vit_h"
                        print(f"[INFO] Detected vit_h from extracted checkpoint (embed_dim={embed_dim})")
                # Also check for standard SAM structure (without trunk)
                elif 'image_encoder.patch_embed.proj.weight' in extracted_state_dict:
                    embed_dim = extracted_state_dict['image_encoder.patch_embed.proj.weight'].shape[0]
                    if embed_dim == 768:
                        detected_type = "vit_b"
                        print(f"[INFO] Detected vit_b from extracted checkpoint (embed_dim={embed_dim})")
                    elif embed_dim == 1024:
                        detected_type = "vit_l"
                        print(f"[INFO] Detected vit_l from extracted checkpoint (embed_dim={embed_dim})")
                    elif embed_dim == 1280:
                        detected_type = "vit_h"
                        print(f"[INFO] Detected vit_h from extracted checkpoint (embed_dim={embed_dim})")
                
                # Use detected type if found, otherwise keep the original detection
                if detected_type is not None:
                    if model_type is not None and model_type != detected_type:
                        print(f"[WARNING] Model type mismatch: previously detected '{model_type}', but extracted checkpoint suggests '{detected_type}'. Using '{detected_type}'.")
                    model_type = detected_type
                elif model_type is None:
                    # If we couldn't detect from extracted state_dict and no previous detection, default to vit_b
                    model_type = "vit_b"
                    print(f"[WARNING] Could not detect model type from extracted checkpoint, defaulting to 'vit_b'")
                
                # Save extracted state_dict to temporary file for SAM to load
                temp_checkpoint = tempfile.NamedTemporaryFile(delete=False, suffix='.pth')
                temp_checkpoint_path = temp_checkpoint.name
                temp_checkpoint.close()
                torch.save(extracted_state_dict, temp_checkpoint_path)
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
                # Re-detect model type if not already detected
                if model_type is None or model_type == "vit_b":
                    if 'image_encoder.patch_embed.proj.weight' in checkpoint_data:
                        embed_dim = checkpoint_data['image_encoder.patch_embed.proj.weight'].shape[0]
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
        
        # Only try the detected model type - don't fallback to other types unless detection failed
        if model_type is None:
            # If detection completely failed, try all types in order
            model_types_to_try = ["vit_b", "vit_l", "vit_h"]
            print(f"[WARNING] Model type detection failed, will try all types: {model_types_to_try}")
        else:
            # Use only the detected model type
            model_types_to_try = [model_type]
            print(f"[INFO] Will try loading with detected model type: {model_type}")
        
        # Detect checkpoint image size from position embeddings
        checkpoint_image_size = None
        try:
            checkpoint_state = torch.load(checkpoint_to_use, map_location='cpu', weights_only=False)
            if 'image_encoder.pos_embed' in checkpoint_state:
                pos_embed_shape = checkpoint_state['image_encoder.pos_embed'].shape
                # pos_embed shape is [1, H, W, C] where H=W=image_size/patch_size
                # patch_size is typically 16 for SAM
                if len(pos_embed_shape) == 4:
                    patch_h, patch_w = pos_embed_shape[1], pos_embed_shape[2]
                    checkpoint_image_size = patch_h * 16  # Assuming patch_size=16
                    print(f"[INFO] Detected checkpoint image size: {checkpoint_image_size}x{checkpoint_image_size} (from pos_embed shape {pos_embed_shape})")
        except Exception as e:
            print(f"[WARNING] Could not detect checkpoint image size: {e}")
        
        # If checkpoint has different image size, we'll need to resize position embeddings
        needs_resize = checkpoint_image_size is not None and checkpoint_image_size != image_size
        
        try:
            for try_model_type in model_types_to_try:
                try:
                    print(f"[INFO] Trying to build SAM-Med2D model with type '{try_model_type}'...")
                    
                    # Build model without checkpoint first if we need to resize
                    if needs_resize:
                        print(f"[INFO] Checkpoint image size ({checkpoint_image_size}) differs from target ({image_size}), will resize position embeddings")
                        # Build model without loading checkpoint
                        sam_model = sam_model_registry[try_model_type](checkpoint=None)
                        
                        # Load checkpoint state dict
                        checkpoint_state = torch.load(checkpoint_to_use, map_location='cpu', weights_only=False)
                        
                        # Resize position embeddings
                        if 'image_encoder.pos_embed' in checkpoint_state:
                            old_pos_embed = checkpoint_state['image_encoder.pos_embed']
                            # old_pos_embed shape: [1, H_old, W_old, C]
                            # new_pos_embed shape: [1, H_new, W_new, C]
                            old_h, old_w = old_pos_embed.shape[1], old_pos_embed.shape[2]
                            new_h = image_size // 16  # patch_size = 16
                            new_w = image_size // 16
                            
                            # Resize using interpolation
                            old_pos_embed_2d = old_pos_embed.squeeze(0).permute(2, 0, 1)  # [C, H, W]
                            new_pos_embed_2d = torch.nn.functional.interpolate(
                                old_pos_embed_2d.unsqueeze(0),
                                size=(new_h, new_w),
                                mode='bilinear',
                                align_corners=False
                            ).squeeze(0)  # [C, H_new, W_new]
                            new_pos_embed = new_pos_embed_2d.permute(1, 2, 0).unsqueeze(0)  # [1, H_new, W_new, C]
                            checkpoint_state['image_encoder.pos_embed'] = new_pos_embed
                            print(f"[INFO] Resized pos_embed from [{old_h}, {old_w}] to [{new_h}, {new_w}]")
                        
                        # Resize relative position embeddings
                        # rel_pos_h and rel_pos_w have shape [2*H-1, head_dim] or [2*W-1, head_dim]
                        for key in list(checkpoint_state.keys()):
                            if 'rel_pos_h' in key or 'rel_pos_w' in key:
                                old_rel_pos = checkpoint_state[key]
                                if len(old_rel_pos.shape) == 2:
                                    # Shape: [2*H-1, head_dim]
                                    old_size = (old_rel_pos.shape[0] + 1) // 2
                                    new_size = image_size // 16
                                    new_rel_pos_size = 2 * new_size - 1
                                    
                                    # Interpolate relative position embeddings
                                    # We need to handle the 1D nature of these embeddings
                                    old_rel_pos_expanded = old_rel_pos.unsqueeze(0).unsqueeze(0)  # [1, 1, 2*H-1, head_dim]
                                    new_rel_pos_expanded = torch.nn.functional.interpolate(
                                        old_rel_pos_expanded.permute(0, 3, 1, 2),  # [1, head_dim, 1, 2*H-1]
                                        size=(1, new_rel_pos_size),
                                        mode='bilinear',
                                        align_corners=False
                                    ).permute(0, 2, 3, 1).squeeze(0).squeeze(0)  # [2*H_new-1, head_dim]
                                    checkpoint_state[key] = new_rel_pos_expanded
                                    print(f"[INFO] Resized {key} from [{old_rel_pos.shape[0]}] to [{new_rel_pos_size}]")
                        
                        # Load the resized state dict
                        missing_keys, unexpected_keys = sam_model.load_state_dict(checkpoint_state, strict=False)
                        if missing_keys:
                            print(f"[WARNING] Missing keys when loading checkpoint: {len(missing_keys)} keys")
                        if unexpected_keys:
                            print(f"[WARNING] Unexpected keys in checkpoint: {len(unexpected_keys)} keys")
                    else:
                        # Normal loading path
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
                    # Size mismatch or other loading error
                    last_error = e
                    error_msg = str(e).lower()
                    
                    # If we detected a specific model type, don't try others - the detection was wrong or checkpoint is corrupted
                    if len(model_types_to_try) == 1:
                        raise RuntimeError(
                            f"Failed to load SAM-Med2D checkpoint '{checkpoint_path}' with detected model type '{try_model_type}'.\n"
                            f"Error: {e}\n"
                            f"This suggests either:\n"
                            f"  1. The checkpoint is corrupted or incomplete\n"
                            f"  2. The checkpoint is for a different model architecture\n"
                            f"  3. The model type detection was incorrect\n"
                            f"Please verify the checkpoint file and model type."
                        )
                    
                    # Check if it's a size mismatch or shape error - only try next if we're trying multiple types
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

