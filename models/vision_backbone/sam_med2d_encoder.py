import torch
import torch.nn as nn
from torchvision import transforms
import functools
from argparse import Namespace

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

        print(f"[INFO] Model type: {model_type}, Target image size: {image_size}")
        
        # SAM-Med2D uses image_size=256 by default (as per their repo)
        # We'll load it with their default size, then wrap it to handle PRS-Med's 1024x1024 requirement
        sam_med2d_image_size = 256  # SAM-Med2D default
        
        # Check if checkpoint is a training checkpoint and extract model state_dict if needed
        import tempfile
        import os
        checkpoint_to_use = checkpoint_path
        temp_checkpoint_path = None
        
        try:
            checkpoint_data = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            if isinstance(checkpoint_data, dict) and "model" in checkpoint_data:
                print(f"[INFO] Detected training checkpoint format, extracting model state_dict...")
                model_state_dict = checkpoint_data["model"]
                
                # Check if nested further
                if isinstance(model_state_dict, dict) and not any(
                    key.startswith("image_encoder") or 
                    key.startswith("prompt_encoder") or 
                    key.startswith("mask_decoder")
                    for key in model_state_dict.keys()
                ):
                    # Try common nested keys
                    for key in ["state_dict", "model_state_dict", "model", "sam_model"]:
                        if key in model_state_dict and isinstance(model_state_dict[key], dict):
                            model_state_dict = model_state_dict[key]
                            break
                
                # Save extracted state_dict to temporary file
                temp_checkpoint = tempfile.NamedTemporaryFile(delete=False, suffix='.pth')
                temp_checkpoint_path = temp_checkpoint.name
                temp_checkpoint.close()
                torch.save(model_state_dict, temp_checkpoint_path)
                checkpoint_to_use = temp_checkpoint_path
                print(f"[INFO] Saved extracted model state_dict to temporary file")
        except Exception as e:
            print(f"[WARNING] Could not check checkpoint format: {e}, using checkpoint as-is")
        
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
        
        # Standard SAM registry expects checkpoint as a keyword argument (string path)
        # SAM-Med2D checkpoints are trained with image_size=256, but standard SAM builds with 1024
        # We'll load the checkpoint and handle size mismatches if needed
        try:
            for try_model_type in model_types_to_try:
                try:
                    print(f"[INFO] Loading SAM-Med2D model with type '{try_model_type}'...")
                    print(f"[INFO] Using checkpoint: {checkpoint_to_use}")
                    
                    # Standard SAM API: sam_model_registry[model_type](checkpoint=path)
                    # This builds a model with image_size=1024, but SAM-Med2D checkpoint has 256
                    # We'll load with strict=False to handle size mismatches
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
                    # Handle size mismatch errors (SAM-Med2D uses 256, standard SAM uses 1024)
                    error_msg = str(e).lower()
                    last_error = e
                    
                    if "size mismatch" in error_msg or "shape" in error_msg or "copying a param" in error_msg:
                        print(f"[DEBUG] ========== SIZE MISMATCH DETECTED ==========")
                        print(f"[DEBUG] Original error: {e}")
                        print(f"[DEBUG] Attempting to load with size adaptation...")
                        
                        # Build model without checkpoint first to see what size it expects
                        print(f"[DEBUG] Building model without checkpoint...")
                        sam_model = sam_model_registry[try_model_type](checkpoint=None)
                        
                        # Check what size the model expects - check multiple sources
                        model_pos_embed = sam_model.image_encoder.pos_embed
                        model_img_size = None
                        model_patch_size = None
                        if len(model_pos_embed.shape) == 4:
                            model_patch_h, model_patch_w = model_pos_embed.shape[1], model_pos_embed.shape[2]
                            model_patch_size = model_patch_h
                            model_img_size = model_patch_h * 16
                            print(f"[DEBUG] Model pos_embed shape: {model_pos_embed.shape}")
                            print(f"[DEBUG] Model expects image_size={model_img_size} (patches: {model_patch_h}x{model_patch_w})")
                        
                        # Also check rel_pos in model to verify - this is the ground truth
                        model_rel_pos_sizes = {}
                        for name, param in sam_model.image_encoder.named_parameters():
                            if 'rel_pos_h' in name or 'rel_pos_w' in name:
                                if len(param.shape) == 2:
                                    model_rel_pos_size = param.shape[0]
                                    model_rel_pos_sizes[name] = model_rel_pos_size
                                    model_patch_from_rel = (model_rel_pos_size + 1) // 2
                                    model_img_from_rel = model_patch_from_rel * 16
                                    print(f"[DEBUG] Model {name} shape: {param.shape}, implies patch_size={model_patch_from_rel}, image_size={model_img_from_rel}")
                        
                        # Use rel_pos to determine model size (more reliable than pos_embed)
                        if model_rel_pos_sizes:
                            # Get the most common rel_pos size
                            rel_pos_size_counts = {}
                            for size in model_rel_pos_sizes.values():
                                rel_pos_size_counts[size] = rel_pos_size_counts.get(size, 0) + 1
                            most_common_size = max(rel_pos_size_counts.items(), key=lambda x: x[1])[0]
                            model_patch_size = (most_common_size + 1) // 2
                            model_img_size = model_patch_size * 16
                            print(f"[DEBUG] Model rel_pos sizes: {set(model_rel_pos_sizes.values())}")
                            print(f"[DEBUG] Most common rel_pos size: {most_common_size} -> patch_size={model_patch_size}, image_size={model_img_size}")
                            print(f"[DEBUG] Using model image_size={model_img_size} (patch_size={model_patch_size})")
                        
                        # Load checkpoint and detect its size
                        print(f"[DEBUG] Loading checkpoint from: {checkpoint_to_use}")
                        checkpoint_state = torch.load(checkpoint_to_use, map_location='cpu', weights_only=False)
                        print(f"[DEBUG] Checkpoint has {len(checkpoint_state)} keys")
                        
                        # Filter out Adapter keys (SAM-Med2D specific, not in standard SAM)
                        adapter_keys = [k for k in checkpoint_state.keys() if 'Adapter' in k]
                        if adapter_keys:
                            print(f"[DEBUG] Filtering out {len(adapter_keys)} Adapter keys (SAM-Med2D specific)")
                            for key in adapter_keys:
                                del checkpoint_state[key]
                        
                        # Detect checkpoint size from pos_embed (most reliable)
                        checkpoint_img_size = None
                        checkpoint_patch_size = None
                        if 'image_encoder.pos_embed' in checkpoint_state:
                            pos_embed_shape = checkpoint_state['image_encoder.pos_embed'].shape
                            print(f"[DEBUG] Checkpoint pos_embed shape: {pos_embed_shape}")
                            if len(pos_embed_shape) == 4:
                                patch_h, patch_w = pos_embed_shape[1], pos_embed_shape[2]
                                checkpoint_patch_size = patch_h
                                checkpoint_img_size = patch_h * 16
                                print(f"[DEBUG] Checkpoint has image_size={checkpoint_img_size} (detected from pos_embed: {patch_h}x{patch_w} patches)")
                        
                        # Check rel_pos sizes in checkpoint
                        rel_pos_sizes = {}
                        for key in checkpoint_state.keys():
                            if 'rel_pos_h' in key or 'rel_pos_w' in key:
                                rel_pos_shape = checkpoint_state[key].shape
                                if len(rel_pos_shape) == 2:
                                    rel_pos_sizes[key] = rel_pos_shape[0]
                                    patch_size_from_rel = (rel_pos_shape[0] + 1) // 2
                                    img_size_from_rel = patch_size_from_rel * 16
                                    print(f"[DEBUG] Checkpoint {key} shape: {rel_pos_shape}, implies patch_size={patch_size_from_rel}, image_size={img_size_from_rel}")
                                    if checkpoint_img_size is None:
                                        checkpoint_patch_size = patch_size_from_rel
                                        checkpoint_img_size = img_size_from_rel
                        
                        # Check for size inconsistencies
                        if rel_pos_sizes:
                            unique_sizes = set(rel_pos_sizes.values())
                            if len(unique_sizes) > 1:
                                print(f"[WARNING] Checkpoint has inconsistent rel_pos sizes: {unique_sizes}")
                                print(f"[DEBUG] Will resize all rel_pos to match model size")
                        
                        # Resize if sizes don't match
                        if checkpoint_img_size is not None and model_img_size is not None and model_patch_size is not None:
                            if checkpoint_img_size != model_img_size:
                                print(f"[DEBUG] ========== RESIZING NEEDED ==========")
                                print(f"[DEBUG] Checkpoint: {checkpoint_img_size}x{checkpoint_img_size} ({checkpoint_patch_size}x{checkpoint_patch_size} patches)")
                                print(f"[DEBUG] Model: {model_img_size}x{model_img_size} ({model_patch_size}x{model_patch_size} patches)")
                                
                                # Resize pos_embed
                                if 'image_encoder.pos_embed' in checkpoint_state:
                                    old_pos_embed = checkpoint_state['image_encoder.pos_embed']
                                    old_h, old_w = old_pos_embed.shape[1], old_pos_embed.shape[2]
                                    print(f"[DEBUG] Resizing pos_embed from {old_h}x{old_w} to {model_patch_size}x{model_patch_size}...")
                                    old_pos_embed_2d = old_pos_embed.squeeze(0).permute(2, 0, 1)  # [C, H, W]
                                    new_pos_embed_2d = torch.nn.functional.interpolate(
                                        old_pos_embed_2d.unsqueeze(0),
                                        size=(model_patch_size, model_patch_size),
                                        mode='bilinear',
                                        align_corners=False
                                    ).squeeze(0)  # [C, H_new, W_new]
                                    new_pos_embed = new_pos_embed_2d.permute(1, 2, 0).unsqueeze(0)  # [1, H_new, W_new, C]
                                    checkpoint_state['image_encoder.pos_embed'] = new_pos_embed
                                    print(f"[DEBUG] pos_embed resized: {checkpoint_state['image_encoder.pos_embed'].shape}")
                                
                                # Resize ALL relative position embeddings to match model's expected size
                                # Use the actual model rel_pos sizes (may vary by block)
                                resized_count = 0
                                for key in list(checkpoint_state.keys()):
                                    if 'rel_pos_h' in key or 'rel_pos_w' in key:
                                        # Find corresponding model parameter to get target size
                                        # Key format: "image_encoder.blocks.X.attn.rel_pos_h/w"
                                        target_size = None
                                        for model_key in model_rel_pos_sizes.keys():
                                            # Extract block number from both keys
                                            # checkpoint: "image_encoder.blocks.0.attn.rel_pos_h"
                                            # model: "blocks.0.attn.rel_pos_h" or "image_encoder.blocks.0.attn.rel_pos_h"
                                            checkpoint_parts = key.split('.')
                                            model_parts = model_key.split('.')
                                            
                                            # Find block indices
                                            checkpoint_block_idx = None
                                            model_block_idx = None
                                            for i, part in enumerate(checkpoint_parts):
                                                if part == 'blocks' and i + 1 < len(checkpoint_parts):
                                                    checkpoint_block_idx = checkpoint_parts[i + 1]
                                                    break
                                            for i, part in enumerate(model_parts):
                                                if part == 'blocks' and i + 1 < len(model_parts):
                                                    model_block_idx = model_parts[i + 1]
                                                    break
                                            
                                            # Match block number and rel_pos type (h/w)
                                            if checkpoint_block_idx == model_block_idx:
                                                checkpoint_rel_type = 'rel_pos_h' if 'rel_pos_h' in key else 'rel_pos_w'
                                                model_rel_type = 'rel_pos_h' if 'rel_pos_h' in model_key else 'rel_pos_w'
                                                if checkpoint_rel_type == model_rel_type:
                                                    target_size = model_rel_pos_sizes[model_key]
                                                    break
                                        
                                        # Fallback to most common size if not found
                                        if target_size is None:
                                            target_size = most_common_size
                                            print(f"[DEBUG] Could not find matching model key for {key}, using most common size {target_size}")
                                        
                                        old_rel_pos = checkpoint_state[key]
                                        if len(old_rel_pos.shape) == 2:
                                            old_size = old_rel_pos.shape[0]
                                            if old_size != target_size:
                                                print(f"[DEBUG] Resizing {key} from [{old_size}] to [{target_size}]...")
                                                old_rel_pos_expanded = old_rel_pos.unsqueeze(0).unsqueeze(0)  # [1, 1, 2*H-1, head_dim]
                                                new_rel_pos_expanded = torch.nn.functional.interpolate(
                                                    old_rel_pos_expanded.permute(0, 3, 1, 2),  # [1, head_dim, 1, 2*H-1]
                                                    size=(1, target_size),
                                                    mode='bilinear',
                                                    align_corners=False
                                                ).permute(0, 2, 3, 1).squeeze(0).squeeze(0)  # [2*H_new-1, head_dim]
                                                checkpoint_state[key] = new_rel_pos_expanded
                                                print(f"[DEBUG] {key} resized: {checkpoint_state[key].shape}")
                                                resized_count += 1
                                            else:
                                                print(f"[DEBUG] {key} already has correct size [{target_size}], skipping")
                                print(f"[DEBUG] Resized {resized_count} relative position embedding(s)")
                            else:
                                print(f"[DEBUG] Checkpoint and model have matching image_size={checkpoint_img_size}, no resizing needed")
                        else:
                            print(f"[WARNING] Could not detect sizes - checkpoint_img_size={checkpoint_img_size}, model_img_size={model_img_size}, model_patch_size={model_patch_size}")
                            print(f"[DEBUG] Attempting to load as-is...")
                        
                        # Load with strict=False to handle any remaining missing/unexpected keys
                        print(f"[DEBUG] Attempting to load state_dict...")
                        try:
                            missing_keys, unexpected_keys = sam_model.load_state_dict(checkpoint_state, strict=False)
                            if missing_keys:
                                print(f"[DEBUG] Missing keys: {len(missing_keys)}")
                                for key in list(missing_keys)[:10]:
                                    print(f"[DEBUG]   - {key}")
                            if unexpected_keys:
                                print(f"[DEBUG] Unexpected keys: {len(unexpected_keys)}")
                                for key in list(unexpected_keys)[:10]:
                                    print(f"[DEBUG]   - {key}")
                            
                            encoder = sam_model.image_encoder
                            print(f"[DEBUG] ========== SUCCESSFULLY LOADED ==========")
                            print(f"[INFO] Successfully loaded SAM-Med2D {try_model_type} with size adaptation")
                            model_type = try_model_type
                            break
                        except RuntimeError as load_error:
                            print(f"[DEBUG] ========== LOAD FAILED ==========")
                            print(f"[DEBUG] Load error: {load_error}")
                            # Check if it's still a size mismatch
                            load_error_msg = str(load_error).lower()
                            if "size mismatch" in load_error_msg:
                                # Extract the mismatched keys
                                error_lines = str(load_error).split('\n')
                                size_mismatches = [line for line in error_lines if 'size mismatch' in line.lower()]
                                print(f"[DEBUG] Remaining size mismatches:")
                                for mismatch in size_mismatches[:10]:
                                    print(f"[DEBUG]   {mismatch}")
                            raise
                    
                    # If we detected a specific model type, don't try others
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
                    
                    # Check if it's a missing key error - only try next if we're trying multiple types
                    if "missing key" in error_msg:
                        if try_model_type != model_types_to_try[-1]:
                            print(f"[WARNING] Missing keys with '{try_model_type}', trying next model type...")
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
                        # Re-raise if it's not a handled error
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
        self.sam_med2d_image_size = sam_med2d_image_size  # Store SAM-Med2D's native image size
        self.target_image_size = image_size  # PRS-Med's target image size

        # Detect output channels by running a dummy forward pass with SAM-Med2D's native size
        encoder_channels = 1280  # sensible default
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, sam_med2d_image_size, sam_med2d_image_size).to(self.device)
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

        # Define preprocessing: resize to SAM-Med2D's native size (256x256), then normalize
        # This ensures compatibility with SAM-Med2D's trained weights
        self.transform = transforms.Compose([
            transforms.Resize((sam_med2d_image_size, sam_med2d_image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        # Project to 256 channels to match TinySAM output
        self.output_proj = nn.Conv2d(encoder_channels, 256, kernel_size=1).to(self.device)
        
        # Store target size for output resizing if needed
        self.target_image_size = image_size

    def forward(self, x):
        """
        Forward pass that wraps SAM-Med2D for PRS-Med compatibility.
        
        SAM-Med2D expects 256x256 input, but PRS-Med uses 1024x1024.
        This wrapper:
        1. Resizes input from 1024x1024 to 256x256 (SAM-Med2D's native size)
        2. Processes through SAM-Med2D encoder
        3. Projects to 256 channels and resizes to 16x16 to match TinySAM output format
        """
        if isinstance(x, list):
            # Handle PIL images or tensors in list
            processed = []
            for img in x:
                if isinstance(img, torch.Tensor):
                    # If tensor, resize and normalize
                    if img.dim() == 3:
                        img = img.unsqueeze(0)
                    # Resize to SAM-Med2D's native size
                    img = torch.nn.functional.interpolate(
                        img, 
                        size=(self.sam_med2d_image_size, self.sam_med2d_image_size),
                        mode='bilinear',
                        align_corners=False
                    )
                    # Normalize if needed
                    if img.max() > 2.0:
                        img = img / 255.0
                    mean = torch.tensor([0.485, 0.456, 0.406], device=img.device).view(1, 3, 1, 1)
                    std = torch.tensor([0.229, 0.224, 0.225], device=img.device).view(1, 3, 1, 1)
                    img = (img - mean) / std
                    processed.append(img.squeeze(0))
                else:
                    processed.append(self.transform(img))
            x = torch.stack(processed)
        elif isinstance(x, torch.Tensor):
            # Handle tensor input
            original_size = x.shape[-2:]
            
            # Resize to SAM-Med2D's native size (256x256) if needed
            if original_size != (self.sam_med2d_image_size, self.sam_med2d_image_size):
                x = torch.nn.functional.interpolate(
                    x,
                    size=(self.sam_med2d_image_size, self.sam_med2d_image_size),
                    mode='bilinear',
                    align_corners=False
                )
            
            # Normalize if needed (check if already normalized)
            if x.max() > 2.0:
                # Likely unnormalized [0, 255] range
                x = x / 255.0
                mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
                x = (x - mean) / std

        x = x.to(self.device)

        # Forward through SAM-Med2D encoder (expects 256x256 input)
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

        # Ensure output is 16x16 to match TinySAM output format
        # SAM-Med2D outputs 16x16 features for 256x256 input (256/16=16)
        # We need to maintain 16x16 for PRS-Med compatibility
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

