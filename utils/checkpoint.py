"""Checkpoint save/load utilities."""

import os
import time
import shutil
import torch
from datetime import datetime
from typing import Optional


def check_disk_space(path, required_gb=5.0):
    """Check if there's enough disk space available"""
    try:
        stat = shutil.disk_usage(path)
        free_gb = stat.free / (1024 ** 3)  # Convert to GB
        if free_gb < required_gb:
            print(f"WARNING: Low disk space: {free_gb:.2f} GB free (need at least {required_gb} GB)")
            return False
        return True
    except Exception as e:
        print(f"WARNING: Could not check disk space: {e}")
        return True  # Assume OK if we can't check


def save_checkpoint(epoch, model, optimizer, checkpoint_dir, is_best=False, max_retries=3):
    """Save complete model checkpoint with atomic writes and retry logic"""
    if checkpoint_dir is None:
        print("ERROR: checkpoint_dir is None, cannot save checkpoint")
        return False
    
    if not os.path.exists(checkpoint_dir):
        print(f"ERROR: Checkpoint directory does not exist: {checkpoint_dir}")
        try:
            os.makedirs(checkpoint_dir, exist_ok=True)
            print(f"Created checkpoint directory: {checkpoint_dir}")
        except Exception as e:
            print(f"ERROR: Failed to create checkpoint directory: {e}")
            return False
    
    # Check disk space before attempting to save
    if not check_disk_space(checkpoint_dir, required_gb=10.0):
        print("ERROR: Insufficient disk space to save checkpoint")
        return False
    
    # Unwrap DDP model if needed
    if hasattr(model, 'module'):
        model_to_save = model.module
    else:
        model_to_save = model
    
    if is_best:
        filename = f'best_model_epoch_{epoch+1}.pth'
    else:
        filename = f'checkpoint_epoch_{epoch+1}.pth'
    
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    temp_path = checkpoint_path + '.tmp'
    
    # Prepare checkpoint data
    try:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        print(f"ERROR: Failed to prepare checkpoint data: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Try saving with retries
    for attempt in range(max_retries):
        try:
            # Remove temp file if it exists from previous failed attempt
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
            
            # Save to temporary file first (atomic write)
            print(f"  Saving checkpoint (attempt {attempt + 1}/{max_retries})...")
            torch.save(checkpoint, temp_path)
            
            # Flush filesystem buffers
            import sys
            sys.stdout.flush()
            if hasattr(os, 'sync'):
                try:
                    os.sync()
                except:
                    pass
            
            # Verify temp file was written correctly
            if not os.path.exists(temp_path):
                raise RuntimeError(f"Temporary checkpoint file was not created: {temp_path}")
            
            temp_size = os.path.getsize(temp_path)
            if temp_size == 0:
                raise RuntimeError(f"Temporary checkpoint file is empty: {temp_path}")
            
            # Atomically rename temp file to final location
            # This ensures the checkpoint is either fully written or not present at all
            os.rename(temp_path, checkpoint_path)
            
            # Verify final file exists and has correct size
            if not os.path.exists(checkpoint_path):
                raise RuntimeError(f"Checkpoint file was not created after rename: {checkpoint_path}")
            
            final_size = os.path.getsize(checkpoint_path)
            if final_size != temp_size:
                raise RuntimeError(f"File size mismatch after rename: {final_size} != {temp_size}")
            
            # Success!
            file_size_mb = final_size / (1024 * 1024)  # Size in MB
            print(f"✓ Checkpoint saved to {checkpoint_path}")
            print(f"  File size: {file_size_mb:.2f} MB")
            return True
            
        except RuntimeError as e:
            error_msg = str(e)
            if "file write failed" in error_msg.lower() or "unexpected pos" in error_msg.lower():
                print(f"  Attempt {attempt + 1} failed: File write error - {error_msg}")
                if attempt < max_retries - 1:
                    print(f"  Retrying in 2 seconds...")
                    time.sleep(2)
                    continue
                else:
                    print(f"ERROR: Failed to save checkpoint after {max_retries} attempts")
                    print(f"  This might be due to:")
                    print(f"    - Insufficient disk space")
                    print(f"    - File system issues")
                    print(f"    - Network file system problems (if using network mount)")
                    print(f"    - Disk I/O errors")
                    return False
            else:
                raise  # Re-raise if it's a different error
                
        except Exception as e:
            print(f"ERROR: Failed to save checkpoint on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                print(f"  Retrying in 2 seconds...")
                time.sleep(2)
                continue
            else:
                print(f"ERROR: Failed to save checkpoint after {max_retries} attempts")
                import traceback
                traceback.print_exc()
                return False
    
    # Clean up temp file if it still exists
    temp_path = checkpoint_path + '.tmp'
    if os.path.exists(temp_path):
        try:
            os.remove(temp_path)
        except:
            pass
    
    return False


def load_checkpoint(checkpoint_path: str, model, optimizer: Optional[torch.optim.Optimizer] = None, 
                    device: Optional[torch.device] = None):
    """
    Load model checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load state dict into
        optimizer: Optional optimizer to load state dict into
        device: Device to load checkpoint on
    
    Returns:
        Dictionary with checkpoint metadata (epoch, timestamp, etc.)
    """
    if device is None:
        device = torch.device('cpu')
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        epoch = checkpoint.get('epoch', 0)
        print(f"Loaded checkpoint from epoch {epoch}")
    else:
        state_dict = checkpoint
        epoch = 0
        print("Loaded checkpoint (no epoch info found)")
    
    # Load state dict
    if hasattr(model, 'module'):  # DDP wrapped
        model.module.load_state_dict(state_dict, strict=False)
    else:
        model.load_state_dict(state_dict, strict=False)
    
    # Load optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("✓ Optimizer state loaded")
    
    print("✓ Model loaded successfully")
    
    return {
        'epoch': epoch,
        'timestamp': checkpoint.get('timestamp', 'unknown'),
        'checkpoint_path': checkpoint_path
    }

