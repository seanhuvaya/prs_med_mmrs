"""
Utility script to convert head_and_neck .npz files to individual PNG images and masks per channel.

Each .npz file contains 64 channels. This script extracts each channel as a separate PNG file.

Before:
head_and_neck/
  images/
    example.npz  (64 channels)
  masks/
    example.npz  (64 channels)

After:
head_and_neck/
  images/
    example_channel_00.png
    example_channel_01.png
    ...
    example_channel_63.png
  masks/
    example_channel_00.png
    example_channel_01.png
    ...
    example_channel_63.png
"""

import os
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from utils import logging
logging = logging.get_logger(__name__)


def convert_npz_to_images(npz_path: Path, output_dir: Path, prefix: str = ""):
    """
    Convert a .npz file to individual PNG images, one per channel.
    
    Args:
        npz_path: Path to the .npz file
        output_dir: Directory to save the PNG images
        prefix: Optional prefix for output filenames
    """
    # Load .npz file
    data = np.load(npz_path)
    
    # Get the array (assuming single key or first key)
    keys = list(data.keys())
    if len(keys) == 0:
        raise ValueError(f"No data found in {npz_path}")
    
    # Use first key (or 'arr_0' if it exists)
    array_key = 'arr_0' if 'arr_0' in keys else keys[0]
    array = data[array_key]
    
    # Handle different array shapes
    if array.ndim == 2:
        # Single channel, treat as channel 0
        channels = [array]
    elif array.ndim == 3:
        # Multiple channels: (channels, height, width) or (height, width, channels)
        if array.shape[0] < array.shape[2]:
            # Likely (channels, height, width)
            channels = [array[i] for i in range(array.shape[0])]
        else:
            # Likely (height, width, channels)
            channels = [array[:, :, i] for i in range(array.shape[2])]
    elif array.ndim == 4:
        # Batch dimension: (batch, channels, height, width) or (batch, height, width, channels)
        # Take first batch item
        if array.shape[1] < array.shape[3]:
            channels = [array[0, i] for i in range(array.shape[1])]
        else:
            channels = [array[0, :, :, i] for i in range(array.shape[3])]
    else:
        raise ValueError(f"Unsupported array shape: {array.shape}")
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get base filename without extension
    base_name = npz_path.stem
    
    # Save each channel as PNG
    for channel_idx, channel_data in enumerate(channels):
        # Normalize to 0-255 range
        channel_min = channel_data.min()
        channel_max = channel_data.max()
        
        if channel_max > channel_min:
            normalized = ((channel_data - channel_min) / (channel_max - channel_min) * 255).astype(np.uint8)
        else:
            normalized = np.zeros_like(channel_data, dtype=np.uint8)
        
        # Create PIL Image and save
        img = Image.fromarray(normalized, mode='L')  # Grayscale
        
        # Construct output filename
        if prefix:
            output_name = f"{prefix}_{base_name}_channel_{channel_idx:02d}.png"
        else:
            output_name = f"{base_name}_channel_{channel_idx:02d}.png"
        
        output_path = output_dir / output_name
        img.save(output_path)
    
    return len(channels)


def _convert_directory_npz_files(directory: Path, item_type: str, remove_npz: bool = False):
    """
    Convert all .npz files in a directory to individual PNG images per channel.
    
    Args:
        directory: Directory containing .npz files
        item_type: Type of items being converted (e.g., "images" or "masks") for logging
        remove_npz: If True, remove original .npz files after conversion
    """
    if not directory.exists():
        logging.warning(f"{item_type.capitalize()} directory not found: {directory}")
        return
    
    logging.info(f"Converting {item_type} .npz files...")
    npz_files = list(directory.glob("*.npz"))
    
    if len(npz_files) == 0:
        logging.warning(f"No .npz files found in {directory}")
    else:
        for npz_file in tqdm(npz_files, desc=f"Converting {item_type}"):
            try:
                num_channels = convert_npz_to_images(npz_file, directory, prefix="")
                logging.info(f"  Converted {npz_file.name}: {num_channels} channels")
                if remove_npz:
                    npz_file.unlink()
                    logging.info(f"  Removed original .npz file: {npz_file.name}")
            except Exception as e:
                logging.error(f"  Error converting {npz_file}: {e}")


def convert_head_neck_dataset(head_neck_root: str, remove_npz: bool = False):
    """
    Convert all .npz files in head_and_neck dataset to individual PNG images per channel.
    
    Args:
        head_neck_root: Root directory of the head_and_neck dataset
        remove_npz: If True, remove original .npz files after conversion
    """
    head_neck_path = Path(head_neck_root)
    
    if not head_neck_path.exists():
        raise ValueError(f"Head and neck dataset root not found: {head_neck_root}")
    
    images_dir = head_neck_path / "images"
    masks_dir = head_neck_path / "masks"
    
    if not images_dir.exists():
        raise ValueError(f"Expected 'images' directory in {head_neck_root}")
    
    # Convert images and masks using the same helper function
    _convert_directory_npz_files(images_dir, "images", remove_npz)
    _convert_directory_npz_files(masks_dir, "masks", remove_npz)
    
    logging.info("✓ Head and neck dataset conversion complete!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert head_and_neck .npz files to PNG images per channel")
    parser.add_argument(
        "--head_neck_root",
        type=str,
        required=True,
        help="Root directory of the head_and_neck dataset (should contain 'images' and optionally 'masks' folders)"
    )
    parser.add_argument(
        "--remove_npz",
        action="store_true",
        help="Remove original .npz files after conversion (default: keep them)"
    )
    
    args = parser.parse_args()
    convert_head_neck_dataset(args.head_neck_root, remove_npz=args.remove_npz)

