import os
import shutil
from pathlib import Path
from tqdm import tqdm
from utils import logging
logging = logging.get_logger(__name__)


def _reorganize_directory(parent_dir: Path, split_dirs: list, item_type: str):
    """
    Reorganize files from split subdirectories (train/test/val) to parent directory.
    
    Args:
        parent_dir: Parent directory (e.g., images/ or masks/)
        split_dirs: List of split directory names to process (e.g., ["train", "test", "val"])
        item_type: Type of items being moved (e.g., "images" or "masks") for logging
    """
    logging.info(f"Reorganizing {item_type}...")
    
    for split_dir in split_dirs:
        split_path = parent_dir / split_dir
        if not split_path.exists():
            continue
        
        # Find all PNG files in the split directory
        files = list(split_path.glob("*.png"))
        
        # Move files to parent directory
        for file in tqdm(files, desc=f"Moving {split_dir} {item_type}"):
            dest = parent_dir / file.name
            if dest.exists():
                logging.warning(f"Warning: {dest} already exists, skipping {file}")
            else:
                shutil.move(str(file), str(dest))
        
        # Remove split directory (check if empty first)
        try:
            remaining_files = list(split_path.iterdir())
            if len(remaining_files) == 0:
                split_path.rmdir()
                logging.info(f"  Removed empty directory: {split_path}")
            else:
                logging.warning(f"  Warning: {split_path} is not empty, contains: {[f.name for f in remaining_files]}")
                logging.info(f"  Attempting to remove anyway...")
                # Try to remove any remaining files (like hidden files)
                for item in remaining_files:
                    if item.is_file():
                        item.unlink()
                        logging.info(f"    Removed: {item.name}")
                # Try rmdir again
                try:
                    split_path.rmdir()
                    logging.info(f"  Removed directory: {split_path}")
                except OSError as e:
                    logging.error(f"  Error removing {split_path}: {e}")
        except OSError as e:
            logging.error(f"  Error checking/removing {split_path}: {e}")


def reorganize_prostate_dataset(prostate_root: str):
    """
    Reorganize prostate dataset by moving all images and masks from train/test subdirectories
    to the parent images/ and masks/ directories.
    
    Args:
        prostate_root: Root directory of the prostate dataset
    """
    prostate_path = Path(prostate_root)
    
    if not prostate_path.exists():
        raise ValueError(f"Prostate dataset root not found: {prostate_root}")
    
    images_dir = prostate_path / "images"
    masks_dir = prostate_path / "masks"
    
    if not images_dir.exists() or not masks_dir.exists():
        raise ValueError(f"Expected 'images' and 'masks' directories in {prostate_root}")
    
    split_dirs = ["train", "test", "val"]
    
    # Process images and masks using the same helper function
    _reorganize_directory(images_dir, split_dirs, "images")
    _reorganize_directory(masks_dir, split_dirs, "masks")
    
    logging.info("✓ Prostate dataset reorganization complete!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Reorganize prostate dataset structure")
    parser.add_argument(
        "--prostate_root",
        type=str,
        required=True,
        help="Root directory of the prostate dataset (should contain 'images' and 'masks' folders)"
    )
    
    args = parser.parse_args()
    reorganize_prostate_dataset(args.prostate_root)

