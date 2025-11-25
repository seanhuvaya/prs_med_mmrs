import os
import shutil
from pathlib import Path
from typing import Optional, Dict, List
import pandas as pd
from tqdm import tqdm

from utils import logging
from sklearn.model_selection import train_test_split
logger = logging.get_logger(__name__)


def _reorganize_directory_to_splits(
    source_dir: Path,
    target_base: Path,
    split_mapping: Dict[str, str],
    item_type: str
):
    """
    Reorganize files from a single directory into split-specific directories.
    
    Args:
        source_dir: Source directory containing files to reorganize
        target_base: Base directory where split directories will be created
        split_mapping: Dictionary mapping filenames to split names (train/test/val)
        item_type: Type of items (e.g., "images" or "masks") for logging
    """
    if not source_dir.exists():
        logger.warning(f"{item_type.capitalize()} directory not found: {source_dir}")
        return
    
    # Get all files in source directory
    files = list(source_dir.glob("*.png"))
    
    if len(files) == 0:
        logger.warning(f"No PNG files found in {source_dir}")
        return
    
    # Count files per split
    split_counts = {"train": 0, "test": 0, "val": 0}
    
    # Process each file
    for file in tqdm(files, desc=f"Reorganizing {item_type}"):
        filename = file.name
        split = split_mapping.get(filename, "train")  # Default to train if not in mapping
        
        if split not in ["train", "test", "val"]:
            logger.warning(f"Unknown split '{split}' for {filename}, defaulting to 'train'")
            split = "train"
        
        # Create target directory
        target_dir = target_base / f"{split}_{item_type}"
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Move file to target directory
        target_path = target_dir / filename
        if target_path.exists():
            logger.warning(f"Target file already exists: {target_path}, skipping {file}")
        else:
            shutil.move(str(file), str(target_path))
            split_counts[split] += 1
    
    # Log summary
    logger.info(f"  {item_type.capitalize()} reorganization complete:")
    for split, count in split_counts.items():
        logger.info(f"    {split}: {count} files")


def reorganize_dataset_to_splits(
    dataset_root: str,
    split_mapping: Optional[Dict[str, str]] = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    seed: int = 42
):
    """
    Reorganize a dataset into train/test/val split structure.
    
    Args:
        dataset_root: Root directory of the dataset
        split_mapping: Optional dictionary mapping filenames to split names.
                       If None, will use sklearn train_test_split.
        train_ratio: Ratio for training split (default: 0.7)
        val_ratio: Ratio for validation split (default: 0.1)
        test_ratio: Ratio for test split (default: 0.2)
        seed: Random seed for reproducible splits (default: 42)
    """
    dataset_path = Path(dataset_root)
    
    if not dataset_path.exists():
        raise ValueError(f"Dataset root not found: {dataset_root}")
    
    images_dir = dataset_path / "images"
    masks_dir = dataset_path / "masks"
    
    if not images_dir.exists():
        raise ValueError(f"Expected 'images' directory in {dataset_root}")
    
    # Validate ratios sum to 1.0
    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        logger.warning(f"Ratios sum to {ratio_sum}, not 1.0. Normalizing...")
        train_ratio = train_ratio / ratio_sum
        val_ratio = val_ratio / ratio_sum
        test_ratio = test_ratio / ratio_sum
    
    # Use sklearn for splitting
    
    
    logger.info(f"Creating splits (seed={seed}): train={train_ratio:.1%}, val={val_ratio:.1%}, test={test_ratio:.1%}")
    
    image_files = sorted(list(images_dir.glob("*.png")))  # Sort for reproducibility
    if len(image_files) == 0:
        raise ValueError(f"No PNG files found in {images_dir}")
    
    n_total = len(image_files)
    
    # First split: separate train from (val + test)
    # train_ratio of total goes to train, (1 - train_ratio) goes to val+test
    train_files, val_test_files = train_test_split(
        image_files,
        test_size=(1 - train_ratio),
        random_state=seed,
        shuffle=True
    )
    
    # Second split: separate val from test
    # From the val_test portion, val_ratio/(val_ratio + test_ratio) goes to val
    val_test_ratio = val_ratio + test_ratio
    val_size_in_val_test = val_ratio / val_test_ratio if val_test_ratio > 0 else 0
    
    if len(val_test_files) > 0:
        val_files, test_files = train_test_split(
            val_test_files,
            test_size=(1 - val_size_in_val_test),
            random_state=seed,
            shuffle=True
        )
    else:
        val_files = []
        test_files = []
    
    # Create split mapping
    split_mapping = {}
    for img_file in train_files:
        split_mapping[img_file.name] = "train"
    for img_file in val_files:
        split_mapping[img_file.name] = "val"
    for img_file in test_files:
        split_mapping[img_file.name] = "test"
    
    n_train = len(train_files)
    n_val = len(val_files)
    n_test = len(test_files)
    
    logger.info(f"Split created: {n_train} train ({n_train/n_total:.1%}), {n_val} val ({n_val/n_total:.1%}), {n_test} test ({n_test/n_total:.1%})")
    
    # Reorganize images and masks
    _reorganize_directory_to_splits(images_dir, dataset_path, split_mapping, "images")
    _reorganize_directory_to_splits(masks_dir, dataset_path, split_mapping, "masks")
    
    # Remove empty original directories
    try:
        if images_dir.exists() and len(list(images_dir.iterdir())) == 0:
            images_dir.rmdir()
            logger.info(f"Removed empty directory: {images_dir}")
    except OSError:
        pass
    
    try:
        if masks_dir.exists() and len(list(masks_dir.iterdir())) == 0:
            masks_dir.rmdir()
            logger.info(f"Removed empty directory: {masks_dir}")
    except OSError:
        pass
    
    logger.info("✓ Dataset reorganization complete!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Reorganize dataset into train/test/val split structure"
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        required=True,
        help="Root directory of the dataset (should contain 'images' and 'masks' folders)"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.7,
        help="Ratio for training split (used if no CSV provided, default: 0.7)"
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Ratio for validation split (used if no CSV provided, default: 0.1)"
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.2,
        help="Ratio for test split (used if no CSV provided, default: 0.2)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splitting (default: 42)"
    )
    
    args = parser.parse_args()
    
    reorganize_dataset_to_splits(
        dataset_root=args.dataset_root,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )

