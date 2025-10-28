#!/usr/bin/env python3
"""
Data preprocessing script for PRS-Med MMRS dataset.
Organizes raw data into the expected structure for training.
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List
import argparse

def organize_dataset(raw_dir: Path, output_dir: Path, modalities: List[str]):
    """Organize raw data into MMRS format."""
    
    for modality in modalities:
        print(f"Processing {modality}...")
        
        # Create output directories
        modality_dir = output_dir / modality
        modality_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        for split in ["train", "val", "test"]:
            for data_type in ["images", "masks"]:
                (modality_dir / split / data_type).mkdir(parents=True, exist_ok=True)
        
        # Process each split
        raw_modality_dir = raw_dir / modality
        if not raw_modality_dir.exists():
            print(f"Warning: {modality} not found in raw data")
            continue
            
        for split in ["train", "val", "test"]:
            img_dir = raw_modality_dir / f"{split}_images"
            mask_dir = raw_modality_dir / f"{split}_masks"
            
            if img_dir.exists() and mask_dir.exists():
                # Copy images
                for img_file in img_dir.glob("*"):
                    if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        shutil.copy2(img_file, modality_dir / split / "images" / img_file.name)
                
                # Copy masks
                for mask_file in mask_dir.glob("*"):
                    if mask_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        shutil.copy2(mask_file, modality_dir / split / "masks" / mask_file.name)
                
                print(f"  {split}: {len(list(img_dir.glob('*')))} images, {len(list(mask_dir.glob('*')))} masks")

def main():
    parser = argparse.ArgumentParser(description="Prepare MMRS dataset")
    parser.add_argument("--raw_dir", type=str, default="data/raw", help="Raw data directory")
    parser.add_argument("--output_dir", type=str, default="data/mmrs", help="Output directory")
    parser.add_argument("--modalities", nargs="+", 
                       default=["brain_tumors_ct_scan", "breast_tumors_ct_scan", "dental_xray", 
                               "lung_CT", "lung_Xray", "polyp_endoscopy", "skin_rgbimage"],
                       help="Modalities to process")
    
    args = parser.parse_args()
    
    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    
    print(f"Organizing data from {raw_dir} to {output_dir}")
    organize_dataset(raw_dir, output_dir, args.modalities)
    print("Data preparation complete!")

if __name__ == "__main__":
    main()
