"""
Main script to prepare both prostate and head_and_neck datasets.

This script orchestrates the reorganization and conversion of both datasets.
"""

import argparse
import sys
from pathlib import Path

# Add data directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from reorganize_prostate import reorganize_prostate_dataset
from convert_head_neck_npz import convert_head_neck_dataset
from reorganize_to_split_structure import reorganize_dataset_to_splits


def main():
    parser = argparse.ArgumentParser(
        description="Prepare prostate and head_and_neck datasets for PRS-Med"
    )
    parser.add_argument(
        "--prostate_root",
        type=str,
        default=None,
        help="Root directory of the prostate dataset"
    )
    parser.add_argument(
        "--head_neck_root",
        type=str,
        default=None,
        help="Root directory of the head_and_neck dataset"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Root directory containing both datasets (prostate/ and head_and_neck/ subdirectories)"
    )
    parser.add_argument(
        "--remove_npz",
        action="store_true",
        help="Remove original .npz files after conversion (for head_and_neck)"
    )
    
    args = parser.parse_args()
    
    # If data_root is provided, use it to find subdirectories
    if args.data_root:
        data_root = Path(args.data_root)
        if not args.prostate_root:
            prostate_path = data_root / "prostate"
            if prostate_path.exists():
                args.prostate_root = str(prostate_path)
        if not args.head_neck_root:
            head_neck_path = data_root / "head_and_neck"
            if head_neck_path.exists():
                args.head_neck_root = str(head_neck_path)
    
    # Process prostate dataset
    if args.prostate_root:
        print("=" * 60)
        print("Processing Prostate Dataset")
        print("=" * 60)
        try:
            reorganize_prostate_dataset(args.prostate_root)
        except Exception as e:
            print(f"Error processing prostate dataset: {e}")
    else:
        print("Skipping prostate dataset (--prostate_root not provided)")
    
    # Process head_and_neck dataset
    if args.head_neck_root:
        print("\n" + "=" * 60)
        print("Processing Head and Neck Dataset")
        print("=" * 60)
        try:
            convert_head_neck_dataset(args.head_neck_root, remove_npz=args.remove_npz)
        except Exception as e:
            print(f"Error processing head_and_neck dataset: {e}")
    else:
        print("Skipping head_and_neck dataset (--head_neck_root not provided)")
    
    print("\n" + "=" * 60)
    print("Dataset preparation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

