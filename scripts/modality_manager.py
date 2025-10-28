#!/usr/bin/env python3
"""
Utility script for managing PRS-Med modality configurations.
Allows easy addition/modification of dataset-to-modality mappings.
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from configs.modality_config import (
    MODALITY_MAPPINGS, STANDARD_MODALITIES, 
    get_modality_mapping, get_all_modalities,
    get_datasets_by_modality, get_modality_stats
)

def list_modalities():
    """List all available modalities and their datasets."""
    print("=== Available Modalities ===")
    stats = get_modality_stats()
    
    for modality in STANDARD_MODALITIES.keys():
        datasets = get_datasets_by_modality(modality)
        info = STANDARD_MODALITIES[modality]
        
        print(f"\n{modality}:")
        print(f"  Description: {info['description']}")
        print(f"  Typical Use: {info['typical_use']}")
        print(f"  Color Space: {info['color_space']}")
        print(f"  Datasets ({len(datasets)}): {', '.join(datasets) if datasets else 'None'}")

def list_datasets():
    """List all datasets and their modality mappings."""
    print("=== Dataset to Modality Mapping ===")
    
    for dataset_name, mapping in MODALITY_MAPPINGS.items():
        print(f"\n{dataset_name}:")
        print(f"  Modality: {mapping.modality_type}")
        print(f"  Description: {mapping.description}")
        print(f"  Image Type: {mapping.image_type}")
        print(f"  Expected Channels: {mapping.expected_channels}")

def add_dataset_mapping(dataset_name, modality_type, description, image_type, channels=3):
    """Add a new dataset-to-modality mapping."""
    if modality_type not in STANDARD_MODALITIES:
        print(f"Error: Modality '{modality_type}' not in standard modalities: {list(STANDARD_MODALITIES.keys())}")
        return False
    
    if dataset_name in MODALITY_MAPPINGS:
        print(f"Warning: Dataset '{dataset_name}' already exists. Use --update to modify.")
        return False
    
    # This would require updating the actual config file
    print(f"Would add mapping: {dataset_name} → {modality_type}")
    print(f"  Description: {description}")
    print(f"  Image Type: {image_type}")
    print(f"  Channels: {channels}")
    print("Note: This is a preview. Modify configs/modality_config.py directly to add mappings.")
    
    return True

def validate_data_structure(data_dir="data/mmrs"):
    """Validate that data structure matches modality mappings."""
    print("=== Validating Data Structure ===")
    
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"Error: Data directory '{data_dir}' does not exist")
        return False
    
    missing_datasets = []
    extra_datasets = []
    
    # Check for missing datasets
    for dataset_name in MODALITY_MAPPINGS.keys():
        dataset_path = data_path / dataset_name
        if not dataset_path.exists():
            missing_datasets.append(dataset_name)
    
    # Check for extra datasets
    for item in data_path.iterdir():
        if item.is_dir() and item.name not in MODALITY_MAPPINGS:
            extra_datasets.append(item.name)
    
    print(f"Total mapped datasets: {len(MODALITY_MAPPINGS)}")
    print(f"Missing datasets: {len(missing_datasets)}")
    print(f"Extra datasets: {len(extra_datasets)}")
    
    if missing_datasets:
        print(f"\nMissing datasets: {', '.join(missing_datasets)}")
    
    if extra_datasets:
        print(f"\nExtra datasets (not in mappings): {', '.join(extra_datasets)}")
        print("Consider adding mappings for these datasets.")
    
    return len(missing_datasets) == 0

def show_training_configs():
    """Show available training configurations."""
    from configs.training_config import CONFIGS
    
    print("=== Available Training Configurations ===")
    
    for config_name, config in CONFIGS.items():
        print(f"\n{config_name}:")
        print(f"  Modalities: {config.modalities}")
        print(f"  Epochs: {config.epochs}")
        print(f"  Batch Size: {config.batch_size}")
        print(f"  Learning Rate: {config.learning_rate}")

def main():
    parser = argparse.ArgumentParser(description="PRS-Med Modality Configuration Manager")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # List modalities
    subparsers.add_parser('list-modalities', help='List all available modalities')
    
    # List datasets
    subparsers.add_parser('list-datasets', help='List all dataset mappings')
    
    # Add dataset mapping
    add_parser = subparsers.add_parser('add-dataset', help='Add new dataset mapping')
    add_parser.add_argument('dataset_name', help='Name of the dataset folder')
    add_parser.add_argument('modality_type', help='Modality type (CT, MRI, X-ray, etc.)')
    add_parser.add_argument('description', help='Description of the dataset')
    add_parser.add_argument('image_type', help='Image type for templates')
    add_parser.add_argument('--channels', type=int, default=3, help='Expected channels (default: 3)')
    
    # Validate data structure
    validate_parser = subparsers.add_parser('validate', help='Validate data structure')
    validate_parser.add_argument('--data-dir', default='data/mmrs', help='Data directory path')
    
    # Show training configs
    subparsers.add_parser('list-configs', help='List training configurations')
    
    args = parser.parse_args()
    
    if args.command == 'list-modalities':
        list_modalities()
    elif args.command == 'list-datasets':
        list_datasets()
    elif args.command == 'add-dataset':
        add_dataset_mapping(args.dataset_name, args.modality_type, 
                           args.description, args.image_type, args.channels)
    elif args.command == 'validate':
        validate_data_structure(args.data_dir)
    elif args.command == 'list-configs':
        show_training_configs()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
