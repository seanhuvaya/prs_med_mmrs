#!/usr/bin/env python3
"""
Main entry point for PRS-Med training and inference.
"""

import argparse
import torch
from pathlib import Path

from configs.training_config import get_config, TrainingConfig
from models.prs_med_model import PRSMedModel
from scripts.train import main as train_main
from scripts.prepare_data import main as prepare_data_main

def prepare_data(args):
    """Prepare data for training."""
    print("Preparing data...")
    import sys
    sys.argv = ['prepare_data.py', '--raw_dir', 'data/raw', '--output_dir', 'data/mmrs']
    prepare_data_main()

def train(args):
    """Train the PRS-Med model."""
    print("Starting training...")
    
    # Get the configuration
    from configs.training_config import get_config
    config = get_config(args.config)
    
    # Override config with command line arguments
    if hasattr(args, 'epochs') and args.epochs:
        config.epochs = args.epochs
    if hasattr(args, 'batch_size') and args.batch_size:
        config.batch_size = args.batch_size
    if hasattr(args, 'device') and args.device:
        config.device = args.device
    
    # Pass the config to the training script
    import sys
    sys.argv = ['train.py', '--data_dir', 'data/mmrs', '--modalities'] + config.modalities + [
        '--epochs', str(config.epochs), 
        '--batch_size', str(config.batch_size), 
        '--device', config.device
    ]
    train_main()

def inference(args):
    """Run inference on test data."""
    print("Running inference...")
    
    # Load model
    model = PRSMedModel()
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    
    # TODO: Implement inference pipeline
    print("Inference pipeline not yet implemented")

def main():
    parser = argparse.ArgumentParser(description="PRS-Med: Position Reasoning Segmentation")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Prepare data command
    prepare_parser = subparsers.add_parser('prepare', help='Prepare data for training')
    prepare_parser.set_defaults(func=prepare_data)
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--config', type=str, default='multi_modality', 
                            help='Training configuration')
    train_parser.add_argument('--data_dir', type=str, default='data/mmrs',
                            help='Data directory')
    train_parser.add_argument('--output_dir', type=str, default='outputs',
                            help='Output directory')
    train_parser.add_argument('--epochs', type=int, default=50,
                            help='Number of epochs')
    train_parser.add_argument('--batch_size', type=int, default=8,
                            help='Batch size')
    train_parser.add_argument('--device', type=str, default='mps',
                            help='Device (mps/cuda/cpu)')
    train_parser.set_defaults(func=train)
    
    # Inference command
    infer_parser = subparsers.add_parser('infer', help='Run inference')
    infer_parser.add_argument('--checkpoint', type=str, required=True,
                            help='Path to model checkpoint')
    infer_parser.add_argument('--data_dir', type=str, default='data/mmrs/test',
                            help='Test data directory')
    infer_parser.add_argument('--output_dir', type=str, default='inference_outputs',
                            help='Output directory')
    infer_parser.set_defaults(func=inference)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    args.func(args)

if __name__ == "__main__":
    main()
