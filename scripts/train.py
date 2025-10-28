#!/usr/bin/env python3
"""
Training script for PRS-Med model.
Supports multi-modal training across different medical imaging datasets.
"""

import argparse
import json
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from models.prs_med_model import PRSMedModel
from data_pipeline.dataset_mmrs import MMRSDataset
from training.train_loop import Trainer
from training.losses import SegmentationLoss, TextLoss
from training.optimizer import build_optimizer

def create_data_loaders(data_dir: Path, modalities: list, batch_size: int = 8, num_workers: int = 4):
    """Create data loaders for all modalities."""
    
    datasets = {}
    loaders = {}
    
    for modality in modalities:
        modality_dir = data_dir / modality
        if not modality_dir.exists():
            print(f"Warning: {modality} not found, skipping...")
            continue
            
        # Create datasets for each split
        for split in ["train", "val", "test"]:
            split_dir = modality_dir / split
            if split_dir.exists() and (split_dir / "images").exists():
                dataset = MMRSDataset(split_dir, split=split, img_size=224)
                datasets[f"{modality}_{split}"] = dataset
                
                loader = DataLoader(
                    dataset, 
                    batch_size=batch_size, 
                    shuffle=(split == "train"),
                    num_workers=num_workers,
                    collate_fn=custom_collate_fn
                )
                loaders[f"{modality}_{split}"] = loader
                
                print(f"Created {split} loader for {modality}: {len(dataset)} samples")
    
    return datasets, loaders

def custom_collate_fn(batch):
    """Custom collate function for batching."""
    images = torch.stack([item["image"] for item in batch])
    masks = torch.stack([item["mask"] for item in batch])
    questions = [item["question"] for item in batch]
    answers = [item["answer"] for item in batch]
    answer_ids = torch.stack([item["answer_ids"] for item in batch])
    
    return {
        "image": images,
        "mask": masks,
        "question": questions,
        "answer": answers,
        "answer_ids": answer_ids
    }

def main():
    parser = argparse.ArgumentParser(description="Train PRS-Med model")
    parser.add_argument("--data_dir", type=str, default="data/mmrs", help="Data directory")
    parser.add_argument("--modalities", nargs="+", 
                       default=["brain_tumors_ct_scan", "breast_tumors_ct_scan", "dental_xray", 
                               "lung_CT", "lung_Xray", "polyp_endoscopy", "skin_rgbimage"],
                       help="Modalities to train on")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="mps", help="Device (mps/cuda/cpu)")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--save_every", type=int, default=10, help="Save checkpoint every N epochs")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    
    # Create model
    model = PRSMedModel(base_llava_model="microsoft/DialoGPT-medium")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        lambda_seg=1.0,
        lambda_text=1.0,
        lr=args.lr,
        device=args.device
    )
    
    # Create data loaders
    data_dir = Path(args.data_dir)
    datasets, loaders = create_data_loaders(data_dir, args.modalities, args.batch_size)
    
    # Combine all training data
    train_loaders = [loader for key, loader in loaders.items() if "train" in key]
    val_loaders = [loader for key, loader in loaders.items() if "val" in key]
    
    if not train_loaders:
        print("No training data found!")
        return
    
    print(f"Training on {len(train_loaders)} modalities")
    print(f"Validation on {len(val_loaders)} modalities")
    
    # Training loop
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        
        # Train on all modalities
        total_loss = 0.0
        total_batches = 0
        
        for loader in train_loaders:
            trainer.model.train()
            for batch in loader:
                # Move batch to device
                batch = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = trainer.model(batch["image"], batch["question"])
                
                # Compute losses
                seg_loss = trainer.seg_loss(outputs["mask"], batch["mask"])
                text_loss = trainer.text_loss(outputs["logits"], batch["answer_ids"])
                loss = trainer.lambda_seg * seg_loss + trainer.lambda_text * text_loss
                
                # Backward pass
                trainer.optimizer.zero_grad()
                loss.backward()
                trainer.optimizer.step()
                
                total_loss += loss.item()
                total_batches += 1
        
        avg_loss = total_loss / total_batches if total_batches > 0 else 0
        print(f"Average Loss: {avg_loss:.4f}")
        
        # Validation
        if val_loaders and epoch % 5 == 0:
            trainer.model.eval()
            val_loss = 0.0
            val_batches = 0
            
            with torch.no_grad():
                for loader in val_loaders:
                    for batch in loader:
                        batch = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v 
                                for k, v in batch.items()}
                        
                        outputs = trainer.model(batch["image"], batch["question"])
                        seg_loss = trainer.seg_loss(outputs["mask"], batch["mask"])
                        text_loss = trainer.text_loss(outputs["logits"], batch["answer_ids"])
                        loss = trainer.lambda_seg * seg_loss + trainer.lambda_text * text_loss
                        
                        val_loss += loss.item()
                        val_batches += 1
            
            avg_val_loss = val_loss / val_batches if val_batches > 0 else 0
            print(f"Validation Loss: {avg_val_loss:.4f}")
        
        # Save checkpoint
        if epoch % args.save_every == 0:
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': trainer.model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")
    
    # Save final model
    final_path = output_dir / "final_model.pth"
    torch.save(trainer.model.state_dict(), final_path)
    print(f"Training complete! Final model saved: {final_path}")

if __name__ == "__main__":
    import torch
    main()
