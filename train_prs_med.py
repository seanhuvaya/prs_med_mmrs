import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
from datetime import datetime

# Import your custom modules
from data import PRSMedDataLoader
from models.mllm.llava_med_mllm import LLaVAMedMLLM
from models.decoder.fusion_module import PromptMaskFusionModule
from models.decoder.mask_prediction_module import MaskPredictionModule
from models.loss.objective_function import PRSMedLoss

def parse_args():
    parser = argparse.ArgumentParser(description='PRS-Med Multi-Modal Medical Training')
    parser.add_argument('--data_root', type=str, required=True,
                       help='Root directory of the dataset (should contain annotations/ and images_and_masks/)')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                       help='Directory to save checkpoints')
    return parser.parse_args()

def save_checkpoint(epoch, mllm, fusion_module, decoder, optimizer, checkpoint_dir, is_best=False):
    """Save training checkpoint"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'mllm_state_dict': mllm.state_dict(),
        'fusion_state_dict': fusion_module.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'timestamp': datetime.now().isoformat()
    }
    
    if is_best:
        filename = f'best_model_epoch_{epoch+1}.pth'
    else:
        filename = f'checkpoint_epoch_{epoch+1}.pth'
    
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")
    
    # Also save latest checkpoint
    latest_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
    torch.save(checkpoint, latest_path)

def load_checkpoint(checkpoint_path, mllm, fusion_module, decoder, optimizer, device):
    """Load training checkpoint"""
    if os.path.isfile(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        mllm.load_state_dict(checkpoint['mllm_state_dict'])
        fusion_module.load_state_dict(checkpoint['fusion_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from epoch {start_epoch}")
        return start_epoch
    else:
        print(f"No checkpoint found at {checkpoint_path}, starting from scratch")
        return 0

def main():
    # Parse arguments
    args = parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create checkpoint directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_dir = os.path.join(args.checkpoint_dir, f'training_{timestamp}')
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Checkpoints will be saved to: {checkpoint_dir}")
    
    # Initialize data loaders
    print(f"Loading data from {args.data_root}...")
    data_loader = PRSMedDataLoader(
        batch_size=8,
        num_workers=4,
        data_root=args.data_root
    )
    
    # Get dataloaders
    train_loader = data_loader.get_dataloader('train', shuffle=True)
    val_loader = data_loader.get_dataloader('val', shuffle=False)
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Create model
    print("Initializing models...")
    mllm = LLaVAMedMLLM().to(device)
    fusion_module = PromptMaskFusionModule().to(device)
    decoder = MaskPredictionModule().to(device)
    
    # Create optimizer
    optimizer = optim.AdamW(
        list(mllm.parameters()) + 
        list(fusion_module.parameters()) + 
        list(decoder.parameters()), 
        lr=1e-4
    )
    
    # Loss function
    criterion = PRSMedLoss()
    
    # Try to resume from latest checkpoint
    latest_checkpoint = os.path.join(args.checkpoint_dir, 'latest_checkpoint.pth')
    start_epoch = load_checkpoint(latest_checkpoint, mllm, fusion_module, decoder, optimizer, device)
    
    # Training parameters
    num_epochs = 100
    best_val_loss = float('inf')
    
    # Training loop
    print("Starting training...")
    for epoch in range(start_epoch, num_epochs):
        # Training
        mllm.train()
        fusion_module.train()
        decoder.train()
        
        epoch_loss = 0.0
        epoch_start_time = time.time()
        
        for batch_idx, batch in enumerate(train_loader):
            # Move data to device
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            questions = batch['question']
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass (simplified - adjust based on your actual forward pass)
            visual_features = mllm.encode_image(images)
            text_features = mllm.encode_text(questions)
            fused_features = fusion_module(visual_features, text_features)
            pred_masks = decoder(fused_features)
            
            # Calculate loss
            loss = criterion(pred_masks, masks)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        avg_train_loss = epoch_loss / len(train_loader)
        epoch_time = time.time() - epoch_start_time
        
        print(f'Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}, Time: {epoch_time:.2f}s')
        
        # Validation
        mllm.eval()
        fusion_module.eval()
        decoder.eval()
        
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                masks = batch['mask'].to(device)
                questions = batch['question']
                
                visual_features = mllm.encode_image(images)
                text_features = mllm.encode_text(questions)
                fused_features = fusion_module(visual_features, text_features)
                pred_masks = decoder(fused_features)
                
                loss = criterion(pred_masks, masks)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f'Epoch {epoch+1} - Val Loss: {avg_val_loss:.4f}')
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            save_checkpoint(epoch, mllm, fusion_module, decoder, optimizer, checkpoint_dir)
            print(f"Checkpoint saved at epoch {epoch+1}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_checkpoint(epoch, mllm, fusion_module, decoder, optimizer, checkpoint_dir, is_best=True)
            print(f"New best model saved with val_loss: {avg_val_loss:.4f}")
        
        # Save latest checkpoint every epoch
        save_checkpoint(epoch, mllm, fusion_module, decoder, optimizer, checkpoint_dir)
    
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final checkpoints saved in: {checkpoint_dir}")

if __name__ == "__main__":
    main()