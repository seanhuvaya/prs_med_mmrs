import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import time
from datetime import datetime

from data.dataset import PRSMedDataLoader
from models.mllm.llava_med_lora_adapter import LLavaMedWithLoRA
from models.decoder.fusion_module import PromptMaskFusionModule
from models.decoder.mask_prediction_module import MaskPredictionModule
from models.loss.objective_function import PRSMedLoss

def parse_args():
    parser = argparse.ArgumentParser(description='PRS-Med Multi-Modal Medical Training with LLaVA-Med + LoRA')
    parser.add_argument('--data_root', type=str, required=True,
                       help='Root directory of the dataset (should contain annotations/ and images_and_masks/)')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--lora_rank', type=int, default=16,
                       help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=16,
                       help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.05,
                       help='LoRA dropout')
    parser.add_argument('--freeze_llm', action='store_true', default=True,
                       help='Freeze the base LLM (only train LoRA adapters)')
    parser.add_argument('--lambda_seg', type=float, default=1.0,
                       help='Weight for segmentation loss')
    parser.add_argument('--lambda_txt', type=float, default=0.5,
                       help='Weight for text reasoning loss')
    return parser.parse_args()

def save_checkpoint(epoch, mllm, fusion_module, mask_predictor, optimizer, checkpoint_dir, is_best=False):
    """Save training checkpoint"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'mllm_state_dict': mllm.state_dict(),
        'fusion_state_dict': fusion_module.state_dict(),
        'mask_predictor_state_dict': mask_predictor.state_dict(),
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

def load_checkpoint(checkpoint_path, mllm, fusion_module, mask_predictor, optimizer, device):
    """Load training checkpoint"""
    if os.path.isfile(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        mllm.load_state_dict(checkpoint['mllm_state_dict'])
        fusion_module.load_state_dict(checkpoint['fusion_state_dict'])
        mask_predictor.load_state_dict(checkpoint['mask_predictor_state_dict'])
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
    device_str = str(device)  # Convert to string for model initialization
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
    
    # Create model with LoRA - pass device as string
    print("Initializing LLaVA-Med with LoRA...")
    mllm = LLavaMedWithLoRA(
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        freeze_llm=args.freeze_llm,
        device=device_str  # Pass as string instead of torch.device object
    )
    mllm = mllm.to(device)  # Then move to device
    
    # Initialize other modules
    fusion_module = PromptMaskFusionModule().to(device)
    mask_predictor = MaskPredictionModule().to(device)
    
    # Create optimizer (only trainable parameters)
    trainable_params = []
    trainable_params.extend(fusion_module.parameters())
    trainable_params.extend(mask_predictor.parameters())
    
    # Add LoRA parameters from MLLM
    for name, param in mllm.named_parameters():
        if param.requires_grad:
            trainable_params.append(param)
            print(f"Training parameter: {name}")
    
    optimizer = optim.AdamW(trainable_params, lr=1e-4)
    
    # Loss function with your custom weights
    criterion = PRSMedLoss(lambda_seg=args.lambda_seg, lambda_txt=args.lambda_txt)
    
    # Try to resume from latest checkpoint
    latest_checkpoint = os.path.join(args.checkpoint_dir, 'latest_checkpoint.pth')
    start_epoch = load_checkpoint(latest_checkpoint, mllm, fusion_module, mask_predictor, optimizer, device)
    
    # Training parameters
    num_epochs = 100
    best_val_loss = float('inf')
    
    # Training loop
    print("Starting training...")
    for epoch in range(start_epoch, num_epochs):
        # Training
        mllm.train()
        fusion_module.train()
        mask_predictor.train()
        
        epoch_loss_total = 0.0
        epoch_loss_seg = 0.0
        epoch_loss_txt = 0.0
        epoch_start_time = time.time()
        
        for batch_idx, batch in enumerate(train_loader):
            # Move data to device
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            questions = batch['question']
            answers = batch['answer']  # We'll need this for text loss
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass through LLaVA-Med with LoRA
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                mllm_output = mllm(images, questions, return_projected=True)
                
                # Use the projected embeddings for fusion
                visual_embeddings = mllm_output["z_emb_proj"]  # Projected visual features
                text_embeddings = mllm_output["z_txt"]         # Text embeddings
                pred_ids = mllm_output["pred_ids"]             # Generated text tokens
                
                # Fusion and mask prediction
                fused_features = fusion_module(visual_embeddings, text_embeddings)
                pred_masks = mask_predictor(fused_features)
                
                # Prepare text targets (you might need to adjust this based on your data)
                # For now, using pred_ids as both input and target for text loss
                # You'll need to tokenize the answers properly for real training
                text_targets = pred_ids  # This is a placeholder - adjust based on your needs
                
                # Calculate multi-modal loss
                loss_dict = criterion(
                    z_mask=pred_masks,
                    y_mask=masks,
                    z_txt=text_embeddings,  # Or use appropriate text predictions
                    y_txt=text_targets
                )
                
                loss_total = loss_dict["loss_total"]
                loss_seg = loss_dict["loss_seg"]
                loss_txt = loss_dict["loss_txt"]
            
            # Backward pass
            loss_total.backward()
            optimizer.step()
            
            epoch_loss_total += loss_total.item()
            epoch_loss_seg += loss_seg.item()
            epoch_loss_txt += loss_txt.item()
            
            if batch_idx % 50 == 0:
                print(f'Epoch {epoch+1}, Batch {batch_idx}, '
                      f'Total Loss: {loss_total.item():.4f}, '
                      f'Seg Loss: {loss_seg.item():.4f}, '
                      f'Text Loss: {loss_txt.item():.4f}')
        
        # Calculate epoch averages
        avg_train_loss_total = epoch_loss_total / len(train_loader)
        avg_train_loss_seg = epoch_loss_seg / len(train_loader)
        avg_train_loss_txt = epoch_loss_txt / len(train_loader)
        epoch_time = time.time() - epoch_start_time
        
        print(f'Epoch {epoch+1} - '
              f'Total Loss: {avg_train_loss_total:.4f}, '
              f'Seg Loss: {avg_train_loss_seg:.4f}, '
              f'Text Loss: {avg_train_loss_txt:.4f}, '
              f'Time: {epoch_time:.2f}s')
        
        # Validation
        mllm.eval()
        fusion_module.eval()
        mask_predictor.eval()
        
        val_loss_total = 0.0
        val_loss_seg = 0.0
        val_loss_txt = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                masks = batch['mask'].to(device)
                questions = batch['question']
                answers = batch['answer']
                
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    mllm_output = mllm(images, questions, return_projected=True)
                    visual_embeddings = mllm_output["z_emb_proj"]
                    text_embeddings = mllm_output["z_txt"]
                    pred_ids = mllm_output["pred_ids"]
                    
                    fused_features = fusion_module(visual_embeddings, text_embeddings)
                    pred_masks = mask_predictor(fused_features)
                    
                    text_targets = pred_ids  # Placeholder - adjust as needed
                    
                    loss_dict = criterion(
                        z_mask=pred_masks,
                        y_mask=masks,
                        z_txt=text_embeddings,
                        y_txt=text_targets
                    )
                    
                    val_loss_total += loss_dict["loss_total"].item()
                    val_loss_seg += loss_dict["loss_seg"].item()
                    val_loss_txt += loss_dict["loss_txt"].item()
        
        # Calculate validation averages
        avg_val_loss_total = val_loss_total / len(val_loader)
        avg_val_loss_seg = val_loss_seg / len(val_loader)
        avg_val_loss_txt = val_loss_txt / len(val_loader)
        
        print(f'Epoch {epoch+1} - VALIDATION - '
              f'Total Loss: {avg_val_loss_total:.4f}, '
              f'Seg Loss: {avg_val_loss_seg:.4f}, '
              f'Text Loss: {avg_val_loss_txt:.4f}')
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            save_checkpoint(epoch, mllm, fusion_module, mask_predictor, optimizer, checkpoint_dir)
            print(f"Checkpoint saved at epoch {epoch+1}")
            
            # Also save LoRA adapter separately
            lora_save_path = os.path.join(checkpoint_dir, f'llava_med_lora_epoch_{epoch+1}')
            mllm.model.save_pretrained(lora_save_path)
            print(f"LoRA adapter saved to {lora_save_path}")
        
        # Save best model
        if avg_val_loss_total < best_val_loss:
            best_val_loss = avg_val_loss_total
            save_checkpoint(epoch, mllm, fusion_module, mask_predictor, optimizer, checkpoint_dir, is_best=True)
            print(f"New best model saved with val_loss: {avg_val_loss_total:.4f}")
        
        # Save latest checkpoint every epoch
        save_checkpoint(epoch, mllm, fusion_module, mask_predictor, optimizer, checkpoint_dir)
    
    # Save final LoRA adapter
    final_lora_path = os.path.join(checkpoint_dir, 'llava_med_lora_final')
    mllm.model.save_pretrained(final_lora_path)
    print(f"Final LoRA adapter saved to {final_lora_path}")
    
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final checkpoints saved in: {checkpoint_dir}")

if __name__ == "__main__":
    main()