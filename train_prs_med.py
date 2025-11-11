import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
from datetime import datetime

# Import all components including vision backbone
from data.dataset import PRSMedDataLoader
from models.vision_backbone.tiny_sam_encoder import TinySAMVisionBackbone  # Add this
from models.mllm.llava_med_lora_adapter import LLavaMedWithLoRA
from models.decoder.fusion_module import PromptMaskFusionModule
from models.decoder.mask_prediction_module import MaskPredictionModule
from models.loss.objective_function import PRSMedLoss

def parse_args():
    parser = argparse.ArgumentParser(description='PRS-Med Training')
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--tinysam_checkpoint', type=str, default='weights/tinysam_42.3.pth',
                       help='Path to TinySAM checkpoint')
    parser.add_argument('--lora_rank', type=int, default=16)
    parser.add_argument('--lora_alpha', type=int, default=16)
    parser.add_argument('--lora_dropout', type=float, default=0.05)
    parser.add_argument('--lambda_seg', type=float, default=1.0)
    parser.add_argument('--lambda_txt', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--image_size', type=int, default=1024)
    return parser.parse_args()

class PRSMedModel(nn.Module):
    """
    Complete PRS-Med model with proper dtype handling
    """
    def __init__(self, args, device):
        super().__init__()
        self.device = device
        self.image_size = args.image_size
        
        # Vision backbone (TinySAM) - typically float32
        self.vision_backbone = TinySAMVisionBackbone(
            checkpoint_path=args.tinysam_checkpoint,
            image_size=args.image_size,
            device=str(device)
        )
        
        # Multimodal LLM with LoRA - might use mixed precision
        self.mllm = LLavaMedWithLoRA(
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            freeze_llm=True,
            device=str(device)
        )
        
        # Fusion and mask modules
        self.fusion_module = PromptMaskFusionModule().to(device)
        self.mask_predictor = MaskPredictionModule().to(device)
        
    def preprocess_images(self, images):
        """
        Preprocess images for TinySAM backbone
        """
        # If images are already tensors, ensure they're the right size
        if isinstance(images, torch.Tensor):
            B, C, H, W = images.shape
            
            # Resize if necessary
            if H != self.image_size or W != self.image_size:
                images = torch.nn.functional.interpolate(
                    images, 
                    size=(self.image_size, self.image_size), 
                    mode='bilinear', 
                    align_corners=False
                )
            
            # Normalize if needed (check if already normalized)
            if images.max() > 2.0:  # Likely unnormalized [0, 255]
                images = images / 255.0
                mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
                images = (images - mean) / std
        
        return images
        
    def forward(self, images, text_prompts):
        """
        Proper forward pass with dtype consistency
        """
        # Preprocess images for TinySAM
        processed_images = self.preprocess_images(images)
        
        # 1. Extract visual features using TinySAM (float32)
        z_image = self.vision_backbone(processed_images)  # (B, 256, 16, 16) - float32
        
        # 2. Get multimodal embeddings from LLaVA-Med (might be float16)
        mllm_output = self.mllm(processed_images, text_prompts, return_projected=True)
        z_emb = mllm_output["z_emb"]      # (B, L, 4096) - could be float16
        z_txt_logits = mllm_output["z_txt"] # (B, L, vocab_size)
        pred_ids = mllm_output["pred_ids"]  # (B, L)
        
        # 3. Fuse visual and multimodal features (handles dtype conversion internally)
        z_fused = self.fusion_module(z_image, z_emb)  # (B, 256, 16, 16)
        
        # 4. Generate segmentation mask
        z_mask = self.mask_predictor(z_fused)  # (B, 1, 1024, 1024)
        
        return {
            "z_mask": z_mask,        # Segmentation logits
            "z_txt_logits": z_txt_logits,  # Text generation logits
            "pred_ids": pred_ids,    # Predicted token IDs
        }

def prepare_text_targets(answers, tokenizer, max_length=512):
    """
    Properly tokenize answers for text generation loss
    """
    # Tokenize answers with the same tokenizer used in LLaVA-Med
    tokenized = tokenizer(
        answers,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )
    
    return tokenized.input_ids

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create checkpoint directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_dir = os.path.join(args.checkpoint_dir, f'training_{timestamp}')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize data loaders
    print(f"Loading data from {args.data_root}...")
    data_loader = PRSMedDataLoader(
        batch_size=args.batch_size,
        num_workers=4,
        data_root=args.data_root
    )
    
    train_loader = data_loader.get_dataloader('train', shuffle=True)
    val_loader = data_loader.get_dataloader('val', shuffle=False)
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Initialize complete PRS-Med model
    print("Initializing PRS-Med model...")
    model = PRSMedModel(args, device)
    
    # Test forward pass
    print("Testing forward pass...")
    model.eval()
    with torch.no_grad():
        test_batch = next(iter(train_loader))
        images = test_batch['image'].to(device)
        questions = test_batch['question']
        
        outputs = model(images, questions)
        print(f"Segmentation output: {outputs['z_mask'].shape}")
        print(f"Text logits: {outputs['z_txt_logits'].shape}")
        print(f"Predicted IDs: {outputs['pred_ids'].shape}")
        print("Forward pass test successful!")
    
    # Setup optimizer and loss
    trainable_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append(param)
            print(f"Training parameter: {name}")
    
    optimizer = optim.AdamW(trainable_params, lr=args.learning_rate)
    criterion = PRSMedLoss(lambda_seg=args.lambda_seg, lambda_txt=args.lambda_txt)
    
    # Training parameters
    best_val_loss = float('inf')
    
    print("Starting training...")
    for epoch in range(args.num_epochs):
        # Training phase
        model.train()
        epoch_loss_total = 0.0
        epoch_loss_seg = 0.0
        epoch_loss_txt = 0.0
        epoch_start_time = time.time()
        
        for batch_idx, batch in enumerate(train_loader):
            # Move data to device
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            questions = batch['question']
            answers = batch['answer']
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images, questions)
            pred_masks = outputs["z_mask"]
            text_logits = outputs["z_txt_logits"]
            
            # Prepare text targets
            text_targets = prepare_text_targets(answers, model.mllm.processor.tokenizer)
            text_targets = text_targets.to(device)
            
            # Calculate loss
            loss_dict = criterion(
                z_mask=pred_masks,
                y_mask=masks,
                z_txt=text_logits,  # Use text logits, not embeddings
                y_txt=text_targets
            )
            
            loss_total = loss_dict["loss_total"]
            
            # Backward pass
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            
            # Accumulate losses
            epoch_loss_total += loss_total.item()
            epoch_loss_seg += loss_dict["loss_seg"].item()
            epoch_loss_txt += loss_dict["loss_txt"].item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch+1}, Batch {batch_idx}, '
                      f'Total Loss: {loss_total.item():.4f}, '
                      f'Seg Loss: {loss_dict["loss_seg"].item():.4f}, '
                      f'Text Loss: {loss_dict["loss_txt"].item():.4f}')
        
        # Calculate epoch averages
        avg_train_loss_total = epoch_loss_total / len(train_loader)
        avg_train_loss_seg = epoch_loss_seg / len(train_loader)
        avg_train_loss_txt = epoch_loss_txt / len(train_loader)
        epoch_time = time.time() - epoch_start_time
        
        print(f'Epoch {epoch+1} - TRAIN - '
              f'Total Loss: {avg_train_loss_total:.4f}, '
              f'Seg Loss: {avg_train_loss_seg:.4f}, '
              f'Text Loss: {avg_train_loss_txt:.4f}, '
              f'Time: {epoch_time:.2f}s')
        
        # Validation phase
        model.eval()
        val_loss_total = 0.0
        val_loss_seg = 0.0
        val_loss_txt = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                masks = batch['mask'].to(device)
                questions = batch['question']
                answers = batch['answer']
                
                outputs = model(images, questions)
                pred_masks = outputs["z_mask"]
                text_logits = outputs["z_txt_logits"]
                
                text_targets = prepare_text_targets(answers, model.mllm.processor.tokenizer)
                text_targets = text_targets.to(device)
                
                loss_dict = criterion(
                    z_mask=pred_masks,
                    y_mask=masks,
                    z_txt=text_logits,
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
        
        # Save checkpoints
        if avg_val_loss_total < best_val_loss:
            best_val_loss = avg_val_loss_total
            save_checkpoint(epoch, model, optimizer, checkpoint_dir, is_best=True)
            print(f"New best model saved with val_loss: {avg_val_loss_total:.4f}")
        
        # Save periodic checkpoints
        if (epoch + 1) % 5 == 0:
            save_checkpoint(epoch, model, optimizer, checkpoint_dir)
            print(f"Checkpoint saved at epoch {epoch+1}")
    
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final checkpoints saved in: {checkpoint_dir}")

def save_checkpoint(epoch, model, optimizer, checkpoint_dir, is_best=False):
    """Save complete model checkpoint"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
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

if __name__ == "__main__":
    main()