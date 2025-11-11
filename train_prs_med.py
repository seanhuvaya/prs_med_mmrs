import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from data import PRSMedDataLoader
from models.mllm.llava_med_lora_adapter import LLavaMedWithLoRA
from models.decoder.fusion_module import PromptMaskFusionModule
from models.decoder.mask_prediction_module import MaskPredictionModule
from models.loss.objective_function import PRSMedLoss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create checkpoints directory
    os.makedirs('checkpoints', exist_ok=True)
    
    # Initialize data and models
    data_loader = PRSMedDataLoader(data_root=args.data_root)
    train_loader = data_loader.get_dataloader('train')
    
    # Initialize LLaVA-Med with LoRA
    mllm = LLavaMedWithLoRA(
        rank=16,
        alpha=16,
        dropout=0.05,
        freeze_llm=True,
        device=device
    ).to(device)
    
    fusion_module = PromptMaskFusionModule(img_dim=256, emb_dim=4096, fused_dim=256, num_heads=8).to(device)
    decoder = MaskPredictionModule(in_channels=256).to(device)
    
    # Optimizer - only trainable parameters (LoRA + fusion + decoder)
    trainable_params = []
    trainable_params.extend(fusion_module.parameters())
    trainable_params.extend(decoder.parameters())
    
    # Add LoRA parameters
    for name, param in mllm.named_parameters():
        if param.requires_grad:
            trainable_params.append(param)
    
    optimizer = optim.AdamW(trainable_params, lr=1e-4)
    criterion = PRSMedLoss(lambda_seg=1.0, lambda_text=1.0)
    
    # Training loop
    num_epochs = 20
    for epoch in range(num_epochs):
        mllm.train()
        fusion_module.train()
        decoder.train()
        
        epoch_loss = 0.0
        epoch_seg_loss = 0.0
        epoch_text_loss = 0.0   
        for batch_idx, batch in enumerate(train_loader):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            questions = batch['question']
            
            optimizer.zero_grad()
            
            # Forward pass through LLaVA-Med + LoRA
            mllm_output = mllm(images, questions, return_projected=True)
            visual_embeddings = mllm_output["z_emb_proj"]
            text_embeddings = mllm_output["z_txt"]
            
            # Fusion and decoding
            fused_features = fusion_module(visual_embeddings, text_embeddings)
            pred_masks = decoder(fused_features)
            
            loss = criterion(pred_masks, masks, text_embeddings, questions)
            loss_seg = loss['loss_seg']
            loss_text = loss['loss_txt']
            
            loss = loss_seg + loss_text
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        avg_loss = epoch_loss / len(train_loader)
        print(f'Epoch {epoch+1} Average Loss: {avg_loss:.4f}')
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'mllm_state_dict': mllm.state_dict(),
                'fusion_state_dict': fusion_module.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(checkpoint, f'checkpoints/epoch_{epoch+1}.pth')
            
            # Save LoRA adapter separately
            lora_path = f'checkpoints/llava_med_lora_epoch_{epoch+1}'
            mllm.model.save_pretrained(lora_path)
            print(f"Saved checkpoint and LoRA adapter for epoch {epoch+1}")
    
    # Save final model
    final_checkpoint = {
        'mllm_state_dict': mllm.state_dict(),
        'fusion_state_dict': fusion_module.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
    }
    torch.save(final_checkpoint, 'checkpoints/final_model.pth')
    
    # Save final LoRA adapter
    mllm.model.save_pretrained('checkpoints/llava_med_lora_final')
    print("Saved final model and LoRA adapter")

if __name__ == "__main__":
    main()