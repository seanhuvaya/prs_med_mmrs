import os
import torch
import torch.nn as nn
import gc, psutil
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

from models.vision_backbone.tiny_sam_encoder import TinySAMVisionBackbone
from models.mllm.llava_med_lora_adapter import LLavaMedWithLoRA
from models.decoder.fusion_module import PromptMaskFusionModule
from models.decoder.mask_prediction_module import MaskPredictionModule
from models.loss.objective_function import PRSMedLoss

class PRSMedModel(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        
        self.encoder = TinySAMVisionBackbone(
            checkpoint_path="weights/tinysam_42.3.pth", 
            device=self.device
        )
        self.mllm = LLavaMedWithLoRA(
            rank=4, alpha=8, dropout=0.05, device=self.device
        )
        self.fusion = PromptMaskFusionModule(img_dim=256, emb_dim=4096, fused_dim=256)
        self.decoder = MaskPredictionModule(in_channels=256)
        
        # Add final resize layer to match target size
        self.final_resize = nn.Upsample(size=1024, mode='bilinear', align_corners=False)

    def forward(self, images, questions):
        # Vision + MLLM forward
        z_image = self.encoder(images)
        mllm_out = self.mllm(images, questions, return_projected=True)
        z_emb = mllm_out["z_emb"]
        z_txt = mllm_out["z_txt"]

        z_fused = self.fusion(z_image, z_emb)
        z_mask = self.decoder(z_fused)
        
        # Resize to match target mask size
        z_mask = self.final_resize(z_mask)

        return {"z_mask": z_mask, "z_txt": z_txt}

class FixedPRSMedLoss(nn.Module):
    def __init__(self, lambda_seg=1.0, lambda_txt=0.5):
        super().__init__()
        self.lambda_seg = lambda_seg
        self.lambda_txt = lambda_txt
        self.seg_loss = nn.BCEWithLogitsLoss()  # Assuming binary segmentation
        self.txt_loss = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding tokens
    
    def forward(self, z_mask, y_mask, z_txt, y_txt):
        # Handle text sequence length mismatch
        seq_len = z_txt.shape[1]  
        target_len = y_txt.shape[1]  
        
        if seq_len != target_len:
            # Option 1: Truncate target to match model output
            if seq_len < target_len:
                y_txt_adj = y_txt[:, :seq_len]  # Truncate target
                print(f"Truncated target to: {y_txt_adj.shape}")
            # Option 2: Pad model output (less common)
            else:
                # Pad z_txt to match target length
                pad_size = target_len - seq_len
                z_txt_padded = torch.nn.functional.pad(
                    z_txt, 
                    (0, 0, 0, pad_size),  # Pad last dimension (sequence length)
                    value=0
                )
                z_txt = z_txt_padded
                y_txt_adj = y_txt
                print(f"Padded model output to: {z_txt.shape}")
        else:
            y_txt_adj = y_txt
        
        # Calculate segmentation loss
        seg_loss = self.seg_loss(z_mask, y_mask)
        
        # Calculate text loss - reshape for cross entropy
        # z_txt: [B, seq_len, vocab_size] -> [B * seq_len, vocab_size]
        # y_txt_adj: [B, seq_len] -> [B * seq_len]
        txt_loss = self.txt_loss(
            z_txt.contiguous().view(-1, z_txt.size(-1)), 
            y_txt_adj.contiguous().view(-1)
        )
        
        total_loss = self.lambda_seg * seg_loss + self.lambda_txt * txt_loss
        
        return {
            "loss_total": total_loss,
            "loss_seg": seg_loss,
            "loss_txt": txt_loss
        }

def train_prs_med(train_loader, val_loader, device="cpu"):
    torch.set_num_threads(2)
    if device == "cpu":
        os.environ['OMP_NUM_THREADS'] = '2'
    
    model = PRSMedModel(device=device)
    criterion = FixedPRSMedLoss(lambda_seg=1.0, lambda_txt=0.5)
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    num_epochs = 5
    best_val_loss = float("inf")
    ckpt_dir = "checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")
        for batch_idx, batch in enumerate(pbar):
            if batch_idx % 10 == 0:
                gc.collect()
                
            images, questions, gt_masks, gt_tokens = batch

            optimizer.zero_grad()
            outputs = model(images, questions)
            
            losses = criterion(
                z_mask=outputs["z_mask"],
                y_mask=gt_masks,
                z_txt=outputs["z_txt"],
                y_txt=gt_tokens,
            )

            loss = losses["loss_total"]
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({
                "train_loss": loss.item(),
                "seg_loss": losses["loss_seg"].item(),
                "txt_loss": losses["loss_txt"].item()
            })

        avg_train = total_loss / len(train_loader)
        print(f"Epoch {epoch} | Train Loss: {avg_train:.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, questions, gt_masks, gt_tokens in val_loader:
                outputs = model(images, questions)
                losses = criterion(
                    z_mask=outputs["z_mask"],
                    y_mask=gt_masks,
                    z_txt=outputs["z_txt"],
                    y_txt=gt_tokens,
                )
                val_loss += losses["loss_total"].item()

        avg_val = val_loss / len(val_loader)
        print(f"Validation Loss: {avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), f"{ckpt_dir}/best_model_epoch{epoch}.pt")

if __name__ == "__main__":
    B = 1
    image_size = 1024
    
    dummy_imgs_raw = torch.rand(B, 3, image_size, image_size)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    dummy_imgs = (dummy_imgs_raw - mean) / std
    
    dummy_masks = (torch.rand(B, 1, image_size, image_size) > 0.5).float()
    
    # Use the correct sequence length that matches your MLLM output (593 instead of 595)
    correct_seq_len = 593  # Adjust this based on your MLLM output
    dummy_tokens = torch.randint(0, 32064, (B, correct_seq_len))
    
    dummy_questions = ["Where is the tumor?"]

    dataset = [
        (dummy_imgs[i], dummy_questions[i], dummy_masks[i], dummy_tokens[i])
        for i in range(B)
    ]
    loader = DataLoader(dataset, batch_size=1)

    device = "cpu"
    print("🚨 Using CPU for stability")
    train_prs_med(loader, loader, device=device)