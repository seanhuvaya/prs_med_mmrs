"""Complete PRS-Med model implementation."""

import torch
import torch.nn as nn
from models.tiny_sam_encoder import TinySAMVisionBackbone
from models.mllm.llava_med_lora_adapter import LLavaMedWithLoRA
from models.decoder.fusion_module import PromptMaskFusionModule
from models.decoder.mask_prediction_module import MaskPredictionModule
from utils.config import PRSMedConfig

class PRSMedModel(nn.Module):
    """
    Complete PRS-Med model with explicit dtype handling
    """
    def __init__(self, args: PRSMedConfig, device):
        super().__init__()
        self.device = device
        self.image_size = args.image_size
        
        # Vision backbone (TinySAM)
        # Note: We'll move it to device after creation to ensure all submodules are on the same device
        self.vision_backbone = TinySAMVisionBackbone(
            checkpoint_path=args.tinysam_ckpt,
            image_size=args.image_size,
            device=str(device)
        )
        # Explicitly move to device to ensure all submodules are on the correct device
        self.vision_backbone = self.vision_backbone.to(device)
        
        # Multimodal LLM with LoRA
        self.mllm = LLavaMedWithLoRA(
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            freeze_llm=True,
            device=str(device)
        )
        # Explicitly move to device to ensure all submodules are on the correct device
        self.mllm = self.mllm.to(device)
        
        # Fusion and mask modules - explicitly set to float32
        self.fusion_module = PromptMaskFusionModule().to(device).float()
        self.mask_predictor = MaskPredictionModule().to(device).float()
        
    def preprocess_images(self, images):
        """
        Preprocess images and ensure float32
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
        
        # Ensure float32
        return images.float()
        
    def forward(self, images, text_prompts):
        """
        Forward pass with explicit dtype conversion and device checking
        """
        # Ensure input images are on the correct device
        if isinstance(images, torch.Tensor) and images.device != self.device:
            images = images.to(self.device)
        
        # Preprocess images and ensure float32
        processed_images = self.preprocess_images(images)
        
        # Ensure processed images are on the correct device
        if processed_images.device != self.device:
            processed_images = processed_images.to(self.device)
        
        # 1. Extract visual features using TinySAM (ensure float32)
        z_image = self.vision_backbone(processed_images).float()  # (B, 256, 16, 16)
        if z_image.device != self.device:
            z_image = z_image.to(self.device)
        
        # 2. Get multimodal embeddings from LLaVA-Med and convert to float32
        mllm_output = self.mllm(processed_images, text_prompts, return_projected=True)
        z_emb = mllm_output["z_emb"].float()      # Convert to float32
        z_txt_logits = mllm_output["z_txt"].float() # Convert to float32
        pred_ids = mllm_output["pred_ids"]
        
        # Ensure all outputs are on the correct device
        if z_emb.device != self.device:
            z_emb = z_emb.to(self.device)
        if z_txt_logits.device != self.device:
            z_txt_logits = z_txt_logits.to(self.device)
        
        # 3. Fuse visual and multimodal features
        z_fused = self.fusion_module(z_image, z_emb)  # (B, 256, 16, 16)
        if z_fused.device != self.device:
            z_fused = z_fused.to(self.device)
        
        # 4. Generate segmentation mask
        z_mask = self.mask_predictor(z_fused)  # (B, 1, 1024, 1024)
        if z_mask.device != self.device:
            z_mask = z_mask.to(self.device)
        
        return {
            "z_mask": z_mask,        # Segmentation logits
            "z_txt_logits": z_txt_logits,  # Text generation logits
            "pred_ids": pred_ids,    # Predicted token IDs
        }

