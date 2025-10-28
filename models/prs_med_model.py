import torch
import torch.nn as nn

from models.image_encoder import TinyVisionEncoder
from models.text_encoder import MultimodalTextEncoder
from models.fusion_module import CrossAttentionFusion
from models.mask_decoder import MaskDecoder

class PRSMedModel(nn.Module):
    def __init__(self, base_llava_model: str = "microsoft/DialoGPT-medium"):
        super().__init__() 
        self.image_encoder = TinyVisionEncoder()
        self.text_encoder = MultimodalTextEncoder(base_model=base_llava_model)
        self.fusion = CrossAttentionFusion()
        self.mask_decoder = MaskDecoder()

    def forward(self, image: torch.Tensor, text: list[str]) -> torch.Tensor:
        z_img = self.image_encoder(image)
        z_txt = self.text_encoder(text)
        z_fused = self.fusion(z_img, z_txt)
        mask = self.mask_decoder(z_fused)
        logits = self.text_encoder.model.lm_head(z_txt)
        return {"mask": mask, "logits": logits}
        