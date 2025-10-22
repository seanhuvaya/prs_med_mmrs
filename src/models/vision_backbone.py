import torch
import torch.nn as nn
import timm

class TinyVisionBackbone(nn.Module):
    """
    TinyViT-based encoder that yields a 1/16 resolution feature map with C=256.
    Mirrors the 'Tiny SAM' encoder role (kept trainable by default) as in the paper.  # :contentReference[oaicite:7]{index=7}
    """
    def __init__(self, model_name="tiny_vit_21m_224", pretrained=True, out_dim=256):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, features_only=True, out_indices=(3,))
        in_ch = self.backbone.feature_info.channels()[-1]
        self.proj = nn.Conv2d(in_ch, out_dim, kernel_size=1)

    def forward(self, x):
        # x: [B,3,H,W] -> features at ~1/16 scale
        feats = self.backbone(x)[0]  # [B,C,h,w]
        feats = self.proj(feats)     # [B,256,h,w]
        return feats
