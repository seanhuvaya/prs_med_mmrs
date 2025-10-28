import torch
import torch.nn as nn
import timm

class TinyVisionEncoder(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model("vit_tiny_patch16_224", pretrained=pretrained, features_only=True)
        self.projection = nn.Conv2d(192, 256, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)[-1]
        out =  self.projection(features)
        return out
