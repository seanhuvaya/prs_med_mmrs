import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttentionFusion(nn.Module):
    """
    Cross-attend z_image (B,256,h,w) with z_emb (B,L,D) by projecting both into a shared 256-d space,
    do (Q=img_tokens, K/V=text_tokens), then reshape and add skip as in the paper Eq.(3-4).  # :contentReference[oaicite:9]{index=9}
    """
    def __init__(self, emb_dim: int, proj_dim: int = 256, n_heads: int = 8):
        super().__init__()
        self.img_proj = nn.Linear(256, proj_dim)
        self.txt_proj = nn.Linear(emb_dim, proj_dim)
        self.attn = nn.MultiheadAttention(embed_dim=proj_dim, num_heads=n_heads, batch_first=True)
        self.sa = nn.MultiheadAttention(embed_dim=proj_dim, num_heads=n_heads, batch_first=True)

    def forward(self, z_image, z_emb):
        B, C, H, W = z_image.shape
        img_tokens = z_image.flatten(2).permute(0,2,1)      # [B,HW,256]
        img_tokens = self.img_proj(img_tokens)               # [B,HW,proj]
        txt_tokens = self.txt_proj(z_emb)                    # [B,L,proj]

        # Cross-attention: Q=img, K/V=txt
        fused, _ = self.attn(img_tokens, txt_tokens, txt_tokens)  # [B,HW,proj]
        # Self-attention on fused
        fused, _ = self.sa(fused, fused, fused)
        # Residual to image tokens (skip)
        fused = fused + img_tokens
        fused = fused.permute(0,2,1).reshape(B, -1, H, W)    # [B,proj,H,W]
        return fused

class UpsampleDecoder(nn.Module):
    def __init__(self, in_ch=256, channels=(256,128,64,32,16)):
        super().__init__()
        chs = [in_ch] + list(channels)
        layers = []
        for i in range(len(chs)-1):
            layers += [
                nn.ConvTranspose2d(chs[i], chs[i+1], kernel_size=2, stride=2),
                nn.BatchNorm2d(chs[i+1]),
                nn.ReLU(inplace=True),
            ]
        self.up = nn.Sequential(*layers)
        self.head = nn.Conv2d(chs[-1], 1, kernel_size=1)

    def forward(self, x):
        x = self.up(x)
        return self.head(x)

class PRSMedModel(nn.Module):
    """
    Full model = vision backbone -> fusion with LLM embeddings -> upsampling decoder to mask.
    LLM logits are returned for text CE loss. Joint objective per paper Eq.(5-7).  # :contentReference[oaicite:10]{index=10}
    """
    def __init__(self, vision, mllm, proj_dim=256, n_heads=8, decoder_channels=(256,128,64,32,16)):
        super().__init__()
        self.vision = vision
        self.mllm = mllm
        self.fusion = CrossAttentionFusion(emb_dim=4096, proj_dim=proj_dim, n_heads=n_heads)  # LLaVA hidden size (adjust if needed)
        self.decoder = UpsampleDecoder(in_ch=proj_dim, channels=decoder_channels)

    def forward(self, pixel_values, question, text_labels=None):
        z_image = self.vision(pixel_values)                      # [B,256,h,w]
        z_emb, logits = self.mllm(pixel_values, question)        # [B,L,4096], [B,L,V]
        z_fused = self.fusion(z_image, z_emb)                    # [B,256,h,w]
        z_mask = self.decoder(z_fused)                           # [B,1,1024,1024]
        return {"mask_logits": z_mask, "logits": logits}
