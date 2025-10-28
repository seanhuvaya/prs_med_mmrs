import torch
import torch.nn as nn

class CrossAttentionFusion(nn.Module):
    def __init__(self, img_dim=256, txt_dim=1024, hidden_dim=256, num_heads=8):
        super().__init__()
        self.proj_img = nn.Linear(img_dim, hidden_dim)
        self.proj_txt = nn.Linear(txt_dim, hidden_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.self_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)

    def forward(self, z_img: torch.Tensor, z_txt: torch.Tensor) -> torch.Tensor:
        B, C, H, W = z_img.shape
        z_img_tok = z_img.flatten(2).permute(0, 2, 1)
        q = self.proj_img(z_img_tok)
        k = self.proj_txt(z_txt)
        v = k
        attn_out, _ = self.cross_attn(q, k, v)
        attn_out, _ = self.self_attn(attn_out, attn_out, attn_out)
        fused = attn_out + q 
        fused = fused.permute(0, 2, 1).reshape(B, C, H, W)
        return fused