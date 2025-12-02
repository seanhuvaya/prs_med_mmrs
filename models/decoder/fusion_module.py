import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttentionBlock(nn.Module):
    """Cross-attention between vision tokens (query) and multimodal tokens (key/value)."""
    def __init__(self, dim=256, num_heads=8, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, 
            num_heads=num_heads, 
            dropout=dropout, 
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, kv):
        """
        q: (B, N, C) vision tokens
        kv: (B, L, C) multimodal tokens
        """
        # Ensure same dtype
        if q.dtype != kv.dtype:
            kv = kv.to(q.dtype)
            
        # Layer norm before attention
        q_norm = self.norm1(q)
        kv_norm = self.norm1(kv)

        attn_out, _ = self.attn(q_norm, kv_norm, kv_norm)
        q = q + self.dropout(attn_out)  # residual

        # Feed-forward
        out = self.mlp(self.norm2(q))
        q = q + self.dropout(out)
        return q


class PromptMaskFusionModule(nn.Module):
    """
    Implements the Fusion Module with proper dtype handling
    """
    def __init__(self, img_dim=256, emb_dim=4096, fused_dim=256, num_heads=8):
        super().__init__()
        self.img_dim = img_dim
        self.emb_dim = emb_dim
        self.fused_dim = fused_dim
        
        # Projection layers
        self.proj_image = nn.Sequential(
            nn.Conv2d(img_dim, fused_dim, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(fused_dim),
        )
        self.proj_text = nn.Sequential(
            nn.Linear(emb_dim, fused_dim),
            nn.GELU(),
            nn.LayerNorm(fused_dim),
        )
        self.cross_attn = CrossAttentionBlock(dim=fused_dim, num_heads=num_heads)
        self.output_conv = nn.Sequential(
            nn.Conv2d(fused_dim, fused_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(fused_dim),
        )

    def forward(self, z_image, z_emb):
        """
        Args:
            z_image: (B, 256, 16, 16) - from TinySAM
            z_emb:   (B, L, 4096) - from LLaVA-Med (could be float16)
        Returns:
            z_fused: (B, 256, 16, 16)
        """
        B, C, H, W = z_image.shape
        _, L, _ = z_emb.shape

        # Ensure both inputs are on the same device and dtype
        device = next(self.parameters()).device
        
        # Move inputs to correct device if needed
        if z_image.device != device:
            z_image = z_image.to(device)
        if z_emb.device != device:
            z_emb = z_emb.to(device)
            
        # Convert both to float32 for stability (or match z_image's dtype)
        target_dtype = z_image.dtype
        if z_emb.dtype != target_dtype:
            z_emb = z_emb.to(target_dtype)

        # Projection into shared latent space
        z_image_proj = self.proj_image(z_image)        # (B, 256, 16, 16)
        z_emb_proj = self.proj_text(z_emb)             # (B, L, 256)

        # Flatten image to tokens
        z_img_tokens = z_image_proj.flatten(2).permute(0, 2, 1)  # (B, 256, 256) -> (B, 256, 256)

        # Cross-attention fusion
        z_fused_tokens = self.cross_attn(z_img_tokens, z_emb_proj)  # (B, 256, 256)

        # Reshape back to spatial map
        z_fused = z_fused_tokens.permute(0, 2, 1).reshape(B, self.fused_dim, H, W)  # (B, 256, 16, 16)

        # Skip connection (ensure same dtype)
        if z_fused.dtype != z_image.dtype:
            z_fused = z_fused.to(z_image.dtype)
            
        z_fused = z_fused + z_image

        # Output refinement
        z_fused = self.output_conv(z_fused)
        return z_fused

if __name__ == "__main__":
    print("Testing PromptMaskFusionModule with mixed precision...")

    B, L, H, W = 2, 595, 16, 16
    # Simulate mixed precision inputs
    z_image = torch.randn(B, 256, H, W, dtype=torch.float32)  # TinySAM output (float32)
    z_emb = torch.randn(B, L, 4096, dtype=torch.float16)      # LLaVA-Med output (float16)

    fusion = PromptMaskFusionModule(img_dim=256, emb_dim=4096, fused_dim=256, num_heads=8)
    out = fusion(z_image, z_emb)

    print(f"Input z_image: {tuple(z_image.shape)}, dtype: {z_image.dtype}")
    print(f"Input z_emb:   {tuple(z_emb.shape)}, dtype: {z_emb.dtype}")
    print(f"Output z_fused: {tuple(out.shape)}, dtype: {out.dtype}")