import torch 
import torch.nn as nn 
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.InstanceNorm2d(in_channels, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=1)
        self.bn2 = nn.InstanceNorm2d(out_channels, affine=True)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=0)

    def weight_init(self):
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        torch.nn.init.xavier_uniform_(self.conv3.weight)
        if self.conv1.bias is not None:
            torch.nn.init.zeros_(self.conv1.bias)
        if self.conv2.bias is not None:
            torch.nn.init.zeros_(self.conv2.bias)
        if self.conv3.bias is not None:
            torch.nn.init.zeros_(self.conv3.bias)
        torch.nn.init.ones_(self.bn1.weight)
        torch.nn.init.zeros_(self.bn1.bias)
        torch.nn.init.ones_(self.bn2.weight)
        torch.nn.init.zeros_(self.bn2.bias)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = out + identity
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.conv3(out)
        return out

class FFN(nn.Module):
    def __init__(self, in_features, out_features):
        super(FFN, self).__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(out_features, in_features)
        self.norm = nn.LayerNorm(in_features, eps=1e-5)
        self.norm1 = nn.LayerNorm(in_features, eps=1e-5)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.zeros_(self.fc2.bias)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.zeros_(self.fc1.bias)

    def forward(self, x):
        identity = x
        x = self.fc1(x)        
        x = self.act(x)
        x = x + identity
        x = self.fc2(x)
        x = self.act(x)
        x= self.norm(x)
        return x

class PromptedMaskDecoder(nn.Module):
    def __init__(self, prompt_dim=4096, image_dim=256, hidden_dim=512):
        super().__init__()

        self.prompt_projection = nn.Sequential(
            nn.Linear(prompt_dim, hidden_dim),
            nn.LayerNorm(hidden_dim, eps = 1e-5),
            nn.ReLU(),
            nn.Linear(hidden_dim, image_dim),
            nn.LayerNorm(image_dim, eps = 1e-5),
            nn.ReLU()
        )

        for layer in self.prompt_projection:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.ones_(layer.bias)
            elif isinstance(layer, nn.LayerNorm):
                nn.init.ones_(layer.weight)
                nn.init.ones_(layer.bias)

        # cross-attention: image tokens attend to prompt
        self.attn = nn.MultiheadAttention(embed_dim=image_dim, num_heads=8, batch_first=True)
        self.norm1 = nn.LayerNorm(image_dim, eps=1e-5)
        nn.init.xavier_uniform_(self.attn.in_proj_weight)
        nn.init.xavier_uniform_(self.attn.out_proj.weight)
        nn.init.ones_(self.attn.in_proj_bias)
        nn.init.ones_(self.attn.out_proj.bias)

        self.ffn = FFN(image_dim, image_dim)

        torch.nn.init.xavier_uniform_(self.ffn.fc1.weight)
        torch.nn.init.ones_(self.ffn.fc1.bias)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=16, batch_first=True, activation = "relu", dim_feedforward=512)
        torch.nn.init.xavier_uniform_(self.encoder_layer.self_attn.in_proj_weight)
        torch.nn.init.xavier_uniform_(self.encoder_layer.self_attn.out_proj.weight)
        torch.nn.init.ones_(self.encoder_layer.self_attn.in_proj_bias)
        torch.nn.init.ones_(self.encoder_layer.self_attn.out_proj.bias)
        torch.nn.init.xavier_uniform_(self.encoder_layer.linear1.weight)
        torch.nn.init.ones_(self.encoder_layer.linear1.bias)
        torch.nn.init.xavier_uniform_(self.encoder_layer.linear2.weight)
        torch.nn.init.ones_(self.encoder_layer.linear2.bias)

        self.mask_generation = nn.Sequential(
            BasicBlock(image_dim, image_dim),
            BasicBlock(image_dim, image_dim // 2),
            BasicBlock(image_dim // 2, image_dim // 4),
        )

        self.relu = nn.ReLU(inplace=True)
        self.out_dec = nn.Conv2d(image_dim // 4, 1, 1)
        self.image_norm = nn.LayerNorm(image_dim, eps=1e-5)
        self.prompt_norm = nn.LayerNorm(image_dim, eps=1e-5)
        torch.nn.init.xavier_uniform_(self.out_dec.weight)
        torch.nn.init.zeros_(self.out_dec.bias)

    def forward(self, image_feat, prompt_feat):
        """
        image_feat: (B, 256, 64, 64) - float32
        prompt_feat: (B, T, 2048) - float16
        """
        B, _, H, W = image_feat.shape
        image_feat = image_feat.float()

        prompt_feat = prompt_feat.float()
        
        # Clamp extreme values to prevent numerical instability
        image_feat = torch.clamp(image_feat, min=-1e4, max=1e4)
        prompt_feat = torch.clamp(prompt_feat, min=-1e4, max=1e4)
        
        image_identity = image_feat
        
        prompt_proj = self.prompt_projection(prompt_feat)  # (B, T, hidden_dim)
        
        # Check for NaN after projection
        if torch.isnan(prompt_proj).any() or torch.isinf(prompt_proj).any():
            # If NaN detected, use zero-initialized projection as fallback
            prompt_proj = torch.zeros_like(prompt_proj)
        
        image_flat = image_feat.flatten(2).transpose(1, 2)  # (B, H*W, hidden_dim)
        image_flat = self.image_norm(image_flat)  # (B, H*W, hidden_dim)
        prompt_proj = self.prompt_norm(prompt_proj)
        
        # Clamp after normalization to prevent extreme values
        image_flat = torch.clamp(image_flat, min=-1e4, max=1e4)
        prompt_proj = torch.clamp(prompt_proj, min=-1e4, max=1e4)
        
        attn_out, _ = self.attn(image_flat, prompt_proj, prompt_proj)  # (B, H*W, hidden_dim)
        attn_out = attn_out + image_flat  # (B, H*W, hidden_dim)
        
        # Check for NaN after attention
        if torch.isnan(attn_out).any() or torch.isinf(attn_out).any():
            # Fallback to identity if attention produces NaN
            attn_out = image_flat
        
        attn_out = self.norm1(attn_out)  # (B, H*W, hidden_dim)
        attn_out = self.encoder_layer(attn_out)
        
        # Check for NaN after encoder layer
        if torch.isnan(attn_out).any() or torch.isinf(attn_out).any():
            # Fallback to previous state
            attn_out = self.norm1(image_flat)
        
        attn_out = self.ffn(attn_out)  # (B, H*W, hidden_dim)
        
        # Check for NaN after FFN
        if torch.isnan(attn_out).any() or torch.isinf(attn_out).any():
            # Fallback to identity
            attn_out = image_flat
        
        attn_map = attn_out.transpose(1, 2).reshape(B, -1, H, W)  # (B, hidden_dim, H, W)
        
        attn_map = attn_map + image_identity
        
        # Clamp before mask generation to prevent extreme values
        attn_map = torch.clamp(attn_map, min=-1e4, max=1e4)

        attn_map = self.mask_generation(attn_map)  # (B, hidden_dim // 4, H, W)
        
        # Check for NaN after mask generation
        if torch.isnan(attn_map).any() or torch.isinf(attn_map).any():
            # Fallback to zero mask
            attn_map = torch.zeros_like(attn_map)
        
        mask = self.out_dec(attn_map)  # (B, 1, 128, 128)
        mask = F.interpolate(mask, scale_factor=16, mode='bilinear', align_corners=True)  # (B, 1, 64, 64)
        
        # Final clamp to ensure valid output range
        mask = torch.clamp(mask, min=-10.0, max=10.0)  # Reasonable range before sigmoid
        
        return mask

