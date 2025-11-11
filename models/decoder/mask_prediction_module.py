import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskPredictionModule(nn.Module):
    """
    Correct implementation that upsamples from (B,256,16,16) → (B,1,1024,1024)
    Each transpose conv 2x upsampling: 16 → 32 → 64 → 128 → 256 → 512 → 1024
    """
    def __init__(self, in_channels=256, out_channels=1):
        super().__init__()
        
        # Calculate the number of upsampling steps needed
        # Start: 16x16, Target: 1024x1024 → 6 upsampling steps (2^6 = 64x multiplier)
        self.upsample_blocks = nn.Sequential(
            # 16x16 → 32x32
            nn.ConvTranspose2d(in_channels, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # 32x32 → 64x64
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # 64x64 → 128x128
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # 128x128 → 256x256
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            # 256x256 → 512x512
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            
            # 512x512 → 1024x1024
            nn.ConvTranspose2d(8, 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            
            # Final convolution to get 1 channel
            nn.Conv2d(4, out_channels, kernel_size=1)
        )

    def forward(self, z_fused):
        """
        Args:
            z_fused: (B, 256, 16, 16)
        Returns:
            z_mask: (B, 1, 1024, 1024)
        """
        return self.upsample_blocks(z_fused)

if __name__ == "__main__":
    B = 2
    z_fused = torch.randn(B, 256, 16, 16)

    decoder = MaskPredictionModule(in_channels=256)
    z_mask = decoder(z_fused)

    print(f"Input z_fused: {tuple(z_fused.shape)}")
    print(f"Output z_mask: {tuple(z_mask.shape)}")
