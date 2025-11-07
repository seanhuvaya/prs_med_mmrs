import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskPredictionModule(nn.Module):
    """
    Upsamples fused representation (B,256,16,16) → (B,1,1024,1024)
    using stacked ConvTranspose2d + BN + ReLU blocks.
    """

    def __init__(self, in_channels=256, mid_channels=[128, 64, 32, 16, 8, 4], out_channels=1):
        super().__init__()

        layers = []
        input_c = in_channels
        for c in mid_channels:
            layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(input_c, c, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(c),
                    nn.ReLU(inplace=True),
                )
            )
            input_c = c

        # Final output layer → 1 channel mask logits
        layers.append(
            nn.Conv2d(input_c, out_channels, kernel_size=1)
        )

        self.upsample_blocks = nn.ModuleList(layers)

    def forward(self, z_fused):
        """
        Args:
            z_fused: (B,256,16,16)
        Returns:
            z_mask: (B,1,1024,1024)
        """
        x = z_fused
        for block in self.upsample_blocks:
            x = block(x)
        return x

if __name__ == "__main__":
    B = 2
    z_fused = torch.randn(B, 256, 16, 16)

    decoder = MaskPredictionModule(in_channels=256)
    z_mask = decoder(z_fused)

    print(f"Input z_fused: {tuple(z_fused.shape)}")
    print(f"Output z_mask: {tuple(z_mask.shape)}")
