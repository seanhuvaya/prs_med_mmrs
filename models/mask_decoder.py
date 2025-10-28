import torch
import torch.nn as nn

class MaskDecoder(nn.Module):
    def __init__(self, in_channels=256, target_size=224):
        super().__init__()
        self.target_size = target_size
        layers = []
        ch = in_channels
        for _ in range(6):
            layers += [
                nn.ConvTranspose2d(ch, ch // 2, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(ch // 2),
                nn.ReLU(inplace=True),
            ]
            ch //= 2
        layers.append(nn.Conv2d(ch, 1, kernel_size=1))
        self.decoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Decode to intermediate size
        decoded = self.decoder(x)
        # Interpolate to exact target size
        output = torch.nn.functional.interpolate(
            decoded, 
            size=(self.target_size, self.target_size), 
            mode='bilinear', 
            align_corners=False
        )
        return torch.sigmoid(output)