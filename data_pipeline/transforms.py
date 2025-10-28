import torch
import torchvision.transforms as T

from PIL import Image

class PRSPreprocess:
    def __init__(self, image_size: int = 224):
        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __call__(self, img) -> torch.Tensor:
        if not isinstance(img, torch.Tensor):
            img = self.transform(Image.open(img).convert("RGB"))
        return img

class MaskTransform:
    def __init__(self, image_size: int = 224):
        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
        ])

    def __call__(self, mask) -> torch.Tensor:
        m = self.transform(Image.open(mask).convert("L"))
        return (m > 0.5).float()

