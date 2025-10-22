from torchvision import transforms as T
from PIL import Image

def build_transforms(img_size: int = 1024):
    img_tf = T.Compose([
        T.Resize((img_size, img_size), interpolation=Image.BICUBIC),
        T.ToTensor()
    ])
    mask_tf = T.Compose([
        T.Resize((img_size, img_size), interpolation=Image.NEAREST),
        T.ToTensor()  # yields {0,1}
    ])
    return img_tf, mask_tf
