import os
import csv
from typing import Dict, Any
from PIL import Image
from torch.utils.data import Dataset


class PRSMedCSVDataset(Dataset):
    """
    CSV columns expected:
    - image_path (mask path or image path; we’ll use mask_root/image_root to resolve)
    - image_name (e.g., 'benign (1).png' OR a bare name without extension)
    - question
    - answer (ground-truth text)
    - position (normalized string like 'top left and near the center')
    - split (train/val/test)

    For masks: if your CSV 'image_path' points to mask files, we’ll pull the paired image from
    a parallel folder by replacing 'train_masks' -> 'train_images' etc.; otherwise, we use roots.
    """

    def __init__(self, csv_file: str, image_root: str, mask_root: str, img_tf, mask_tf):
        self.rows = []
        with open(csv_file, newline='') as f:
            reader = csv.DictReader(f)
            for r in reader:
                self.rows.append(r)
        self.image_root = image_root
        self.mask_root = mask_root
        self.img_tf = img_tf
        self.mask_tf = mask_tf

    def _resolve_paths(self, row: Dict[str, Any]):
        image_name = row["image_name"]
        # Derive mask path
        # Case 1: CSV has a mask path in image_path:
        p = row["image_path"]
        if "mask" in p or "masks" in p:
            mask_path = p if os.path.isabs(p) else os.path.join(self.mask_root, p)
            # Try to map to corresponding image path
            if "train_masks" in mask_path:
                image_path = mask_path.replace("train_masks", "train_images")
            elif "val_masks" in mask_path:
                image_path = mask_path.replace("val_masks", "val_images")
            else:
                # fallback to image_root + image_name
                image_path = os.path.join(self.image_root, image_name)
        else:
            # CSV has image path -> compute mask path by folder swap
            image_path = p if os.path.isabs(p) else os.path.join(self.image_root, p)
            if "train_images" in image_path:
                mask_path = image_path.replace("train_images", "train_masks")
            elif "val_images" in image_path:
                mask_path = image_path.replace("val_images", "val_masks")
            else:
                mask_path = os.path.join(self.mask_root, image_name)

        return image_path, mask_path

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        image_path, mask_path = self._resolve_paths(r)

        img = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        img = self.img_tf(img)
        mask = self.mask_tf(mask)
        # binarize mask robustly
        mask = (mask > 0.5).float()

        sample = {
            "pixel_values": img,  # [3,H,W]
            "mask": mask,  # [1,H,W]
            "question": r["question"],
            "answer_gt": r.get("answer", ""),
            "position": r.get("position", ""),
            "image_path": image_path,
            "mask_path": mask_path
        }
        return sample
