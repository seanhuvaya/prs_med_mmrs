import torch
import numpy as np
import json

from pathlib import Path
from torch.utils.data import Dataset
from .transforms import PRSPreprocess, MaskTransform
from configs.modality_config import get_modality_mapping, STANDARD_MODALITIES

def compute_centroid(mask: np.ndarray) -> np.ndarray:
    y, x = np.nonzero(mask)
    if len(x) == 0:
        return (mask.shape[0] // 2, mask.shape[1] // 2)
    return (float(np.mean(y)), float(np.mean(x)))
    
def position_label(centroid, img_shape, threshold: int = 20):
    cx, cy = centroid
    h, w = img_shape
    center_x, center_y = w / 2, h / 2
    dx, dy = cx - center_x, cy - center_y
    dist = (dx ** 2 + dy ** 2) ** 0.5
    if dist <= threshold:
        return "near-center"
        
    if dy < 0 and dx < 0:
        return "top-left"
    elif dy < 0 and dx >= 0:
        return "top-right"
    elif dy >= 0 and dx < 0:
        return "bottom-left"
    else:
        return "bottom-right"

def generate_question(position, template) -> str:
    return template.replace("{position_description}", position)


TEMPLATES_TRAIN = [
    "Where is the lesion located? {position_description}",
    "What is the position of the abnormality? {position_description}",
    "Describe the location of the finding: {position_description}"
]
TEMPLATES_TEST = [
    "Where is the lesion located? {position_description}",
    "What is the position of the abnormality? {position_description}",
    "Describe the location of the finding: {position_description}"
]

class MMRSDataset(Dataset):
    def __init__(self, root, split="train", img_size=224, tokenizer=None):
        self.root = Path(root)
        self.split = split
        self.img_dir = self.root / "images"
        self.mask_dir = self.root / "masks"
        self.preprocess = PRSPreprocess(img_size)
        self.mask_transform = MaskTransform(img_size)
        # Load templates from JSON files
        template_path = Path(__file__).parent / "templates" / f"{split}_templates.json"
        self.templates = json.load(open(template_path))
        self.tokenizer = tokenizer
        self.img_paths = sorted(list(self.img_dir.glob("*.jpg")) + list(self.img_dir.glob("*.png")))
        self.mask_paths = sorted(list(self.mask_dir.glob("*.jpg")) + list(self.mask_dir.glob("*.png")))
        

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, mask_path = self.img_paths[idx], self.mask_paths[idx]
        image = self.preprocess(img_path)
        mask = self.mask_transform(mask_path)

        mask_np = mask.squeeze(0).numpy()
        centroid = compute_centroid(mask_np)
        position_desc = position_label(centroid, mask_np.shape)
        
        # Select random template and format it
        template = np.random.choice(self.templates)
        question = template["question"].replace("{position_description}", position_desc)
        answer = template["answer"].replace("{position_description}", position_desc)
        
        # Add image type based on modality configuration
        # Extract dataset name from the full path (e.g., data/mmrs/brain_tumors_ct_scan/train -> brain_tumors_ct_scan)
        dataset_name = self.root.parts[-2] if len(self.root.parts) >= 2 else self.root.name
        modality_mapping = get_modality_mapping(dataset_name)
        if modality_mapping:
            image_type = modality_mapping.image_type
            modality_type = modality_mapping.modality_type
        else:
            # Fallback to folder name if no mapping found
            image_type = dataset_name.replace("_", " ").title()
            modality_type = "Unknown"
        
        question = question.replace("{image_type}", image_type)
        answer = answer.replace("{image_type}", image_type)
        
        # Add modality type to question/answer if placeholders exist
        question = question.replace("{modality_type}", modality_type)
        answer = answer.replace("{modality_type}", modality_type)
        
        if self.tokenizer:
            answer_ids = self.tokenizer(answer, return_tensors="pt", padding="max_length", max_length=8, truncation=True).input_ids.squeeze(0)
        else:
            answer_ids = torch.tensor([0])

        return {
            "image": image,
            "mask": mask,
            "question": question,
            "answer": answer,
            "answer_ids": answer_ids}