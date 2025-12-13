import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
import re

# Add parent PRS-Med repo to path to import utilities
parent_repo = os.path.join(os.path.dirname(__file__), '../../PRS-Med')
if os.path.exists(parent_repo):
    sys.path.insert(0, parent_repo)
    from data_utils.utils import load_annotation, load_image, binary_loader
    from llava.mm_utils import tokenizer_image_token, process_images
else:
    # Fallback if PRS-Med repo not found
    from PIL import Image
    import requests
    from io import BytesIO
    
    def load_image(image_file):
        if image_file.startswith('http://') or image_file.startswith('https://'):
            response = requests.get(image_file)
            image = Image.open(BytesIO(response.content)).convert('RGB')
        else:
            image = Image.open(image_file).convert('RGB')
        return image
    
    def binary_loader(mask_path):
        with open(mask_path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
    
    def load_annotation(annotation_path):
        if isinstance(annotation_path, str):
            annotation_path = [annotation_path]
        
        list_df = []
        for ann_path in annotation_path:
            if os.path.isfile(ann_path):
                df = pd.read_csv(ann_path)
                df = df.dropna()
                list_df.append(df)
            elif os.path.isdir(ann_path):
                csv_files = [os.path.join(ann_path, f) for f in os.listdir(ann_path) if f.endswith('.csv')]
                for csv_file in csv_files:
                    df = pd.read_csv(csv_file)
                    df = df.dropna()
                    list_df.append(df)
        
        if len(list_df) == 0:
            raise ValueError(f"No CSV files found in {annotation_path}")
        
        df = pd.concat(list_df, ignore_index=True)
        df = df.dropna()
        df = df.reset_index(drop=True)
        
        if 'split' in df.columns:
            train_df = df[df['split'] == 'train']
            test_df = df[df['split'] == 'test']
        else:
            # If no split column, use 90/10 split
            train_size = int(len(df) * 0.9)
            train_df = df.iloc[:train_size]
            test_df = df.iloc[train_size:]
        
        return train_df, test_df

IGNORE_INDEX = 0
MAX_PROMPT_LENGTH = 512


class PromptSegmentDataset(Dataset):
    def __init__(
        self,
        data_path,
        annotation_path,
        data_config,
        image_processor,
        tokenizer,
        trainsize = 512,
        mode = "train"
    ):
        self.data_path = data_path
        self.annotation_path = annotation_path
        self.tokenizer = tokenizer
        self.annotation_df = None
        
        # Handle multiple annotation paths
        if isinstance(annotation_path, str):
            annotation_path = [annotation_path]
        
        self.train_df, self.test_df = load_annotation(annotation_path)
        self.trainsize = trainsize
        self.annotation_df = self.train_df

        self.IMAGE_TOKEN_INDEX = -200
        self.image_processor = image_processor
        self.data_config = data_config
        
        self.mask_transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
        ])

        self.image_sam_transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225])
        ])
        
    def __len__(self):
        return len(self.annotation_df)

    def answer_process(self, question, prompt, answer):
        input_prompt = "<image>\n" + f"### User: {question} \n" + "### Assistant: \n" + answer
        answer_ids = tokenizer_image_token(
            input_prompt, 
            self.tokenizer, 
            self.IMAGE_TOKEN_INDEX, 
            return_tensors='pt'
        )
        return answer_ids

    def prompt_process(self, prompt):
        prompt_for_vlm = "<image> \n" + prompt 
        input_ids = tokenizer_image_token(
            prompt_for_vlm, 
            self.tokenizer, 
            self.IMAGE_TOKEN_INDEX, 
            return_tensors='pt'
        )
        return input_ids
    
    def process_image(self, image_path):
        image_pil = load_image(image_path)
        image_tensor = process_images(
            [image_pil], 
            self.image_processor, 
            self.data_config
        )
        return image_tensor.squeeze(0).to(torch.float16)
    
    def process_sam_image(self, image_path):
        image_pil = load_image(image_path)
        image_sam_tensor = self.image_sam_transform(image_pil)
        return image_sam_tensor.to(torch.float32)

    def process_mask(self, mask_path):
        mask_image = binary_loader(mask_path)
        mask_tensor = self.mask_transform(mask_image)
        return mask_tensor

    def _get_label_from_path(self, image_path):
        """Extract label from image path based on dataset name"""
        if "ISIC" in image_path or "skin" in image_path.lower():
            return 4
        elif "breast" in image_path.lower():
            return 1
        elif "brain" in image_path.lower() or "head_and_neck" in image_path.lower():
            return 0
        elif "lung_CT" in image_path or "lung_ct" in image_path.lower():
            return 2
        elif "lung_Xray" in image_path or "lung_xray" in image_path.lower():
            return 3
        elif "prostate" in image_path.lower():
            return 5
        else:
            return 0  # default

    def __getitem__(self, idx):
        row = self.annotation_df.iloc[idx]
        
        # Handle image path - support data_v2 structure
        if 'image_path' in row:
            image_path = row['image_path']
            if not os.path.isabs(image_path):
                image_path = os.path.join(self.data_path, image_path)
        elif 'image_name' in row:
            # Construct path from image_name and split
            split = row.get('split', 'train')
            task = row.get('task', 'head_and_neck')
            image_name = row['image_name']
            image_path = os.path.join(self.data_path, task, f"{split}_images", image_name)
        else:
            raise ValueError("CSV must have either 'image_path' or 'image_name' column")
        
        image_path = image_path.replace("\\", "/")
        
        # Handle mask path - support data_v2 structure
        if 'mask_path' in row:
            mask_path = row['mask_path']
            if not os.path.isabs(mask_path):
                mask_path = os.path.join(self.data_path, mask_path)
        else:
            # Infer mask path from image path
            mask_path = re.sub(r"/(train|test|val)_images/", r"/\1_masks/", image_path)
            if "ISIC" in mask_path:
                mask_path = mask_path.replace(".jpg", ".png")
        
        mask_path = mask_path.replace("\\", "/")
        
        # Get question, answer, position
        question = row.get('question', '')
        answer = row.get('answer', '')
        position = row.get('position', question)  # Fallback to question if position not available
        
        # Process data
        mask_tensor = self.process_mask(mask_path)
        image_sam_tensor = self.process_sam_image(image_path)
        image_tensor = self.process_image(image_path)
        input_ids = self.prompt_process(question)
        answers_ids = self.answer_process(question, position, answer)
        
        # Get label
        label = self._get_label_from_path(image_path)
        
        return {
            'input_ids': input_ids,
            'image_tensor': image_tensor,
            'mask_tensor': mask_tensor,
            'answers_ids': answers_ids,
            "image_sam": image_sam_tensor,
            "label": label
        }
    
def collate_fn(batch):
    padded_input_ids = nn.utils.rnn.pad_sequence(
        [item['input_ids'].squeeze(0) for item in batch], 
        batch_first=True, 
        padding_value=IGNORE_INDEX
    )
    input_ids = padded_input_ids[:, :MAX_PROMPT_LENGTH]
    input_ids = input_ids.to(torch.int64)
    

    padded_answers_ids = nn.utils.rnn.pad_sequence(
        [item['answers_ids'] for item in batch], 
        batch_first=True, 
        padding_value=IGNORE_INDEX
    )

    answers_ids = padded_answers_ids[:, :MAX_PROMPT_LENGTH]
    answers_ids = answers_ids.to(torch.int64)

    attention_masks = torch.ones_like(answers_ids)
    attention_masks[answers_ids == IGNORE_INDEX] = 0
    attention_masks = attention_masks.to(torch.long)

    image_tensor = [item['image_tensor'] for item in batch]
    image_sam_tensor = [item['image_sam'] for item in batch]
    mask_tensor = [item['mask_tensor'] for item in batch]

    image_tensor = torch.stack(image_tensor, dim=0)
    mask_tensor = torch.stack(mask_tensor, dim=0)
    image_sam_tensor = torch.stack(image_sam_tensor, dim=0)
    return {
        'input_ids': input_ids,
        'image_tensor': image_tensor,
        'mask_tensor': mask_tensor,
        'answers_ids': answers_ids,
        'image_sam': image_sam_tensor,
        "attention_masks": attention_masks,
        "label": torch.tensor([item['label'] for item in batch])
    }

def create_dataloader(
    data_path,
    annotation_path,
    data_config,
    image_processor,
    tokenizer,
    batch_size=2,
    mode="train"
):
    dataset = PromptSegmentDataset(
        data_path=data_path,
        annotation_path=annotation_path,
        data_config=data_config,
        image_processor=image_processor,
        tokenizer=tokenizer,
        mode=mode
    )

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, 
        [int(len(dataset) * 0.9), len(dataset) - int(len(dataset) * 0.9)]
    )
    
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=8,
        pin_memory=True
    )

    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=8,
        pin_memory=True
    )
    dataloader = {
        "train": train_dataloader,
        "val": val_dataloader
    }
    
    return dataloader

