import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import glob
import re

class PRSMedDataset(Dataset):
    def __init__(self, split='train', transform=None, mask_transform=None, 
                 data_root="data", img_dir="images_and_masks", max_samples=None,
                 specific_dataset=None):
        self.split = split
        self.data_root = data_root
        self.img_dir = img_dir
        self.specific_dataset = specific_dataset
        self.transform = transform or self._get_default_transform()
        self.mask_transform = mask_transform or self._get_default_mask_transform()
        self.df = self._load_annotations()
        
        # Filter by split
        self.df = self._filter_by_split()
        
        # Filter by specific dataset if provided
        if specific_dataset:
            self.df = self.df[self.df['dataset_name'] == specific_dataset]
            print(f"Filtered to dataset: {specific_dataset}")
        
        # Limit samples if specified
        if max_samples:
            self.df = self.df.head(max_samples)
        
        print(f"Loaded {len(self.df)} samples for {split} split" + 
              (f" from {specific_dataset}" if specific_dataset else " from all datasets"))

    def _load_annotations(self):
        csv_files = glob.glob(os.path.join(self.data_root, "annotations", "*.csv"))
        if len(csv_files) == 0:
            raise FileNotFoundError(f"No CSV files found in {os.path.join(self.data_root, 'annotations')}")
        
        dfs = []
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            df['dataset_name'] = os.path.splitext(os.path.basename(csv_file))[0]
            dfs.append(df)
        
        return pd.concat(dfs, ignore_index=True)

    def _filter_by_split(self):
        """Filter dataframe by split using various column naming conventions"""
        df = self.df.copy()
        
        # Try different split column naming patterns
        split_patterns = [
            f"{self.split}",  # exact match
            f"split_{self.split}",
            f"is_{self.split}",
            f"{self.split}_set"
        ]
        
        found_split = False
        for pattern in split_patterns:
            if pattern in df.columns:
                df = df[df[pattern] == 1]
                found_split = True
                break
        
        # If no specific split column, look for a generic 'split' column
        if not found_split and 'split' in df.columns:
            df = df[df['split'] == self.split]
            found_split = True
        
        # If still no split filtering, use all data (with warning)
        if not found_split:
            print(f"Warning: No split filtering applied for {self.split}. Using all data.")
        
        return df

    def get_available_datasets(self):
        """Get list of all available datasets in the current split"""
        return self.df['dataset_name'].unique().tolist()

    def _get_default_transform(self):
        return transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _get_default_mask_transform(self):
        return transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Handle image path
        image_path = os.path.join(self.data_root, self.img_dir, row["image_path"])
        
        # Handle mask path - your clever regex approach
        mask_path = re.sub(r"/(train|test|val)_images/", r"/\1_masks/", row["image_path"])
        mask_path = os.path.join(self.data_root, self.img_dir, mask_path)
        
        # Load image and mask
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        
        # Apply transforms
        image = self.transform(image)
        mask = self.mask_transform(mask)
        
        return {
            'image': image.float(),
            'mask': mask.float(),
            'question': row["question"],
            'answer': row["answer"],
            'dataset': row['dataset_name'],
            'image_path': image_path
        }

class PRSMedDataLoader:
    def __init__(self, batch_size=8, num_workers=4, data_root="data", max_samples=None):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_root = data_root
        self.max_samples = max_samples
        
    def get_dataloader(self, split='train', shuffle=None, transform=None, 
                      mask_transform=None, specific_dataset=None):
        """Get a single dataloader for a specific split and optionally specific dataset"""
        if shuffle is None:
            shuffle = (split == 'train')
            
        dataset = PRSMedDataset(
            split=split,
            transform=transform,
            mask_transform=mask_transform,
            data_root=self.data_root,
            max_samples=self.max_samples,
            specific_dataset=specific_dataset
        )
        
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def get_training_dataloaders(self):
        """Get dataloaders for training: combined dataset"""
        return {
            'train': self.get_dataloader('train', shuffle=True),
            'val': self.get_dataloader('val', shuffle=False)
        }
    
    def get_testing_dataloaders(self):
        """Get dataloaders for testing: separate dataloader for each dataset"""
        # First get the test split to see available datasets
        test_dataset = PRSMedDataset(split='test', data_root=self.data_root)
        available_datasets = test_dataset.get_available_datasets()
        
        test_dataloaders = {}
        for dataset_name in available_datasets:
            test_dataloaders[dataset_name] = self.get_dataloader(
                split='test', 
                shuffle=False,
                specific_dataset=dataset_name
            )
            print(f"Created test dataloader for {dataset_name} with {len(test_dataloaders[dataset_name].dataset)} samples")
        
        return test_dataloaders
    
    def get_all_dataloaders(self):
        """Get all dataloaders: combined for train/val, separate for test"""
        return {
            'train': self.get_dataloader('train', shuffle=True),
            'val': self.get_dataloader('val', shuffle=False),
            'test_per_dataset': self.get_testing_dataloaders()
        }
    
    def get_dataset_stats(self):
        """Get statistics about each dataset"""
        stats = {}
        for split in ['train', 'val', 'test']:
            dataset = PRSMedDataset(split=split, data_root=self.data_root)
            available_datasets = dataset.get_available_datasets()
            
            stats[split] = {}
            for dataset_name in available_datasets:
                specific_ds = PRSMedDataset(split=split, data_root=self.data_root, specific_dataset=dataset_name)
                stats[split][dataset_name] = len(specific_ds)
        
        return stats

if __name__ == "__main__":
    dataset = PRSMedDataset(split="train", data_root="/Users/seanhuvaya/Documents/Capstone Project/sample/data")
    print(dataset[0])

    dataloader = PRSMedDataLoader(batch_size=8, num_workers=2, data_root="/Users/seanhuvaya/Documents/Capstone Project/sample/data")
    print(dataloader.get_dataset_stats())

