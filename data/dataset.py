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
        
        # Load image and mask (supports standard formats and .npz tensors)
        image = self._load_image_file(image_path, is_mask=False)
        mask = self._load_image_file(mask_path, is_mask=True)
        
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

    def _load_image_file(self, path, is_mask: bool):
        """Load an image or mask from common formats or .npz files."""
        ext = os.path.splitext(path)[1].lower()
        if ext == ".npz":
            array = self._load_npz_array(path)
            pil_img = self._np_to_pil(array, is_mask=is_mask)
        else:
            pil_img = Image.open(path)
        return pil_img.convert("L" if is_mask else "RGB")

    @staticmethod
    def _load_npz_array(path: str):
        """Load numpy array from .npz file, supporting arbitrary key names."""
        data = np.load(path)
        try:
            if "arr_0" in data.files:
                array = data["arr_0"]
            else:
                # Fallback to the first available key
                array = data[data.files[0]]
        finally:
            data.close()
        return array

    @staticmethod
    def _np_to_pil(array: np.ndarray, is_mask: bool):
        """Convert numpy array to PIL Image while handling channel layouts."""
        if not isinstance(array, np.ndarray):
            array = np.array(array)
        
        # Squeeze singleton dimensions
        array = np.squeeze(array)
        
        # Normalize dtype for PIL compatibility
        if array.dtype != np.uint8:
            # Preserve binary masks without scaling
            if is_mask and np.array_equal(np.unique(array), [0, 1]):
                array = (array * 255).astype(np.uint8)
            else:
                array = array.astype(np.float32)
                min_val = array.min()
                max_val = array.max()
                if max_val > min_val:
                    array = (array - min_val) / (max_val - min_val)
                array = (array * 255).clip(0, 255).astype(np.uint8)
        
        if is_mask:
            if array.ndim == 3:
                # Reduce multi-channel mask to single channel
                array = array[..., 0]
            return Image.fromarray(array, mode="L")
        
        # Handle image channels
        if array.ndim == 2:
            array = np.stack([array] * 3, axis=-1)
        elif array.ndim == 3:
            if array.shape[0] in (1, 3) and array.shape[0] != array.shape[-1]:
                array = np.transpose(array, (1, 2, 0))
            if array.shape[-1] == 1:
                array = np.repeat(array, 3, axis=-1)
        else:
            raise ValueError(f"Unsupported array shape for image conversion: {array.shape}")
        
        return Image.fromarray(array.astype(np.uint8), mode="RGB")

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

