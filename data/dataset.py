import glob
import os
import re

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from utils import logging

logger = logging.get_logger(__name__)


def get_default_transform():
    return transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_default_mask_transform():
    return transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
    ])


class PRSMedDataset(Dataset):
    def __init__(self, split='train', transform=None, mask_transform=None,
                 data_root="data", img_dir="images_and_masks",
                 specific_dataset=None):
        self.split = split
        self.data_root = data_root
        self.img_dir = img_dir
        self.specific_dataset = specific_dataset
        self.transform = transform or get_default_transform()
        self.mask_transform = mask_transform or get_default_mask_transform()
        self.df = self._load_annotations()

        # Filter by split
        self.df = self.df[self.df["split"] == self.split]

        # Filter by specific dataset if provided
        if specific_dataset:
            self.df = self.df[self.df['dataset_name'] == specific_dataset]
            logger.info(f"Filtered to dataset: {specific_dataset}")

        logger.info(f"Loaded {len(self.df)} samples for {split} split" +
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

    def get_available_datasets(self):
        """Get list of all available datasets in the current split"""
        return self.df['dataset_name'].unique().tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Handle image path
        image_path = os.path.join(self.data_root, self.img_dir, row["image_path"])

        # Handle mask path
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
            'image_path': image_path
        }


class PRSMedDataLoader:
    @staticmethod
    def get_dataloader(split='train', batch_size=8, num_workers=4, data_root="data", shuffle=None, transform=None,
                       mask_transform=None, specific_dataset=None):
        """Get a single dataloader for a specific split and optionally specific dataset"""
        if shuffle is None:
            shuffle = (split == 'train')

        loaded_dataset = PRSMedDataset(
            split=split,
            transform=transform,
            mask_transform=mask_transform,
            data_root=data_root,
            specific_dataset=specific_dataset
        )

        logger.info(f"Loaded: {len(loaded_dataset)} {split} samples | Batch size: {batch_size}")

        return DataLoader(
            loaded_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True
        )

    @staticmethod
    def get_training_dataloaders(batch_size=8, num_workers=4, data_root="data"):
        """Get dataloaders for training: combined dataset"""
        return (PRSMedDataLoader.get_dataloader(split='train', batch_size=batch_size, num_workers=num_workers,
                                                data_root=data_root, shuffle=True),
                PRSMedDataLoader.get_dataloader(split='val', batch_size=batch_size, num_workers=num_workers,
                                                data_root=data_root, shuffle=False))

    @staticmethod
    def get_testing_dataloaders(data_root="data"):
        """Get dataloaders for testing: separate dataloader for each dataset"""
        # First get the test split to see available datasets
        test_dataset = PRSMedDataset(split='test', data_root=data_root)
        available_datasets = test_dataset.get_available_datasets()

        test_dataloaders = {}
        for dataset_name in available_datasets:
            test_dataloaders[dataset_name] = PRSMedDataLoader.get_dataloader(
                split='test',
                shuffle=False,
                specific_dataset=dataset_name,
                data_root=data_root
            )
            logger.info(
                f"Created test dataloader for {dataset_name} with {len(test_dataloaders[dataset_name].dataset)} samples")

        return test_dataloaders
