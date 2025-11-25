import os
import tempfile
import shutil
import unittest
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import RandomSampler, SequentialSampler
from torchvision import transforms

from data.dataset import PRSMedDataset, PRSMedDataLoader


def _make_image(path, size=(8, 8), color=(128, 128, 128), mode="RGB"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = Image.new(mode, size, color)
    img.save(path)


class TestPRSMedDataset(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        # Create annotations dir
        self.ann_dir = os.path.join(self.tmpdir, "annotations")
        os.makedirs(self.ann_dir, exist_ok=True)
        # Base image root
        self.img_root = os.path.join(self.tmpdir, "images_and_masks")

        # Build small synthetic dataset spanning two CSVs (datasets)
        # Dataset A
        df_a = pd.DataFrame([
            {
                "split": "train",
                "image_path": "dataset_a/train_images/a1.png",
                "question": "Q1?",
                "answer": "A1",
            },
            {
                "split": "test",
                "image_path": "dataset_a/test_images/a2.png",
                "question": "Q2?",
                "answer": "A2",
            },
        ])
        df_a.to_csv(os.path.join(self.ann_dir, "dataset_a.csv"), index=False)

        # Dataset B
        df_b = pd.DataFrame([
            {
                "split": "train",
                "image_path": "dataset_b/train_images/b1.png",
                "question": "QB1?",
                "answer": "AB1",
            },
            {
                "split": "val",
                "image_path": "dataset_b/val_images/b2.png",
                "question": "QB2?",
                "answer": "AB2",
            },
        ])
        df_b.to_csv(os.path.join(self.ann_dir, "dataset_b.csv"), index=False)

        # Create corresponding images and masks for rows above
        # For each image_path, produce RGB image and matching L mask under *_masks
        for rel in [
            "dataset_a/train_images/a1.png",
            "dataset_a/test_images/a2.png",
            "dataset_b/train_images/b1.png",
            "dataset_b/val_images/b2.png",
        ]:
            img_path = os.path.join(self.img_root, rel)
            _make_image(img_path, size=(8, 8), mode="RGB")
            mask_rel = rel.replace("/train_images/", "/train_masks/") \
                           .replace("/test_images/", "/test_masks/") \
                           .replace("/val_images/", "/val_masks/")
            mask_path = os.path.join(self.img_root, mask_rel)
            _make_image(mask_path, size=(8, 8), mode="L")

        # Simple transforms: keep small, no resize to 1024
        self.img_tf = transforms.Compose([transforms.ToTensor()])
        self.mask_tf = transforms.Compose([transforms.ToTensor()])

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_file_not_found_when_no_csv(self):
        empty_root = tempfile.mkdtemp()
        try:
            os.makedirs(os.path.join(empty_root, "annotations"), exist_ok=True)
            with self.assertRaises(FileNotFoundError):
                PRSMedDataset(split="train", data_root=empty_root)
        finally:
            shutil.rmtree(empty_root)

    def test_len_and_split_filtering(self):
        ds_train = PRSMedDataset(
            split="train",
            transform=self.img_tf,
            mask_transform=self.mask_tf,
            data_root=self.tmpdir,
        )
        self.assertEqual(len(ds_train), 2)  # a1 + b1

        ds_val = PRSMedDataset(
            split="val",
            transform=self.img_tf,
            mask_transform=self.mask_tf,
            data_root=self.tmpdir,
        )
        self.assertEqual(len(ds_val), 1)  # b2

        ds_test = PRSMedDataset(
            split="test",
            transform=self.img_tf,
            mask_transform=self.mask_tf,
            data_root=self.tmpdir,
        )
        self.assertEqual(len(ds_test), 1)  # a2

    def test_specific_dataset_and_available(self):
        ds_all = PRSMedDataset(
            split="train",
            transform=self.img_tf,
            mask_transform=self.mask_tf,
            data_root=self.tmpdir,
        )
        av = sorted(ds_all.get_available_datasets())
        self.assertEqual(av, ["dataset_a", "dataset_b"])

        ds_a = PRSMedDataset(
            split="train",
            transform=self.img_tf,
            mask_transform=self.mask_tf,
            data_root=self.tmpdir,
            specific_dataset="dataset_a",
        )
        self.assertEqual(len(ds_a), 1)

    def test_getitem_returns_tensors_and_paths(self):
        ds = PRSMedDataset(
            split="train",
            transform=self.img_tf,
            mask_transform=self.mask_tf,
            data_root=self.tmpdir,
        )
        sample = ds[0]
        self.assertIn("image", sample)
        self.assertIn("mask", sample)
        self.assertIn("question", sample)
        self.assertIn("answer", sample)
        self.assertIn("image_path", sample)

        img = sample["image"]
        msk = sample["mask"]
        self.assertIsInstance(img, torch.Tensor)
        self.assertIsInstance(msk, torch.Tensor)
        self.assertEqual(tuple(img.shape), (3, 8, 8))
        self.assertEqual(tuple(msk.shape), (1, 8, 8))
        self.assertEqual(img.dtype, torch.float32)
        self.assertEqual(msk.dtype, torch.float32)

        # Ensure mask path mapping is correct (exists)
        self.assertTrue(os.path.exists(sample["image_path"]))

    def test_get_dataloader_and_shuffle_default(self):
        # Train split should shuffle by default
        train_loader = PRSMedDataLoader.get_dataloader(
            split="train",
            batch_size=2,
            num_workers=0,
            data_root=self.tmpdir,
            transform=self.img_tf,
            mask_transform=self.mask_tf,
        )
        # Validate sampler type
        self.assertIsInstance(train_loader.sampler, RandomSampler)

        # Val split should not shuffle
        val_loader = PRSMedDataLoader.get_dataloader(
            split="val",
            batch_size=2,
            num_workers=0,
            data_root=self.tmpdir,
            transform=self.img_tf,
            mask_transform=self.mask_tf,
        )
        self.assertIsInstance(val_loader.sampler, SequentialSampler)

        # Lengths
        self.assertEqual(len(train_loader.dataset), 2)
        self.assertEqual(len(val_loader.dataset), 1)

    def test_get_testing_dataloaders_per_dataset(self):
        loaders = PRSMedDataLoader.get_testing_dataloaders(data_root=self.tmpdir)
        # Should have dataset_a only for test split in our synthetic data
        self.assertIn("dataset_a", loaders)
        self.assertIsInstance(loaders["dataset_a"], type(next(iter(loaders.values()))))
        # Dataset lengths
        self.assertEqual(len(loaders["dataset_a"].dataset), 1)


if __name__ == "__main__":
    unittest.main()
