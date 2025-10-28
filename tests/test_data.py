import numpy as np
import torch
from data_pipeline.dataset_mmrs import compute_centroid, position_label, MMRSDataset
from pathlib import Path
from PIL import Image


def test_compute_centroid_center():
    mask = np.zeros((10,10)); mask[4:6,4:6] = 1
    c = compute_centroid(mask)
    assert abs(c[0]-5) < 1 and abs(c[1]-5) < 1

def test_position_labels():
    shape = (100,100)
    assert position_label((10,10), shape) == "top-left"
    assert position_label((90,10), shape) == "top-right"
    assert position_label((10,90), shape) == "bottom-left"
    assert position_label((90,90), shape) == "bottom-right"
    assert position_label((50,50), shape) == "near-center"

def test_dataset_io(tmp_path):
    img = Image.new("RGB",(64,64),(255,255,255))
    mask = Image.new("L",(64,64),0)
    for p in [tmp_path/"images", tmp_path/"masks"]:
        p.mkdir()
    img.save(tmp_path/"images"/"a.png")
    mask.save(tmp_path/"masks"/"a.png")
    ds = MMRSDataset(tmp_path, split="train", img_size=64)
    item = ds[0]
    assert "image" in item and "mask" in item
    assert isinstance(item["question"], str)
    assert item["mask"].shape[-1] == 64
