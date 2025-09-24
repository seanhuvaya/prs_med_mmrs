import os
import shutil
import zipfile

import requests
from tqdm import tqdm
from pathlib import Path

DATASETS = {
    "BUSI": "https://www.kaggle.com/api/v1/datasets/download/aryashah2k/breast-ultrasound-images-dataset",
    "Kvasir-SEG": "https://datasets.simula.no/downloads/kvasir-seg.zip",
}


def download_file(url: str, out_path: Path) -> None:
    """Download a file from a URL with progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    with open(out_path, "wb") as f, tqdm(desc=f"Downloading {out_path.name}", total=total_size, unit="B",
                                         unit_scale=True) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            pbar.update(size)


def extract_zip_and_rename_dir(zip_path: Path, out_path: Path) -> None:
    """Extract a zip file"""
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        temp_path = out_path / "_tmp"
        zip_ref.extractall(temp_path)

    entries = os.listdir(temp_path)
    if len(entries) == 1 and os.path.isdir(os.path.join(temp_path, entries[0])):
        inner_dir = Path(os.path.join(temp_path, entries[0]))
    else:
        inner_dir = temp_path

    os.makedirs(out_path, exist_ok=True)

    for item in tqdm(os.listdir(inner_dir)):
        shutil.move(os.path.join(inner_dir, item), out_path)

    shutil.rmtree(temp_path, ignore_errors=True)

    print(f"Extracted {zip_path} to {out_path}")


def clean_up_zip_files(root: str = "data/raw") -> None:
    for file_path in Path(root).glob("**/*.zip"):
        os.remove(file_path)
        print(f"Deleted: {file_path}")


def fetch_dataset(name: str, root: str = "data/raw") -> Path:
    """Download a dataset"""
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: {name}")
    url = DATASETS[name]
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    zip_path = root / f"{name}.zip"
    out_path = root / name

    if not zip_path.exists():
        download_file(url, zip_path)

    extract_zip_and_rename_dir(zip_path, out_path)
    clean_up_zip_files()
    return out_path
