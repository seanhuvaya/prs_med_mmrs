# data_utils/utils.py
from __future__ import annotations

import csv
import gzip
import json
from io import BytesIO
from pathlib import Path

import pandas as pd
import requests
from PIL import Image


def _read_jsonl(p: Path) -> pd.DataFrame:
    records = []
    opener = open if p.suffix == ".jsonl" else gzip.open
    with opener(p, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return pd.DataFrame.from_records(records)

def _read_one(p: Path) -> pd.DataFrame:
    s = p.suffix.lower()
    if s in {".jsonl", ".gz"}:
        return _read_jsonl(p)
    if s == ".json":
        # handles array-of-objects JSON
        return pd.read_json(p)
    # Fallback: CSV (robust to commas in text)
    return pd.read_csv(
        p,
        engine="python",
        sep=",",
        quoting=csv.QUOTE_MINIMAL,
        quotechar='"',
        escapechar="\\",
        on_bad_lines="error",
    )

def load_annotation(annotation_path: str | Path):
    """
    Accepts:
      - a single file path (train.jsonl OR a CSV)
      - OR a directory containing train.jsonl/test.jsonl/val.json(l)

    Returns:
      train_df, eval_df  (eval = val if present else test if present else empty)
    """
    ap = Path(annotation_path)
    if ap.is_dir():
        train_p = (ap / "train.jsonl")
        test_p  = (ap / "test.jsonl")
        val_p_jl = (ap / "val.jsonl")
        val_p_j  = (ap / "val.json")

        if not train_p.exists():
            raise FileNotFoundError(f"Missing {train_p}")
        train_df = _read_one(train_p)

        eval_df = pd.DataFrame()
        if val_p_jl.exists():
            eval_df = _read_one(val_p_jl)
        elif val_p_j.exists():
            eval_df = _read_one(val_p_j)

        if eval_df.empty and test_p.exists():
            eval_df = _read_one(test_p)
    else:
        # single file: load and then split by 'split' column if present
        df = _read_one(ap)
        cols = {c.lower(): c for c in df.columns}
        df.columns = [c.lower() for c in df.columns]
        if "split" in df.columns:
            train_df = df[df["split"].astype(str).str.lower().isin(["train","training"])].reset_index(drop=True)
            eval_df  = df[df["split"].astype(str).str.lower().isin(
                ["val","valid","validation","test"]
            )].reset_index(drop=True)
        else:
            # no split -> everything as train, empty eval; caller may random_split
            train_df, eval_df = df.reset_index(drop=True), pd.DataFrame()

    # Normalize key column names
    for need in ["image_path","mask_path","question","answer","position"]:
        if need not in train_df.columns:
            pass  # it might not exist in train if eval-only; will be checked when accessed

    return train_df.reset_index(drop=True), eval_df.reset_index(drop=True)



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