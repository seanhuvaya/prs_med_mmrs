import pandas as pd
import numpy as np
import os
from random import shuffle
import random
from PIL import Image
import requests
from io import BytesIO
from sklearn.utils import shuffle

def load_annotation(annotation_path):
    """
    Load annotation CSV files. Supports:
    - Single file path (str)
    - List of file paths (list)
    - Directory path containing CSV files
    """
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
    
    # Return full dataframe - split handling is done in dataset class
    # This maintains compatibility with original code
    if 'split' in df.columns:
        train_df = df[df['split'] == 'train'].copy()
        test_df = df[df['split'].isin(['test', 'val'])].copy() if 'test' in df['split'].values else pd.DataFrame()
    else:
        # If no split column, use 90/10 split
        train_size = int(len(df) * 0.9)
        train_df = df.iloc[:train_size].copy()
        test_df = df.iloc[train_size:].copy()
    
    return train_df, test_df

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

