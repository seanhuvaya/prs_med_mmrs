# Dataset Preparation Guide

This directory contains utility scripts to prepare the prostate and head_and_neck datasets for use with PRS-Med.

## Prostate Dataset Reorganization

The prostate dataset needs to be reorganized to flatten the train/test split structure.

### Before:
```
prostate/
  images/
    test/
      example1.png
    train/
      example2.png
  masks/
    test/
      example1.png
    train/
      example2.png
```

### After:
```
prostate/
  images/
    example1.png
    example2.png
  masks/
    example1.png
    example2.png
```

### Usage:
```bash
cd utils
uv run python reorganize_prostate.py --prostate_root /path/to/prostate
```

## Head and Neck Dataset Conversion

The head_and_neck dataset contains .npz files with 64 channels each. This script converts each channel to a separate PNG image.

### Before:
```
head_and_neck/
  images/
    example.npz  (64 channels)
  masks/
    example.npz  (64 channels)
```

### After:
```
head_and_neck/
  images/
    example_channel_00.png
    example_channel_01.png
    ...
    example_channel_63.png
  masks/
    example_channel_00.png
    example_channel_01.png
    ...
    example_channel_63.png
```

### Usage:
```bash
cd utils
uv run python convert_head_neck_npz.py --head_neck_root /path/to/head_and_neck [--remove_npz]
```

The `--remove_npz` flag will delete the original .npz files after conversion to save space.

## Combined Preparation

Use the main script to prepare both datasets at once:

```bash
cd utils
uv run python prepare_datasets.py \
    --prostate_root /path/to/prostate \
    --head_neck_root /path/to/head_and_neck \
    [--remove_npz]
```

Or if both datasets are in a common parent directory:

```bash
cd utils
uv run python prepare_datasets.py \
    --data_root /path/to/data_root \
    [--remove_npz]
```

This will automatically find `prostate/` and `head_and_neck/` subdirectories in the data_root.

## Notes

- The scripts preserve original files by default (unless `--remove_npz` is used)
- Progress bars show conversion status
- The scripts handle various .npz array shapes automatically
- Images are normalized to 0-255 range before saving as PNG

