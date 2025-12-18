# PRS-Med: Position Reasoning Segmentation with Vision-Language Model
## Quickstart
## ⚙️ Setup environment
Run the ```setup.sh``` script to automatically prepare the project environment:
```shell
sh setup.sh <data_download_dir> <project_repo_dir>
```
### What the script does
#### Downloads the dataset
Downloads the training dataset from a public Amazon S3 bucket used by the models in this project.
If the AWS CLI is not already installed, the script installs it automatically before downloading the data.
#### Clones the project repository
Clones the full project source code into the specified repository directory.
#### Installs and syncs dependencies**
Installs [uv](https://github.com/astral-sh/uv) and synchronizes all project dependencies to ensure a reproducible environment.
### Dataset (public S3)
The dataset is publicly available at:
```shell
s3://prs-med-experiments/data/
```
or via HTTPS:
```http request
https://prs-med-experiments.s3.us-east-1.amazonaws.com/data/
```
### Dataset Structure
```text
data/
├── our-dataset/
    ├── annotations
        ├── head_and_neck.csv
        └── prostate.csv
    ├── head_and_neck
        ├── test_images
            ├── 0204_channel_08.png
            └── ...
        ├── test_masks
            ├── 0204_channel_08.png
            └── ...
        ├── train_images
        ├── train_masks
        ├── val_images
        └── val_masks
    └── prostate
        └── ...
├── prs-med-dataset/
    ├── annotations
        ├── brain_tumors_ct_scan.csv
        └── ...
    ├── brain_tumors_ct_scan
        ├── test_images
            ├── 100.png
            └── ...
        ├── test_masks
            ├── 100.png
            └── ...
        ├── train_images
        ├── train_masks
        ├── val_images
        └── val_masks
    └── ...
```
## 🛠️ Model Training
This project supports training on both the **PRS-Med dataset** (to reproduce reported results) and **custom research datasets** using the same training pipeline.
### Training on PRS-Med (reproducibility)
To reproduce results on the PRS-Med benchmark, run:
```shell
uv run python -m train \
    --data_root /path/to/data/prs-med-dataset \
    --ann_paths "/path/to/data/prs-med-dataset/annotations/brain_tumors_ct_scan.csv,/path/to/data/prs-med-dataset/annotations/breast_tumors_ct_scan.csv,/path/to/data/prs-med-dataset/annotations/lung_CT.csv,/path/to/data/prs-med-dataset/annotations/lung_Xray.csv,/path/to/data/prs-med-dataset/annotations/polyp_endoscopy.csv,/path/to/data/prs-med-dataset/annotations/skin_rgbimage.csv" \
    --vlm_path microsoft/llava-med-v1.5-mistral-7b \
    --sam_ckpt weights/tinysam_42.3.pth \
    --batch_size 8 \
    --epochs 20
```
This configuration uses the official PRS-Med annotations and the LLaVA-Med v1.5 (Mistral-7B) vision–language backbone.

### Training on a custom research dataset
To train the model on our research dataset, specify the corresponding data root and annotation files:
```shell
uv run python -m train \
    --data_root /path/to/data/our-dataset \
    --ann_paths "/path/to/data/our-dataset/annotations/head_and_neck.csv,/path/to/data/our-dataset/annotations/prostate.csv " \
    --vlm_path microsoft/llava-med-v1.5-mistral-7b \
    --sam_ckpt weights/tinysam_42.3.pth \
    --batch_size 8 \
    --epochs 20
```
### Notes
- ```--ann_paths``` accepts a comma-separated list of annotation CSV files. 
- All datasets must follow the directory structure described in the Dataset Structure section. 
- Hyperparameters such as batch_size and epochs can be adjusted based on available GPU memory and dataset size.

## 📚 References

- **Paper**: [PRS-Med: Position Reasoning Segmentation with Vision-Language Model in Medical Imaging](https://arxiv.org/pdf/2505.11872)
- **TinySAM**: [TinySAM: Segment Anything Model in Less Than 50MB](https://github.com/xinghaochen/TinySAM)
- **LLaVA-Med**: [LLaVA-Med: Training a Large Language-and-Vision Assistant for Biomedicine](https://github.com/microsoft/LLaVA-Med)