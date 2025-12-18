# PRS-Med: Position Reasoning Segmentation with Vision-Language Model
## Quickstart
## 1. Setup environment
Run the ```setup.sh``` script to automatically prepare the project environment:
```shell
sh setup.sh <data_download_dir> <project_repo_dir>
```
### What the script does
#### 1. Downloads the dataset
Fetches the training dataset from a public Amazon S3 folder used for this project’s models.
#### 2. Clones the project repository
Clones the full project source code into the specified repository directory.
#### 3. Installs and syncs dependencies**
Installs [uv](https://github.com/astral-sh/uv) and synchronizes all project dependencies to ensure a reproducible environment.
### Dataset (public S3)

The dataset is publicly available at:
```shell
s3://<your-bucket-name>/<public-dataset-folder>/
```
or via HTTPS:
```http request
https://<your-bucket-name>.s3.amazonaws.com/<public-dataset-folder>/
```
### Dataset Structure
```text
data/
├── our-dataset/
├── prs-med-dataset/
```

### 1. Prepare Your Data

Organize your data in the MMRS format:

```
data/
├── images_and_masks/
│   ├── image_001.png
        ├── image_001.png
        └── ...
    ├── image_001.png
        ├── mask_001.png
        └── ...
│   ├── mask_001.png
│   └── ...
└── csv/
    ├── train.csv
    ├── val.csv
    └── test.csv
```

CSV format should include columns: `image_path`, `mask_path`, `question`, `answer`, `modality`