uv run python -m train_original \
  --data_root /workspace/data/our-dataset \
  --ann_paths /workspace/data/our-dataset/annotations/head_and_neck.csv,/workspace/data/annotations/prostate.csv \
  --vlm_path microsoft/llava-med-v1.5-mistral-7b \
  --sam_ckpt weights/tinysam_42.3.pth \
  --batch_size 8 \
  --epochs 20