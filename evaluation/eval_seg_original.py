import argparse
import os
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm


def dice_coefficient(pred_mask: torch.Tensor, true_mask: torch.Tensor, smooth: float = 1e-6) -> np.ndarray:
    """
    Compute Dice coefficient per sample (binary masks).

    This follows the style used in the original PRS-Med evaluation:
    - Resize prediction to GT size if needed
    - Threshold at 0.5
    - Compute per-sample Dice, then average later
    """
    # Ensure 4D
    if pred_mask.ndim == 2:
        pred_mask = pred_mask.unsqueeze(0).unsqueeze(0)
    elif pred_mask.ndim == 3:
        pred_mask = pred_mask.unsqueeze(1)

    if true_mask.ndim == 2:
        true_mask = true_mask.unsqueeze(0).unsqueeze(0)
    elif true_mask.ndim == 3:
        true_mask = true_mask.unsqueeze(1)

    # Resize predicted mask to match ground truth resolution if needed
    if pred_mask.shape != true_mask.shape:
        pred_mask = F.interpolate(
            pred_mask,
            size=true_mask.shape[2:],
            mode="bilinear",
            align_corners=False,
        )

    # Convert logits/probabilities to binary mask
    pred_mask = torch.sigmoid(pred_mask)
    pred_mask = (pred_mask > 0.5).float()
    true_mask = true_mask.float()

    intersection = (pred_mask * true_mask).sum(dim=(1, 2, 3))
    denom = pred_mask.sum(dim=(1, 2, 3)) + true_mask.sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + smooth) / (denom + smooth)
    return dice.cpu().numpy()  # (B,)


def iou_score(pred_mask: torch.Tensor, true_mask: torch.Tensor, smooth: float = 1e-6) -> np.ndarray:
    """
    Compute IoU score per sample (binary masks).
    """
    # Ensure 4D
    if pred_mask.ndim == 2:
        pred_mask = pred_mask.unsqueeze(0).unsqueeze(0)
    elif pred_mask.ndim == 3:
        pred_mask = pred_mask.unsqueeze(1)

    if true_mask.ndim == 2:
        true_mask = true_mask.unsqueeze(0).unsqueeze(0)
    elif true_mask.ndim == 3:
        true_mask = true_mask.unsqueeze(1)

    if pred_mask.shape != true_mask.shape:
        pred_mask = F.interpolate(
            pred_mask,
            size=true_mask.shape[2:],
            mode="bilinear",
            align_corners=False,
        )

    pred_mask = torch.sigmoid(pred_mask)
    pred_mask = (pred_mask > 0.5).float()
    true_mask = true_mask.float()

    intersection = (pred_mask * true_mask).sum(dim=(1, 2, 3))
    union = (pred_mask + true_mask - pred_mask * true_mask).sum(dim=(1, 2, 3))
    iou = (intersection + smooth) / (union + smooth)
    return iou.cpu().numpy()  # (B,)


def load_mask_as_tensor(path: str) -> torch.Tensor:
    """Load a mask image (gt or prediction) as a float tensor in [0, 1]."""
    if path is None or not os.path.exists(path):
        raise FileNotFoundError(f"Mask file not found: {path}")

    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Failed to read mask: {path}")

    mask = mask.astype(np.float32) / 255.0  # [0,1]
    tensor = torch.from_numpy(mask)  # H,W
    return tensor


def evaluate_segmentation(results_csv: str) -> Tuple[float, float]:
    """
    Evaluate segmentation performance using the results CSV produced by infer_original.py.

    Expects columns:
      - mask_path: path to GT mask
      - pred_mask_path: path to predicted mask
    """
    df = pd.read_csv(results_csv)

    if "mask_path" not in df.columns or "pred_mask_path" not in df.columns:
        raise ValueError(
            "results CSV must contain 'mask_path' and 'pred_mask_path' columns. "
            "Run infer_original.py first to generate this CSV."
        )

    dice_all: List[float] = []
    iou_all: List[float] = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating segmentation"):
        mask_path = row["mask_path"]
        pred_path = row["pred_mask_path"]

        if not isinstance(mask_path, str) or not isinstance(pred_path, str):
            continue
        if not os.path.exists(mask_path) or not os.path.exists(pred_path):
            continue

        try:
            gt_tensor = load_mask_as_tensor(mask_path)
            pred_tensor = load_mask_as_tensor(pred_path)

            # Add batch dimension and compute metrics
            d = dice_coefficient(pred_tensor.unsqueeze(0), gt_tensor.unsqueeze(0))[0]
            i = iou_score(pred_tensor.unsqueeze(0), gt_tensor.unsqueeze(0))[0]

            dice_all.append(float(d))
            iou_all.append(float(i))
        except Exception as e:
            print(f"Skipping row {idx} due to error: {e}")
            continue

    if len(dice_all) == 0:
        raise RuntimeError("No valid samples found for evaluation.")

    mean_dice = float(np.mean(dice_all))
    mean_iou = float(np.mean(iou_all))

    return mean_dice, mean_iou


def parse_args():
    parser = argparse.ArgumentParser(
        description="Segmentation evaluation for PRS-Med-MMRS (original model outputs)"
    )
    parser.add_argument(
        "--results_csv",
        type=str,
        required=True,
        help="Path to results CSV produced by infer_original.py (e.g., inference_outputs_original/results_test.csv)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Optional path to save a small CSV with summary metrics",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    results_csv = args.results_csv

    if not os.path.exists(results_csv):
        raise FileNotFoundError(f"results CSV not found: {results_csv}")

    print(f"Evaluating segmentation using: {results_csv}")
    mean_dice, mean_iou = evaluate_segmentation(results_csv)

    print("\n=== Segmentation Metrics (PRS-Med style) ===")
    print(f"Mean Dice (mDice): {mean_dice:.4f}")
    print(f"Mean IoU  (mIoU):  {mean_iou:.4f}")

    if args.output_path is not None:
        # Add timestamp prefix if output_path is a directory or doesn't have timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = Path(args.output_path)
        
        # If it's a directory, create a timestamped file inside it
        if out_path.is_dir() or (not out_path.suffix and not out_path.exists()):
            out_path = out_path / f"seg_eval_results_{timestamp}.csv"
        # If it's a file path but doesn't have timestamp, add it
        elif not any(char.isdigit() for char in out_path.stem[-15:]):  # Check if last 15 chars have digits
            # Insert timestamp before extension
            out_path = out_path.parent / f"{out_path.stem}_{timestamp}{out_path.suffix}"
        
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(
            [
                {
                    "results_csv": results_csv,
                    "mean_dice": mean_dice,
                    "mean_iou": mean_iou,
                }
            ]
        )
        df.to_csv(out_path, index=False)
        print(f"\nSaved summary metrics to: {out_path}")


if __name__ == "__main__":
    main()


