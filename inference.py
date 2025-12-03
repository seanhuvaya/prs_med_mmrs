import os
import random
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from data.dataset import PRSMedDataset
from train_prs_med import PRSMedModel, set_seed


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="PRS-Med Inference / Sample Visualization")

    # Core inputs
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.pth file, e.g. best_model_epoch_1.pth)",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Root directory of the dataset (same as training/eval)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "test"],
        help="Dataset split to sample from (default: val)",
    )

    # Model hyperparameters (must match training)
    parser.add_argument(
        "--image_size",
        type=int,
        default=1024,
        help="Image size (default: 1024)",
    )
    parser.add_argument(
        "--vision_encoder_type",
        type=str,
        default="tinysam",
        choices=["tinysam", "sam_med2d", "sammed2d"],
        help="Type of vision encoder to use: tinysam or sam_med2d (default: tinysam)",
    )
    parser.add_argument(
        "--vision_encoder_checkpoint",
        type=str,
        default="weights/tinysam_42.3.pth",
        help="Path to vision encoder checkpoint (TinySAM or SAM-Med2D) (default: weights/tinysam_42.3.pth)",
    )
    parser.add_argument(
        "--tinysam_checkpoint",
        type=str,
        default=None,
        help="[Deprecated] Path to TinySAM checkpoint (use --vision_encoder_checkpoint instead)",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=16,
        help="LoRA rank (default: 16)",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=16,
        help="LoRA alpha (default: 16)",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
        help="LoRA dropout (default: 0.05)",
    )

    # Sampling / output
    parser.add_argument(
        "--num_samples",
        type=int,
        default=4,
        help="Number of samples to visualize (default: 4)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="inference_outputs",
        help="Directory to save figure, CSV, and predicted masks (default: inference_outputs)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42)",
    )

    return parser.parse_args()


def to_display_image(img_tensor: torch.Tensor) -> np.ndarray:
    """
    Convert CxHxW tensor to HxWxC numpy image in [0, 1] for visualization.
    We don't assume a specific normalization; just min-max scale.
    """
    img = img_tensor.detach().cpu().float()

    if img.ndim != 3:
        raise ValueError(f"Expected image tensor of shape (C,H,W), got {img.shape}")

    # C x H x W -> H x W x C
    img = img.permute(1, 2, 0)

    # Min-max normalize to [0, 1]
    img = img - img.min()
    if img.max() > 0:
        img = img / img.max()

    return img.numpy()


def load_model_from_checkpoint(checkpoint_path: str, args, device: torch.device):
    """
    Load PRS-Med model from checkpoint (similar to your evaluation code).
    """
    print(f"Loading model from checkpoint: {checkpoint_path}")

    # args-like object for PRSMedModel
    class ModelArgs:
        pass

    margs = ModelArgs()
    margs.image_size = args.image_size
    margs.vision_encoder_type = args.vision_encoder_type
    margs.vision_encoder_checkpoint = args.vision_encoder_checkpoint
    # Support deprecated argument for backward compatibility
    if args.tinysam_checkpoint is not None:
        margs.tinysam_checkpoint = args.tinysam_checkpoint
    else:
        margs.tinysam_checkpoint = None
    margs.lora_rank = args.lora_rank
    margs.lora_alpha = args.lora_alpha
    margs.lora_dropout = args.lora_dropout

    # Initialize model
    model = PRSMedModel(margs, device)
    model.to(device)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        print(f"  Epoch in checkpoint: {checkpoint.get('epoch', 'unknown')}")
    else:
        state_dict = checkpoint
        print("  No 'model_state_dict' key found, using raw checkpoint as state_dict")

    model.load_state_dict(state_dict, strict=False)
    model.eval()

    print("✓ Model loaded and set to eval()")
    return model


def run_inference(args):
    # ------------------ Setup ------------------
    set_seed(args.seed)

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using device: CUDA ({torch.cuda.get_device_name(0)})")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using device: MPS")
    else:
        device = torch.device("cpu")
        print("Using device: CPU")

    # Output dirs
    output_dir = Path(args.output_dir)
    masks_dir = output_dir / "pred_masks"
    output_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model = load_model_from_checkpoint(args.checkpoint, args, device)

    # Load dataset
    print(f"\nLoading '{args.split}' split from {args.data_root}...")
    dataset = PRSMedDataset(split=args.split, data_root=args.data_root)
    print(f"Dataset size: {len(dataset)} samples")

    num_samples = min(args.num_samples, len(dataset))
    indices = random.sample(range(len(dataset)), num_samples)
    print(f"Sampling {num_samples} indices: {indices}")

    # ------------------ Visualization setup ------------------
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    if num_samples == 1:
        axes = axes[None, :]

    rows_for_table = []

    # ------------------ Inference loop ------------------
    model.eval()
    with torch.no_grad():
        for row_idx, idx in enumerate(indices):
            sample = dataset[idx]

            # ---- Inputs ----
            image = sample["image"].unsqueeze(0).to(device)  # [1, C, H, W]
            gt_mask = sample["mask"]
            if isinstance(gt_mask, torch.Tensor):
                gt_mask = gt_mask.unsqueeze(0).to(device) if gt_mask.ndim == 3 else gt_mask.to(device)
            else:
                gt_mask = torch.tensor(gt_mask, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

            question = sample.get("question", "")
            if isinstance(question, str):
                questions = [question]
            else:
                questions = question  # already batch-like

            # ---- Forward pass ----
            outputs = model(image, questions)
            z_mask = outputs["z_mask"]  # [B, 1, H, W] or [B, H, W]

            # Resize pred mask to GT size if needed
            if z_mask.shape != gt_mask.shape:
                z_mask_resized = F.interpolate(
                    z_mask,
                    size=gt_mask.shape[2:],
                    mode="bilinear",
                    align_corners=False,
                )
            else:
                z_mask_resized = z_mask

            pred_prob = torch.sigmoid(z_mask_resized)
            pred_bin = (pred_prob > 0.5).float()

            # Convert for visualization
            pred_mask_np = pred_bin[0, 0].cpu().numpy()
            gt_mask_np = gt_mask[0, 0].cpu().numpy() if gt_mask.ndim == 4 else gt_mask[0].cpu().numpy()
            img_disp = to_display_image(sample["image"])

            # ---- Plot row ----
            ax_img = axes[row_idx, 0]
            ax_gt = axes[row_idx, 1]
            ax_pred = axes[row_idx, 2]

            ax_img.imshow(img_disp)
            ax_img.set_title(f"Image (idx={idx})")
            ax_img.axis("off")

            ax_gt.imshow(gt_mask_np, cmap="gray")
            ax_gt.set_title("GT Mask")
            ax_gt.axis("off")

            ax_pred.imshow(pred_mask_np, cmap="gray")
            ax_pred.set_title("Predicted Mask")
            ax_pred.axis("off")

            # ---- Save predicted mask as PNG ----
            pred_mask_uint8 = (pred_mask_np * 255).astype(np.uint8)
            pred_mask_img = Image.fromarray(pred_mask_uint8)
            pred_mask_name = f"pred_mask_idx{idx}.png"
            pred_mask_path = masks_dir / pred_mask_name
            pred_mask_img.save(pred_mask_path)

            # ---- Optional: decode predicted text ----
            try:
                mllm_model = model.module if hasattr(model, "module") else model
                pred_ids = torch.argmax(outputs["z_txt_logits"], dim=-1)
                pred_text_batch = mllm_model.mllm.processor.batch_decode(
                    pred_ids, skip_special_tokens=True
                )
                pred_text = pred_text_batch[0] if len(pred_text_batch) > 0 else ""
            except Exception as e:
                print(f"Warning: failed to decode text for idx={idx}: {e}")
                pred_text = ""

            # ---- Collect info for table ----
            image_path = sample.get("image_path", "")
            mask_path = sample.get("mask_path", "")
            answer = sample.get("answer", "")

            rows_for_table.append(
                {
                    "index": idx,
                    "image_path": image_path,
                    "mask_path": mask_path,
                    "pred_mask_path": str(pred_mask_path),
                    "question": question,
                    "gt_answer": answer,
                    "pred_answer": pred_text,
                }
            )

    # ------------------ Save figure & CSV ------------------
    checkpoint_name = Path(args.checkpoint).stem
    fig_path = output_dir / f"sample_predictions_{args.split}_{checkpoint_name}.png"
    csv_path = output_dir / f"sample_predictions_{args.split}_{checkpoint_name}.csv"

    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close(fig)

    df = pd.DataFrame(rows_for_table)
    df.to_csv(csv_path, index=False)

    print(f"\n✓ Figure saved to: {fig_path}")
    print(f"✓ Table saved to:  {csv_path}")
    print("\nPreview of table:")
    print(df.head())


def main():
    args = parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()