import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from typing import List, Dict, Tuple
from dotenv import load_dotenv
from scipy.spatial.distance import directed_hausdorff
from scipy.ndimage import binary_erosion

load_dotenv()


def dice_coefficient(pred_mask: torch.Tensor, true_mask: torch.Tensor, smooth=1e-6):
    """
    Compute Dice coefficient per sample.
    Returns per-sample Dice scores for proper aggregation across dataset.
    """
    # Resize predicted mask to match ground truth resolution
    if pred_mask.shape != true_mask.shape:
        pred_mask = F.interpolate(pred_mask, size=true_mask.shape[2:], mode="bilinear", align_corners=False)
    
    pred_mask = torch.sigmoid(pred_mask)
    pred_mask = (pred_mask > 0.5).float()
    intersection = (pred_mask * true_mask).sum(dim=(1, 2, 3))
    dice = (2. * intersection + smooth) / (
        pred_mask.sum(dim=(1, 2, 3)) + true_mask.sum(dim=(1, 2, 3)) + smooth
    )
    return dice.cpu().numpy()  # Return per-sample scores


def iou_score(pred_mask: torch.Tensor, true_mask: torch.Tensor, smooth=1e-6):
    """
    Compute IoU score per sample.
    Returns per-sample IoU scores for proper aggregation across dataset.
    """
    # Ensure both tensors have same spatial resolution
    if pred_mask.shape != true_mask.shape:
        pred_mask = F.interpolate(pred_mask, size=true_mask.shape[2:], mode="bilinear", align_corners=False)

    pred_mask = torch.sigmoid(pred_mask)
    pred_mask = (pred_mask > 0.5).float()
    intersection = (pred_mask * true_mask).sum(dim=(1, 2, 3))
    union = (pred_mask + true_mask - pred_mask * true_mask).sum(dim=(1, 2, 3))
    iou = (intersection + smooth) / (union + smooth)
    return iou.cpu().numpy()  # Return per-sample scores


def hausdorff_distance_per_sample(pred_mask: torch.Tensor, true_mask: torch.Tensor, percentile: float = 95.0):
    """
    Compute 95th percentile Hausdorff Distance (HD95) per sample.
    
    The Hausdorff distance measures the maximum distance between boundaries of 
    predicted and ground truth masks. HD95 uses the 95th percentile to be more 
    robust to outliers.
    
    Args:
        pred_mask: (B, 1, H, W) predicted mask logits
        true_mask: (B, 1, H, W) ground truth binary mask
        percentile: Percentile for HD95 (default: 95.0)
    
    Returns:
        Per-sample HD95 distances (numpy array)
    """
    # Resize if needed
    if pred_mask.shape != true_mask.shape:
        pred_mask = F.interpolate(pred_mask, size=true_mask.shape[2:], mode="bilinear", align_corners=False)
    
    # Convert to binary masks
    pred_mask = torch.sigmoid(pred_mask)
    pred_mask = (pred_mask > 0.5).float()
    true_mask = true_mask.float()
    
    batch_size = pred_mask.shape[0]
    hd95_scores = []
    
    for i in range(batch_size):
        pred_np = pred_mask[i, 0].cpu().numpy()
        true_np = true_mask[i, 0].cpu().numpy()
        
        # Skip if either mask is empty
        if pred_np.sum() == 0 or true_np.sum() == 0:
            hd95_scores.append(float('inf'))
            continue
        
        # Get boundary points using morphological operations
        pred_boundary = pred_np.astype(bool) & ~binary_erosion(pred_np.astype(bool))
        true_boundary = true_np.astype(bool) & ~binary_erosion(true_np.astype(bool))
        
        # Get coordinates of boundary points
        pred_coords = np.column_stack(np.where(pred_boundary))
        true_coords = np.column_stack(np.where(true_boundary))
        
        if len(pred_coords) == 0 or len(true_coords) == 0:
            hd95_scores.append(float('inf'))
            continue
        
        # For HD95, compute distances from each point to the other set
        # and take the 95th percentile
        distances_forward = []
        for point in pred_coords:
            dists = np.sqrt(((true_coords - point) ** 2).sum(axis=1))
            distances_forward.append(dists.min())
        
        distances_backward = []
        for point in true_coords:
            dists = np.sqrt(((pred_coords - point) ** 2).sum(axis=1))
            distances_backward.append(dists.min())
        
        all_distances = distances_forward + distances_backward
        if len(all_distances) > 0:
            hd95 = np.percentile(all_distances, percentile)
        else:
            hd95 = float('inf')
        
        hd95_scores.append(hd95)
    
    return np.array(hd95_scores)  # Return per-sample scores


def evaluate_position_reasoning_simple(pred_texts: List[str], gt_texts: List[str]) -> Dict[str, float]:
    """
    Simple keyword-based position reasoning evaluation.
    Extracts position keywords and checks for matches.
    """
    # Common position keywords
    position_keywords = [
        "top-left", "top left", "upper-left", "upper left",
        "top-right", "top right", "upper-right", "upper right",
        "bottom-left", "bottom left", "lower-left", "lower left",
        "bottom-right", "bottom right", "lower-right", "lower right",
        "center", "centre", "middle", "central",
        "top", "upper", "bottom", "lower",
        "left", "right",
    ]
    
    def extract_position_keywords(text: str) -> set:
        """Extract position keywords from text."""
        text_lower = text.lower()
        found = set()
        for keyword in position_keywords:
            if keyword in text_lower:
                found.add(keyword)
        return found
    
    exact_matches = 0
    keyword_matches = 0
    total = len(pred_texts)
    
    for pred, gt in zip(pred_texts, gt_texts):
        # Exact match (case-insensitive)
        if pred.lower().strip() == gt.lower().strip():
            exact_matches += 1
            keyword_matches += 1
            continue
        
        # Keyword-based match
        pred_keywords = extract_position_keywords(pred)
        gt_keywords = extract_position_keywords(gt)
        
        if pred_keywords and gt_keywords:
            # Check if there's overlap in keywords
            if pred_keywords.intersection(gt_keywords):
                keyword_matches += 1
    
    exact_acc = exact_matches / total if total > 0 else 0.0
    keyword_acc = keyword_matches / total if total > 0 else 0.0
    
    return {
        "exact_match_acc": exact_acc,
        "keyword_match_acc": keyword_acc,
    }


def evaluate_text_reasoning(pred_texts: List[str], gt_texts: List[str]) -> Dict[str, float]:
    """
    Position reasoning evaluation using keyword matching.
    This matches standard evaluation practices in medical segmentation papers.
    
    Args:
        pred_texts: List of predicted position descriptions
        gt_texts: List of ground truth position descriptions
    
    Returns:
        Dictionary with evaluation metrics
    """
    # Use simple keyword-based evaluation (standard in papers)
    return evaluate_position_reasoning_simple(pred_texts, gt_texts)



def evaluate_prs_med(model, data_loader, device):
    """
    Evaluation of PRS-Med model following standard medical segmentation practices.
    Metrics are computed per-sample and aggregated across the entire dataset.
    
    Args:
        model: PRS-Med model
        data_loader: DataLoader with test data
        device: Device to run evaluation on
    
    Returns:
        Dictionary with all evaluation metrics
    """
    model.to(device)
    model.eval()
    
    # Collect per-sample scores across entire dataset
    all_dice_scores = []
    all_iou_scores = []
    all_hd95_scores = []
    pred_texts, gt_texts = [], []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            # Handle different data loader formats
            if isinstance(batch, dict):
                images = batch['image'].to(device)
                gt_masks = batch['mask'].to(device)
                questions = batch['question']
                gt_answers = batch['answer']
            else:
                # Legacy format: (images, questions, gt_masks, gt_tokens, gt_answers)
                images, questions, gt_masks, _, gt_answers = batch
                images = images.to(device)
                gt_masks = gt_masks.to(device)
            
            outputs = model(images, questions)

            # Segmentation metrics - compute per-sample
            dice_per_sample = dice_coefficient(outputs["z_mask"], gt_masks)
            iou_per_sample = iou_score(outputs["z_mask"], gt_masks)
            hd95_per_sample = hausdorff_distance_per_sample(outputs["z_mask"], gt_masks)
            
            all_dice_scores.extend(dice_per_sample)
            all_iou_scores.extend(iou_per_sample)
            all_hd95_scores.extend(hd95_per_sample)

            # Convert token predictions back to text
            pred_ids = torch.argmax(outputs["z_txt_logits"], dim=-1)
            # Handle both regular and DDP-wrapped models
            mllm_model = model.module if hasattr(model, 'module') else model
            pred_text_batch = mllm_model.mllm.processor.batch_decode(pred_ids, skip_special_tokens=True)
            pred_texts.extend(pred_text_batch)
            gt_texts.extend(gt_answers)

    # Aggregate metrics across all samples (standard practice)
    all_dice_scores = np.array(all_dice_scores)
    all_iou_scores = np.array(all_iou_scores)
    all_hd95_scores = np.array(all_hd95_scores)
    
    # Compute mean Dice and IoU
    mdice = np.mean(all_dice_scores)
    miou = np.mean(all_iou_scores)
    
    # Compute mean HD95, excluding inf values (standard practice)
    valid_hd95 = all_hd95_scores[all_hd95_scores != float('inf')]
    if len(valid_hd95) > 0:
        mhd95 = np.mean(valid_hd95)
    else:
        mhd95 = float('inf')
    
    print(f"\n{'='*60}")
    print(f"Segmentation Metrics (computed per-sample, aggregated across dataset):")
    print(f"  mDice:  {mdice:.4f}")
    print(f"  mIoU:   {miou:.4f}")
    print(f"  mHD95:  {mhd95:.2f}" if mhd95 != float('inf') else f"  mHD95:  inf")
    print(f"{'='*60}")

    # Position reasoning metrics
    text_metrics = evaluate_text_reasoning(pred_texts, gt_texts)
    
    print(f"\nPosition Reasoning Metrics:")
    print(f"  Exact Match Accuracy:  {text_metrics.get('exact_match_acc', 0):.4f}")
    print(f"  Keyword Match Accuracy: {text_metrics.get('keyword_match_acc', 0):.4f}")
    print(f"{'='*60}\n")

    return {
        "mDice": float(mdice),
        "mIoU": float(miou),
        "mHD95": float(mhd95) if mhd95 != float('inf') else float('inf'),
        **text_metrics
    }


def load_model_from_checkpoint(checkpoint_path: str, args, device):
    """
    Load PRS-Med model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file (.pth)
        args: Arguments object with model configuration
        device: Device to load model on
    
    Returns:
        Loaded model
    """
    from train_prs_med import PRSMedModel
    
    print(f"Loading model from checkpoint: {checkpoint_path}")
    
    # Initialize model
    model = PRSMedModel(args, device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        state_dict = checkpoint
        print("Loaded checkpoint (no epoch info found)")
    
    # Load state dict
    model.load_state_dict(state_dict, strict=False)
    
    print("✓ Model loaded successfully")
    return model


def main():
    import argparse
    import json
    from pathlib import Path
    from data.dataset import PRSMedDataset
    
    parser = argparse.ArgumentParser(description='Evaluate PRS-Med Model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint (.pth file)')
    parser.add_argument('--data_root', type=str, required=True,
                       help='Root directory of the dataset')
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'val', 'test'],
                       help='Dataset split to evaluate on (default: test)')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for evaluation (default: 8)')
    parser.add_argument('--num_workers', type=int, default=2,
                       help='Number of data loader workers (default: 2)')
    parser.add_argument('--image_size', type=int, default=1024,
                       help='Image size (default: 1024)')
    parser.add_argument('--tinysam_checkpoint', type=str, default='weights/tinysam_42.3.pth',
                       help='Path to TinySAM checkpoint (default: weights/tinysam_42.3.pth)')
    parser.add_argument('--lora_rank', type=int, default=16,
                       help='LoRA rank (default: 16)')
    parser.add_argument('--lora_alpha', type=int, default=16,
                       help='LoRA alpha (default: 16)')
    parser.add_argument('--lora_dropout', type=float, default=0.05,
                       help='LoRA dropout (default: 0.05)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Directory to save evaluation results (optional)')
    parser.add_argument('--specific_dataset', type=str, default=None,
                       help='Evaluate on specific dataset only (optional)')
    
    args = parser.parse_args()
    
    # Set device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using device: CUDA ({torch.cuda.get_device_name(0)})")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using device: MPS")
    else:
        device = torch.device('cpu')
        print("Using device: CPU")
    
    # Load model
    model = load_model_from_checkpoint(args.checkpoint, args, device)
    model.eval()
    
    # Load dataset
    print(f"\nLoading {args.split} dataset from {args.data_root}...")
    test_dataset = PRSMedDataset(
        split=args.split,
        data_root=args.data_root,
        specific_dataset=args.specific_dataset
    )
    
    print(f"Dataset size: {len(test_dataset)} samples")
    if args.specific_dataset:
        print(f"Evaluating on dataset: {args.specific_dataset}")
    
    # Create data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Run evaluation
    print(f"\n{'='*60}")
    print(f"Starting evaluation on {args.split} split")
    print(f"{'='*60}\n")
    
    metrics = evaluate_prs_med(model, test_loader, device)
    
    # Save results if output directory specified
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create results filename
        checkpoint_name = Path(args.checkpoint).stem
        results_file = output_dir / f"results_{args.split}_{checkpoint_name}.json"
        
        # Add metadata
        results = {
            'checkpoint': args.checkpoint,
            'split': args.split,
            'dataset_size': len(test_dataset),
            'specific_dataset': args.specific_dataset,
            'metrics': metrics
        }
        
        # Save JSON
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ Results saved to: {results_file}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("Evaluation Summary:")
    print(f"{'='*60}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Split: {args.split}")
    print(f"Dataset size: {len(test_dataset)}")
    print(f"\nSegmentation Metrics:")
    print(f"  mDice:  {metrics['mDice']:.4f}")
    print(f"  mIoU:   {metrics['mIoU']:.4f}")
    print(f"  mHD95:  {metrics['mHD95']:.2f}" if metrics['mHD95'] != float('inf') else f"  mHD95:  inf")
    print(f"\nPosition Reasoning Metrics:")
    print(f"  Exact Match Accuracy:  {metrics.get('exact_match_acc', 0):.4f}")
    print(f"  Keyword Match Accuracy: {metrics.get('keyword_match_acc', 0):.4f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()