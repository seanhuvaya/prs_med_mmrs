"""
Evaluation metrics for PRS-Med model.
Includes segmentation metrics and position reasoning accuracy.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def dice_coefficient(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> float:
    """Calculate Dice coefficient."""
    pred = pred.flatten()
    target = target.flatten()
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def iou_score(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> float:
    """Calculate Intersection over Union (IoU)."""
    pred = pred.flatten()
    target = target.flatten()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + smooth) / (union + smooth)

def hausdorff_distance(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Calculate Hausdorff distance between contours."""
    # Convert to numpy for contour detection
    pred_np = pred.squeeze().cpu().numpy()
    target_np = target.squeeze().cpu().numpy()
    
    # Get contours
    from skimage import measure
    pred_contours = measure.find_contours(pred_np, 0.5)
    target_contours = measure.find_contours(target_np, 0.5)
    
    if len(pred_contours) == 0 or len(target_contours) == 0:
        return float('inf')
    
    # Calculate Hausdorff distance
    pred_points = np.vstack(pred_contours)
    target_points = np.vstack(target_contours)
    
    def _hausdorff_dist(points1, points2):
        distances = np.sqrt(np.sum((points1[:, np.newaxis] - points2)**2, axis=2))
        return max(np.max(np.min(distances, axis=1)), np.max(np.min(distances, axis=0)))
    
    return _hausdorff_dist(pred_points, target_points)

def position_accuracy(pred_positions: List[str], target_positions: List[str]) -> float:
    """Calculate position reasoning accuracy."""
    if len(pred_positions) != len(target_positions):
        return 0.0
    
    correct = sum(1 for pred, target in zip(pred_positions, target_positions) 
                 if pred.lower().strip() == target.lower().strip())
    return correct / len(pred_positions)

def evaluate_segmentation(pred_masks: torch.Tensor, target_masks: torch.Tensor) -> Dict[str, float]:
    """Evaluate segmentation performance."""
    metrics = {}
    
    # Convert to binary
    pred_binary = (pred_masks > 0.5).float()
    
    # Calculate metrics for each sample
    dice_scores = []
    iou_scores = []
    hausdorff_scores = []
    
    for i in range(pred_masks.shape[0]):
        dice_scores.append(dice_coefficient(pred_binary[i], target_masks[i]))
        iou_scores.append(iou_score(pred_binary[i], target_masks[i]))
        hausdorff_scores.append(hausdorff_distance(pred_binary[i], target_masks[i]))
    
    metrics['dice_mean'] = np.mean(dice_scores)
    metrics['dice_std'] = np.std(dice_scores)
    metrics['iou_mean'] = np.mean(iou_scores)
    metrics['iou_std'] = np.std(iou_scores)
    metrics['hausdorff_mean'] = np.mean([h for h in hausdorff_scores if h != float('inf')])
    
    return metrics

def evaluate_position_reasoning(pred_answers: List[str], target_answers: List[str]) -> Dict[str, float]:
    """Evaluate position reasoning performance."""
    metrics = {}
    
    # Exact match accuracy
    metrics['exact_match'] = position_accuracy(pred_answers, target_answers)
    
    # Keyword-based evaluation
    position_keywords = ['top', 'bottom', 'left', 'right', 'center', 'near-center']
    
    keyword_matches = 0
    for pred, target in zip(pred_answers, target_answers):
        pred_lower = pred.lower()
        target_lower = target.lower()
        
        pred_keywords = [kw for kw in position_keywords if kw in pred_lower]
        target_keywords = [kw for kw in position_keywords if kw in target_lower]
        
        if set(pred_keywords) == set(target_keywords):
            keyword_matches += 1
    
    metrics['keyword_match'] = keyword_matches / len(pred_answers)
    
    return metrics

def comprehensive_evaluation(model, dataloader, device: str = "cpu") -> Dict[str, float]:
    """Comprehensive evaluation of PRS-Med model."""
    model.eval()
    
    all_pred_masks = []
    all_target_masks = []
    all_pred_answers = []
    all_target_answers = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Move to device
            images = batch["image"].to(device)
            questions = batch["question"]
            target_masks = batch["mask"].to(device)
            target_answers = batch["answer"]
            
            # Forward pass
            outputs = model(images, questions)
            pred_masks = outputs["mask"]
            
            # Store results
            all_pred_masks.append(pred_masks.cpu())
            all_target_masks.append(target_masks.cpu())
            all_pred_answers.extend(questions)  # Using questions as proxy for now
            all_target_answers.extend(target_answers)
    
    # Concatenate all results
    all_pred_masks = torch.cat(all_pred_masks, dim=0)
    all_target_masks = torch.cat(all_target_masks, dim=0)
    
    # Evaluate segmentation
    seg_metrics = evaluate_segmentation(all_pred_masks, all_target_masks)
    
    # Evaluate position reasoning
    pos_metrics = evaluate_position_reasoning(all_pred_answers, all_target_answers)
    
    # Combine metrics
    all_metrics = {**seg_metrics, **pos_metrics}
    
    return all_metrics
