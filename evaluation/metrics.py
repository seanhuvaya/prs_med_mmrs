"""Evaluation metrics for PRS-Med model."""

from typing import List, Dict

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_erosion


def dice_coefficient(pred_mask: torch.Tensor, true_mask: torch.Tensor, smooth=1e-6):
    """
    Compute Dice coefficient per sample.
    Returns per-sample Dice scores for proper aggregation across dataset.
    
    Args:
        pred_mask: Predicted mask logits (B, 1, H, W)
        true_mask: Ground truth mask (B, 1, H, W)
        smooth: Smoothing factor to avoid division by zero
    
    Returns:
        Per-sample Dice scores (numpy array)
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
    
    Args:
        pred_mask: Predicted mask logits (B, 1, H, W)
        true_mask: Ground truth mask (B, 1, H, W)
        smooth: Smoothing factor to avoid division by zero
    
    Returns:
        Per-sample IoU scores (numpy array)
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
    
    Args:
        pred_texts: List of predicted position descriptions
        gt_texts: List of ground truth position descriptions
    
    Returns:
        Dictionary with exact_match_acc and keyword_match_acc
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


def evaluate_text_reasoning_ensemble_llm(
    pred_texts: List[str], 
    gt_texts: List[str],
    use_llm: bool = True
) -> Dict[str, float]:
    """
    Position reasoning evaluation using ensemble of LLM agents (paper method).
    
    Following the PRS-Med paper methodology:
    - Uses two LLM agents: Qwen 3 and LLaMA 3.1
    - Each agent uses three chain-of-thought prompts
    - Each prompt returns yes/no (1 for yes, 0 for no)
    - Results averaged per agent, then mean across agents
    
    Args:
        pred_texts: List of predicted position descriptions
        gt_texts: List of ground truth position descriptions
        use_llm: If True, use LLM ensemble evaluation. If False, fallback to keyword matching.
    
    Returns:
        Dictionary with evaluation metrics including:
        - ensemble_accuracy: Mean accuracy across both agents
        - agent1_accuracy: Accuracy from first agent (Qwen 3)
        - agent2_accuracy: Accuracy from second agent (LLaMA 3.1)
        - agent1_std: Standard deviation from first agent
        - agent2_std: Standard deviation from second agent
    """
    if not use_llm:
        # Fallback to keyword matching if LLM evaluation is not available
        return evaluate_position_reasoning_simple(pred_texts, gt_texts)
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
    except ImportError:
        print("Warning: transformers not available. Falling back to keyword matching.")
        return evaluate_position_reasoning_simple(pred_texts, gt_texts)
    
    # Chain-of-thought prompts for each agent (3 prompts per agent)
    # These follow the paper's methodology described in Appendix A.3
    prompts_agent1 = [
        """You are evaluating whether a predicted answer about tumor location matches the ground truth answer. 
Think step by step:
1. Analyze the spatial meaning of the predicted answer
2. Analyze the spatial meaning of the ground truth answer
3. Compare if they describe the same or similar anatomical position
4. Consider synonyms and equivalent spatial descriptions

Predicted answer: {predicted}
Ground truth answer: {ground_truth}

Based on your analysis, are these answers describing the same or similar tumor location? Answer only "yes" or "no".""",
        
        """Evaluate if these two answers about tumor position are semantically equivalent:
- Predicted: {predicted}
- Ground truth: {ground_truth}

Consider that position can be described in different ways (e.g., "upper left" vs "top-left corner", "center" vs "middle region").
After reasoning about the spatial meaning, respond with only "yes" if they match, or "no" if they differ.""",
        
        """Compare these medical position descriptions:
Predicted: "{predicted}"
Ground truth: "{ground_truth}"

Step 1: Extract the core spatial location from the predicted answer
Step 2: Extract the core spatial location from the ground truth
Step 3: Determine if they refer to the same anatomical region
Step 4: Respond with "yes" if equivalent, "no" if different."""
    ]
    
    prompts_agent2 = [
        """You are a medical AI evaluator. Determine if the predicted tumor location answer matches the ground truth.
Reasoning process:
1. What anatomical position does the predicted answer indicate?
2. What anatomical position does the ground truth indicate?
3. Do they refer to the same spatial region in the medical image?

Predicted: {predicted}
Ground truth: {ground_truth}

Answer "yes" for match, "no" for mismatch.""",
        
        """Medical position reasoning evaluation:
Analyze whether these two answers describe equivalent tumor locations:
- Prediction: {predicted}
- Reference: {ground_truth}

Think about spatial synonyms and anatomical equivalence. Respond with "yes" or "no".""",
        
        """Evaluate position reasoning accuracy:
Predicted answer: {predicted}
Correct answer: {ground_truth}

Consider the spatial semantics carefully. Medical positions can be expressed differently but mean the same thing.
After your analysis, output only "yes" (if they match) or "no" (if they differ)."""
    ]
    
    # Model names for the two agents
    # Note: These are the models mentioned in the paper (Qwen 3 and LLaMA 3.1)
    # The paper uses Qwen 3 [57] and LLaMA 3.1 [16]
    # You may need to adjust model names based on HuggingFace availability and access permissions
    # For LLaMA models, you may need HuggingFace authentication
    agent_models = [
        "Qwen/Qwen2.5-7B-Instruct",  # Qwen 3 equivalent (paper: Qwen 3)
        "meta-llama/Meta-Llama-3.1-8B-Instruct"  # LLaMA 3.1 equivalent (paper: LLaMA 3.1)
    ]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    agent_results = []
    agent_stds = []
    
    for agent_idx, model_name in enumerate(agent_models):
        try:
            print(f"Loading agent {agent_idx + 1} ({model_name})...")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None
            )
            if device == "cpu":
                model = model.to(device)
            
            # Select prompts for this agent
            prompts = prompts_agent1 if agent_idx == 0 else prompts_agent2
            
            all_scores = []
            
            for pred, gt in zip(pred_texts, gt_texts):
                prompt_scores = []
                
                # Evaluate with each of the 3 prompts
                for prompt_template in prompts:
                    prompt = prompt_template.format(predicted=pred, ground_truth=gt)
                    
                    # Format for the model (adjust based on model's chat template)
                    if "Qwen" in model_name:
                        messages = [{"role": "user", "content": prompt}]
                        formatted_prompt = tokenizer.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True
                        )
                    else:  # LLaMA
                        messages = [{"role": "user", "content": prompt}]
                        formatted_prompt = tokenizer.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True
                        )
                    
                    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
                    
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=10,
                            do_sample=False,
                            temperature=0.1
                        )
                    
                    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                    response = response.strip().lower()
                    
                    # Extract yes/no (1 for yes, 0 for no)
                    if "yes" in response[:10]:  # Check first few words
                        prompt_scores.append(1.0)
                    elif "no" in response[:10]:
                        prompt_scores.append(0.0)
                    else:
                        # If unclear, default to 0
                        prompt_scores.append(0.0)
                
                # Average across the 3 prompts for this sample
                all_scores.append(np.mean(prompt_scores))
            
            # Calculate agent accuracy and std
            agent_acc = np.mean(all_scores)
            agent_std = np.std(all_scores)
            
            agent_results.append(agent_acc)
            agent_stds.append(agent_std)
            
            # Clean up
            del model, tokenizer
            torch.cuda.empty_cache() if device == "cuda" else None
            
        except Exception as e:
            print(f"Warning: Could not load {model_name}: {e}")
            print("This may require HuggingFace authentication or the model may not be available.")
            print("Falling back to keyword matching.")
            return evaluate_position_reasoning_simple(pred_texts, gt_texts)
    
    # Overall result: mean of the two agents
    ensemble_acc = np.mean(agent_results)
    
    return {
        "ensemble_accuracy": float(ensemble_acc),
        "agent1_accuracy": float(agent_results[0]) if len(agent_results) > 0 else 0.0,
        "agent2_accuracy": float(agent_results[1]) if len(agent_results) > 1 else 0.0,
        "agent1_std": float(agent_stds[0]) if len(agent_stds) > 0 else 0.0,
        "agent2_std": float(agent_stds[1]) if len(agent_stds) > 1 else 0.0,
    }


def evaluate_text_reasoning(pred_texts: List[str], gt_texts: List[str], use_llm: bool = True) -> Dict[str, float]:
    """
    Position reasoning evaluation using ensemble LLM method (paper methodology).
    
    This implements the PRS-Med paper's ensemble-based evaluation:
    - Two LLM agents (Qwen 3 and LLaMA 3.1)
    - Three chain-of-thought prompts per agent
    - Yes/no responses (1 for yes, 0 for no)
    - Averaged per agent, then mean across agents
    
    Args:
        pred_texts: List of predicted position descriptions
        gt_texts: List of ground truth position descriptions
        use_llm: If True, use LLM ensemble (paper method). If False, use keyword fallback.
    
    Returns:
        Dictionary with evaluation metrics
    """
    return evaluate_text_reasoning_ensemble_llm(pred_texts, gt_texts, use_llm=use_llm)

