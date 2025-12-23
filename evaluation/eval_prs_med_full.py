import argparse
import os
from pathlib import Path
from typing import List, Tuple
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM


# ======================= Segmentation Metrics ======================= #

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


# ======================= LLM Judge Utilities ======================= #

class HFJudge:
    """
    HuggingFace-based LLM judge for PRS-Med position reasoning benchmark.
    
    Supports both Qwen and Llama models with proper chat template handling.
    Used to implement the PRS-Med position reasoning benchmark:
    - Two agents: Qwen 3 and Llama 3.1.
    - Each agent evaluated with 3 chain-of-thought prompts that must answer strictly 'yes' or 'no'.
    """

    def __init__(self, model_name: str, device: torch.device, max_new_tokens: int = 8, use_auth_token: bool = None):
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                token=use_auth_token,
                trust_remote_code=True,  # Some models require this
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load tokenizer for '{model_name}'. "
                f"Error: {str(e)}\n"
                f"Possible solutions:\n"
                f"  1. Check if the model name is correct\n"
                f"  2. Run 'huggingface-cli login' if the model requires authentication\n"
                f"  3. Ensure transformers>=4.37.0 is installed\n"
                f"  4. Check your internet connection"
            ) from e
        
        # Some chat models require padding side to be left for generation
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Detect model type for chat template handling
        self.is_qwen = "qwen" in model_name.lower()
        self.is_llama = "llama" in model_name.lower()

        # Load model with appropriate device handling
        self.model = None  # Initialize to None to ensure it's defined
        try:
            if device.type == "cuda":
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",  # Automatically handles multi-GPU if available
                    token=use_auth_token,
                    trust_remote_code=True,  # Some models require this
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,
                    token=use_auth_token,
                    trust_remote_code=True,
                )
                self.model.to(device)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load model '{model_name}'. "
                f"Error: {str(e)}\n"
                f"Possible solutions:\n"
                f"  1. Check if the model name is correct\n"
                f"  2. Run 'huggingface-cli login' if the model requires authentication\n"
                f"  3. Ensure transformers>=4.37.0 is installed\n"
                f"  4. Check your internet connection\n"
                f"  5. Ensure you have sufficient disk space and memory"
            ) from e
        
        # Verify model was loaded successfully
        if self.model is None:
            raise RuntimeError(f"Model '{model_name}' failed to load: self.model is None after loading attempt")
        
        # Set pad_token_id in model config to avoid warnings during generation
        # (Do this AFTER model is loaded)
        try:
            if hasattr(self.model, 'config') and self.model.config is not None:
                if self.model.config.pad_token_id is None:
                    self.model.config.pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
                # Also set in generation config if it exists
                if hasattr(self.model, 'generation_config') and self.model.generation_config is not None:
                    if self.model.generation_config.pad_token_id is None:
                        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        except AttributeError as e:
            # If model doesn't have expected attributes, log but continue
            print(f"Warning: Could not set pad_token_id in model config: {e}")
        
        self.model.eval()

    @torch.no_grad()
    def __call__(self, prompt: str) -> str:
        """
        Run the judge on a single prompt and return the raw generated text.
        Handles chat templates for Qwen and Llama models.
        """
        # Use chat template if available (for Qwen/Llama chat models)
        if hasattr(self.tokenizer, "apply_chat_template") and (self.is_qwen or self.is_llama):
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
        else:
            # Fallback to direct tokenization
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        
        # Move inputs to device
        # Note: When using device_map="auto", the model handles device placement,
        # but inputs still need to be on a CUDA device for proper handling
        if self.device.type == "cuda":
            # For CUDA, move to device (model will handle placement if using device_map="auto")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        else:
            # For CPU, ensure inputs are on CPU
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Suppress pad_token_id warnings by explicitly setting it in generate call
        generation_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": False,
        }
        
        # Set pad_token_id if not already set in model config
        if hasattr(self.model, 'config') and self.model.config.pad_token_id is not None:
            generation_kwargs["pad_token_id"] = self.model.config.pad_token_id
        elif self.tokenizer.pad_token_id is not None:
            generation_kwargs["pad_token_id"] = self.tokenizer.pad_token_id
        else:
            generation_kwargs["pad_token_id"] = self.tokenizer.eos_token_id
        
        # Suppress warnings for this specific call
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*pad_token_id.*")
            output_ids = self.model.generate(
                **inputs,
                **generation_kwargs,
            )
        # Take only the newly generated tokens
        gen_ids = output_ids[0, inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        return text.strip()


def parse_yes_no(output_text: str) -> int:
    """
    Parse model output to a binary label:

    - "yes" -> 1
    - "no"  -> 0

    PRS-Med requires the judge to answer in yes/no format.
    """
    text = output_text.strip().lower()

    # Look at very beginning first
    if text.startswith("yes"):
        return 1
    if text.startswith("no"):
        return 0

    # Fallback: search anywhere
    if "yes" in text and "no" not in text:
        return 1
    if "no" in text and "yes" not in text:
        return 0

    # Conservative default if unclear
    return 0


def get_position_benchmark_prompts() -> List[str]:
    """
    Three chain-of-thought style prompt templates for the benchmark,
    following Appendix A.3 of PRS-Med as closely as possible.

    Placeholders:
        {question}, {groundtruth}, {prediction}
    """
    # 1) "As a medical image specialist ..."
    prompt1 = (
        "As a medical image specialist.\n"
        "Instruction: Answer the question related to the position content, return only yes or no.\n"
        "Given the following question and answer with the ground truth, is the position in the answer "
        "similar or same with the ground truth and match with the question?\n\n"
        "Question: {question}\n"
        "Ground Truth: {groundtruth}\n"
        "Prediction: {prediction}\n\n"
        "Return yes if they are similar. Return no if they are different."
    )

    # 2) "As a doctor ..."
    prompt2 = (
        "As a doctor.\n"
        "Instruction: Answer the question related to the position content, return only yes or no.\n"
        "Check if the location information provided in the prediction aligns with the position mentioned "
        "in the ground truth and is relevant to the question.\n\n"
        "Question: {question}\n"
        "Ground Truth: {groundtruth}\n"
        "Prediction: {prediction}\n\n"
        "Respond with Yes if the positions are similar. Respond with No if they are different."
    )

    # 3) "As you are a doctor and you are looking to the medical image ..."
    prompt3 = (
        "As you are a doctor and you are looking to the medical image.\n"
        "Instruction: Answer the question related to the position content, return only yes or no.\n"
        "Evaluate whether the predicted answer captures the same or similar positional context as the ground truth, "
        "based on the provided question.\n\n"
        "Question: {question}\n"
        "Groundtruth: {groundtruth}\n"
        "Prediction: {prediction}\n\n"
        "Answer with \"Yes\" if the position is similar, otherwise \"No\"."
    )

    return [prompt1, prompt2, prompt3]


def run_agent_position_benchmark(
    judge: HFJudge,
    questions: List[str],
    gt_texts: List[str],
    pred_texts: List[str],
) -> Tuple[float, float, List[float]]:
    """
    Run the PRS-Med position reasoning benchmark for a single agent
    (Qwen 3 or Llama 3.1).

    For each agent:
      - Use 3 different prompts (Appendix A.3).
      - For each prompt, evaluate accuracy across all (Q, GT, Pred) triplets.
      - Agent accuracy = mean of the 3 prompt accuracies.
      - Agent std     = std dev over the 3 prompt accuracies.

    Returns:
        agent_acc (float), agent_std (float), per_prompt_acc (list of 3 floats)
    """
    templates = get_position_benchmark_prompts()
    assert len(templates) == 3

    per_prompt_acc = []

    for template in templates:
        correct = 0
        total = len(pred_texts)

        for q, gt, pred in zip(questions, gt_texts, pred_texts):
            prompt = template.format(
                question=q,
                groundtruth=gt,
                prediction=pred,
            )
            output = judge(prompt)
            label = parse_yes_no(output)
            correct += label

        acc = correct / total if total > 0 else 0.0
        per_prompt_acc.append(acc)

    per_prompt_acc = np.array(per_prompt_acc, dtype=np.float32)
    agent_acc = float(per_prompt_acc.mean())
    agent_std = float(per_prompt_acc.std(ddof=0))

    return agent_acc, agent_std, per_prompt_acc.tolist()


# ======================= Main Evaluation Functions ======================= #

def load_text_triplets(results_csv: str) -> Tuple[List[str], List[str], List[str]]:
    """
    Load (question, ground-truth answer, predicted answer) triplets from
    the results CSV produced by infer_original.py.
    """
    df = pd.read_csv(results_csv)

    required_cols = ["question", "answer", "pred_answer"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(
                f"Column '{col}' not found in {results_csv}. "
                "Make sure you used infer_original.py to generate this CSV."
            )

    questions = df["question"].astype(str).tolist()
    gt_texts = df["answer"].astype(str).tolist()
    pred_texts = df["pred_answer"].astype(str).tolist()

    return questions, gt_texts, pred_texts


def evaluate_reasoning(
    results_csv: str,
    qwen_model_name: str,
    llama_model_name: str,
    device: str = "cuda",
    use_auth_token: bool = None,
) -> Tuple[float, float, float, float, List[float], List[float]]:
    """
    Evaluate position reasoning EXACTLY as in the PRS-Med paper:
    - Uses ensemble of TWO judges: Qwen 3 and Llama 3.1
    - Each judge uses three chain-of-thought prompts
    - Final accuracy = mean of the two agents' accuracies
    
    Returns:
        final_acc: Mean of Qwen and Llama accuracies (paper's final metric)
        qwen_acc: Qwen agent accuracy
        qwen_std: Qwen agent std dev over prompts
        llama_acc: Llama agent accuracy
        llama_std: Llama agent std dev over prompts
        qwen_per_prompt: Qwen's per-prompt accuracies (3 values)
        llama_per_prompt: Llama's per-prompt accuracies (3 values)
    """
    device_t = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")

    questions, gt_texts, pred_texts = load_text_triplets(results_csv)

    print(f"\nLoaded {len(questions)} QA triplets from {results_csv} for reasoning benchmark")

    # ---------------------- Qwen 3 Agent ---------------------- #
    print("\nLoading Qwen 3 judge model from HuggingFace:")
    print(f"  {qwen_model_name}")
    qwen_judge = HFJudge(qwen_model_name, device_t, use_auth_token=use_auth_token)

    qwen_acc, qwen_std, qwen_per_prompt = run_agent_position_benchmark(
        qwen_judge,
        questions,
        gt_texts,
        pred_texts,
    )

    # ---------------------- Llama 3.1 Agent ---------------------- #
    print("\nLoading Llama 3.1 judge model from HuggingFace:")
    print(f"  {llama_model_name}")
    llama_judge = HFJudge(llama_model_name, device_t, use_auth_token=use_auth_token)

    llama_acc, llama_std, llama_per_prompt = run_agent_position_benchmark(
        llama_judge,
        questions,
        gt_texts,
        pred_texts,
    )

    # ---------------------- Ensemble (Paper's Final Metric) ---------------------- #
    # Final result: mean of agents' accuracies (as per PRS-Med paper)
    final_acc = (qwen_acc + llama_acc) / 2.0

    return final_acc, qwen_acc, qwen_std, llama_acc, llama_std, qwen_per_prompt, llama_per_prompt


# ======================= Main Entry Point ======================= #

def parse_args():
    parser = argparse.ArgumentParser(
        description="Full PRS-Med-style evaluation for PRS-Med-MMRS (segmentation + position reasoning)"
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
        help="Optional path to save a CSV with summary metrics",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for the judge model (default: cuda if available)",
    )
    parser.add_argument(
        "--qwen_model_name",
        type=str,
        default="Qwen/Qwen2-1.5B-Instruct",
        help="Hugging Face model name for Qwen judge (as per PRS-Med paper). "
             "Examples: 'Qwen/Qwen2-1.5B-Instruct', 'Qwen/Qwen2-7B-Instruct'. "
             "Note: Some models may require authentication. Run 'huggingface-cli login' first.",
    )
    parser.add_argument(
        "--llama_model_name",
        type=str,
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="Hugging Face model name for Llama judge (as per PRS-Med paper). "
             "Examples: 'meta-llama/Meta-Llama-3.1-8B-Instruct', 'meta-llama/Llama-3-8B-Instruct'. "
             "Note: Llama models require authentication. Run 'huggingface-cli login' and accept the model's terms first.",
    )
    parser.add_argument(
        "--use_auth_token",
        action="store_true",
        help="Use Hugging Face authentication token (required for gated models). "
             "Alternatively, run 'huggingface-cli login' before running this script.",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="Hugging Face token as string (alternative to --use_auth_token). "
             "Can also be set via HF_TOKEN environment variable.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    results_csv = args.results_csv

    if not os.path.exists(results_csv):
        raise FileNotFoundError(f"results CSV not found: {results_csv}")
    
    # Handle authentication token
    use_auth_token = None
    if args.hf_token:
        use_auth_token = args.hf_token
    elif args.use_auth_token:
        # Try to get token from environment or huggingface-cli
        use_auth_token = os.getenv("HF_TOKEN") or True
    else:
        # Try environment variable as fallback
        use_auth_token = os.getenv("HF_TOKEN")

    print("=" * 60)
    print("PRS-Med-MMRS Evaluation (Segmentation + Position Reasoning)")
    print("=" * 60)
    
    # Check if authentication might be needed
    if use_auth_token is None:
        print("\nNote: If you encounter authentication errors, you may need to:")
        print("  1. Run 'huggingface-cli login' to authenticate")
        print("  2. Accept the model's terms on Hugging Face Hub")
        print("  3. Use --use_auth_token or --hf_token argument")
        print("  4. Or set HF_TOKEN environment variable\n")

    # ---------------------- Segmentation Metrics ---------------------- #
    print("\n[1/2] Evaluating segmentation (mDice, mIoU)...")
    mean_dice, mean_iou = evaluate_segmentation(results_csv)

    print("\nSegmentation Metrics (PRS-Med style):")
    print(f"  mDice: {mean_dice:.4f}")
    print(f"  mIoU:  {mean_iou:.4f}")

    # ---------------------- Reasoning Metrics ------------------------- #
    print("\n[2/2] Evaluating position reasoning with LLM judge ensemble (PRS-Med paper exact)...")
    final_acc, qwen_acc, qwen_std, llama_acc, llama_std, qwen_per_prompt, llama_per_prompt = evaluate_reasoning(
        results_csv,
        qwen_model_name=args.qwen_model_name,
        llama_model_name=args.llama_model_name,
        device=args.device,
        use_auth_token=use_auth_token,
    )

    print("\n" + "=" * 60)
    print("Position Reasoning Benchmark (PRS-Med Paper - Exact Match)")
    print("Each agent: mean ± std over 3 prompt variants")
    print(f"  Qwen 3:     {qwen_acc:.4f} ± {qwen_std:.4f}")
    print(f"  Llama 3.1:  {llama_acc:.4f} ± {llama_std:.4f}")
    print(f"  Final Acc (Ensemble): {final_acc:.4f}")
    print("=" * 60)
    
    print("\nPer-Prompt Accuracies:")
    print("  Qwen 3:")
    for i, acc in enumerate(qwen_per_prompt):
        print(f"    Prompt {i+1}: {acc:.4f}")
    print("  Llama 3.1:")
    for i, acc in enumerate(llama_per_prompt):
        print(f"    Prompt {i+1}: {acc:.4f}")

    # ---------------------- Save Summary ------------------------------ #
    if args.output_path is not None:
        # Add timestamp prefix if output_path is a directory or doesn't have timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = Path(args.output_path)
        
        # If it's a directory, create a timestamped file inside it
        if out_path.is_dir() or (not out_path.suffix and not out_path.exists()):
            out_path = out_path / f"eval_results_{timestamp}.csv"
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
                    "final_reasoning_acc": final_acc,  # Ensemble accuracy (paper's metric)
                    "qwen_acc": qwen_acc,
                    "qwen_std": qwen_std,
                    "llama_acc": llama_acc,
                    "llama_std": llama_std,
                    "qwen_prompt1_acc": qwen_per_prompt[0] if len(qwen_per_prompt) > 0 else np.nan,
                    "qwen_prompt2_acc": qwen_per_prompt[1] if len(qwen_per_prompt) > 1 else np.nan,
                    "qwen_prompt3_acc": qwen_per_prompt[2] if len(qwen_per_prompt) > 2 else np.nan,
                    "llama_prompt1_acc": llama_per_prompt[0] if len(llama_per_prompt) > 0 else np.nan,
                    "llama_prompt2_acc": llama_per_prompt[1] if len(llama_per_prompt) > 1 else np.nan,
                    "llama_prompt3_acc": llama_per_prompt[2] if len(llama_per_prompt) > 2 else np.nan,
                    "qwen_model": args.qwen_model_name,
                    "llama_model": args.llama_model_name,
                }
            ]
        )
        df.to_csv(out_path, index=False)
        print(f"\nSaved summary metrics to: {out_path}")


if __name__ == "__main__":
    main()
