import argparse
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch

from .eval_seg_original import evaluate_segmentation
from .benchmark_prs_med import HFJudge, run_agent_position_benchmark


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
    qwen_judge = HFJudge(qwen_model_name, device_t)

    qwen_acc, qwen_std, qwen_per_prompt = run_agent_position_benchmark(
        qwen_judge,
        questions,
        gt_texts,
        pred_texts,
    )

    # ---------------------- Llama 3.1 Agent ---------------------- #
    print("\nLoading Llama 3.1 judge model from HuggingFace:")
    print(f"  {llama_model_name}")
    llama_judge = HFJudge(llama_model_name, device_t)

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
        help="Hugging Face model name for Qwen 3 judge (as per PRS-Med paper)",
    )
    parser.add_argument(
        "--llama_model_name",
        type=str,
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="Hugging Face model name for Llama 3.1 judge (as per PRS-Med paper)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    results_csv = args.results_csv

    if not os.path.exists(results_csv):
        raise FileNotFoundError(f"results CSV not found: {results_csv}")

    print("=" * 60)
    print("PRS-Med-MMRS Evaluation (Segmentation + Position Reasoning)")
    print("=" * 60)

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
        out_path = Path(args.output_path)
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


