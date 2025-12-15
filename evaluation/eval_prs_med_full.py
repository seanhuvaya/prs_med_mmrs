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
    device: str = "cuda",
) -> Tuple[float, float, List[float]]:
    """
    Evaluate position reasoning as in the PRS-Med paper using a Hugging Face LLM judge.

    - Uses Qwen (or any HF model) as the judge.
    - Follows the three chain-of-thought prompts in benchmark_prs_med.py.
    """
    device_t = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")

    questions, gt_texts, pred_texts = load_text_triplets(results_csv)

    print(f"\nLoaded {len(questions)} QA triplets from {results_csv} for reasoning benchmark")

    print(f"\nLoading judge model from HuggingFace: {qwen_model_name}")
    judge = HFJudge(qwen_model_name, device_t)

    agent_acc, agent_std, per_prompt_acc = run_agent_position_benchmark(
        judge,
        questions,
        gt_texts,
        pred_texts,
    )

    return agent_acc, agent_std, per_prompt_acc


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
        help="Hugging Face model name for the Qwen-style judge (or any compatible chat model)",
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
    print("\n[2/2] Evaluating position reasoning with LLM judge...")
    agent_acc, agent_std, per_prompt_acc = evaluate_reasoning(
        results_csv,
        qwen_model_name=args.qwen_model_name,
        device=args.device,
    )

    print("\nPosition Reasoning Metrics (PRS-Med style, single judge):")
    print(f"  Accuracy (mean over 3 prompts): {agent_acc:.4f}")
    print(f"  Std over prompts:              {agent_std:.4f}")
    for i, acc in enumerate(per_prompt_acc):
        print(f"  Prompt {i+1} accuracy:          {acc:.4f}")

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
                    "reasoning_acc": agent_acc,
                    "reasoning_std": agent_std,
                    "prompt1_acc": per_prompt_acc[0] if len(per_prompt_acc) > 0 else np.nan,
                    "prompt2_acc": per_prompt_acc[1] if len(per_prompt_acc) > 1 else np.nan,
                    "prompt3_acc": per_prompt_acc[2] if len(per_prompt_acc) > 2 else np.nan,
                    "judge_model": args.qwen_model_name,
                }
            ]
        )
        df.to_csv(out_path, index=False)
        print(f"\nSaved summary metrics to: {out_path}")


if __name__ == "__main__":
    main()


