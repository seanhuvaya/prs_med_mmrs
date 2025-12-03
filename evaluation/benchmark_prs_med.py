import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from typing import List, Dict, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM


# ======================= Segmentation Metrics ======================= #

def dice_coefficient(pred_mask: torch.Tensor, true_mask: torch.Tensor, smooth: float = 1e-6):
    """
    Compute Dice coefficient per sample (binary masks).

    PRS-Med uses mDice as a primary segmentation metric.
    We compute per-sample Dice and then average across the dataset.
    """
    # Resize predicted mask to match ground truth resolution if needed
    if pred_mask.shape != true_mask.shape:
        pred_mask = F.interpolate(
            pred_mask,
            size=true_mask.shape[2:],
            mode="bilinear",
            align_corners=False,
        )

    # Convert logits to binary mask
    pred_mask = torch.sigmoid(pred_mask)
    pred_mask = (pred_mask > 0.5).float()
    true_mask = true_mask.float()

    intersection = (pred_mask * true_mask).sum(dim=(1, 2, 3))
    denom = pred_mask.sum(dim=(1, 2, 3)) + true_mask.sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + smooth) / (denom + smooth)
    return dice.cpu().numpy()  # (B,)


def iou_score(pred_mask: torch.Tensor, true_mask: torch.Tensor, smooth: float = 1e-6):
    """
    Compute IoU score per sample (binary masks).

    PRS-Med uses mIoU as the other segmentation metric.
    We compute per-sample IoU and then average across the dataset.
    """
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


# ======================= LLM Judge Utilities ======================= #

class HFJudge:
    """
    Simple HuggingFace-based LLM judge.

    Used to implement the PRS-Med position reasoning benchmark:
    - Two agents: Qwen 3 and Llama 3.1.
    - Each agent evaluated with 3 chain-of-thought prompts that must answer strictly 'yes' or 'no'.
    """

    def __init__(self, model_name: str, device: torch.device, max_new_tokens: int = 8):
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Some chat models require padding side to be left for generation
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        )
        self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def __call__(self, prompt: str) -> str:
        """
        Run the judge on a single prompt and return the raw generated text.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True).to(self.device)
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
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


# ======================= Core Evaluation ======================= #

def evaluate_prs_med(
    model,
    data_loader,
    device: torch.device,
    qwen_model_name: str,
    llama_model_name: str,
    judge_device: torch.device = None,
):
    """
    Evaluation of PRS-Med model, following the paper:

    - Segmentation: mDice, mIoU.
    - Position reasoning: accuracy via ensemble of agents Qwen 3 & Llama 3.1.
      Each agent uses three chain-of-thought prompts and returns yes/no.
    """
    model.to(device)
    model.eval()

    if judge_device is None:
        judge_device = device

    # Collect per-sample segmentation scores and text triplets
    all_dice_scores = []
    all_iou_scores = []
    questions_all: List[str] = []
    gt_texts_all: List[str] = []
    pred_texts_all: List[str] = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating PRS-Med"):
            # Handle different dataloader formats
            if isinstance(batch, dict):
                images = batch["image"].to(device)
                gt_masks = batch["mask"].to(device)
                questions = batch["question"]
                gt_answers = batch["answer"]
            else:
                # Legacy format: (images, questions, gt_masks, _, gt_answers)
                images, questions, gt_masks, _, gt_answers = batch
                images = images.to(device)
                gt_masks = gt_masks.to(device)

            # Forward pass (model expects: images, questions, answers)
            outputs = model(images, questions, gt_answers)

            # Segmentation metrics
            dice_per_sample = dice_coefficient(outputs["z_mask"], gt_masks)
            iou_per_sample = iou_score(outputs["z_mask"], gt_masks)

            all_dice_scores.extend(dice_per_sample)
            all_iou_scores.extend(iou_per_sample)

            # Decode predicted text from logits (teacher-forced output)
            # This uses the same tokenization as in training.
            mllm_model = model.module if hasattr(model, "module") else model
            pred_ids = torch.argmax(outputs["z_txt_logits"], dim=-1)
            pred_text_batch = mllm_model.mllm.processor.batch_decode(
                pred_ids,
                skip_special_tokens=True,
            )

            questions_all.extend(list(questions))
            gt_texts_all.extend(list(gt_answers))
            pred_texts_all.extend(pred_text_batch)

    # Aggregate segmentation metrics
    all_dice_scores = np.array(all_dice_scores, dtype=np.float32)
    all_iou_scores = np.array(all_iou_scores, dtype=np.float32)

    mDice = float(all_dice_scores.mean()) if all_dice_scores.size > 0 else 0.0
    mIoU = float(all_iou_scores.mean()) if all_iou_scores.size > 0 else 0.0

    print("\n" + "=" * 60)
    print("Segmentation Metrics (PRS-Med paper)")
    print(f"  mDice: {mDice:.4f}")
    print(f"  mIoU:  {mIoU:.4f}")
    print("=" * 60 + "\n")

    # ----------------- Position Reasoning Benchmark ----------------- #
    # Qwen 3 agent
    print("Loading Qwen 3 judge model from HuggingFace:")
    print(f"  {qwen_model_name}")
    qwen_judge = HFJudge(qwen_model_name, judge_device)

    qwen_acc, qwen_std, qwen_per_prompt = run_agent_position_benchmark(
        qwen_judge,
        questions_all,
        gt_texts_all,
        pred_texts_all,
    )

    # Llama 3.1 agent
    print("\nLoading Llama 3.1 judge model from HuggingFace:")
    print(f"  {llama_model_name}")
    llama_judge = HFJudge(llama_model_name, judge_device)

    llama_acc, llama_std, llama_per_prompt = run_agent_position_benchmark(
        llama_judge,
        questions_all,
        pred_texts_all=pred_texts_all,
        gt_texts=gt_texts_all,
    )

    # Final result: mean of agents' accuracies
    final_reasoning_acc = (qwen_acc + llama_acc) / 2.0

    print("\n" + "=" * 60)
    print("Position Reasoning Benchmark (PRS-Med)")
    print("Each agent: mean ± std over 3 prompt variants")
    print(f"  Qwen 3:     {qwen_acc:.3f} ({qwen_std:.3f})")
    print(f"  Llama 3.1:  {llama_acc:.3f} ({llama_std:.3f})")
    print(f"  Final Acc:  {final_reasoning_acc:.3f}")
    print("=" * 60 + "\n")

    metrics = {
        "mDice": mDice,
        "mIoU": mIoU,
        "qwen_acc": qwen_acc,
        "qwen_std": qwen_std,
        "llama_acc": llama_acc,
        "llama_std": llama_std,
        "final_reasoning_acc": final_reasoning_acc,
        "qwen_per_prompt_acc": qwen_per_prompt,   # 3 values
        "llama_per_prompt_acc": llama_per_prompt, # 3 values
    }

    return metrics


# ======================= Model Loading & CLI ======================= #

def load_model_from_checkpoint(checkpoint_path: str, args, device: torch.device):
    """
    Load PRS-Med model from checkpoint.

    Assumes PRSMedModel definition matches the training code (same as the paper).
    """
    from train_prs_med import PRSMedModel

    print(f"Loading model from checkpoint: {checkpoint_path}")

    # Initialize model
    model = PRSMedModel(args, device)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        state_dict = checkpoint
        print("Loaded checkpoint (no epoch info found)")

    model.load_state_dict(state_dict, strict=False)
    print("✓ Model loaded successfully")
    return model


def main():
    import argparse
    import json
    from pathlib import Path
    from data.dataset import PRSMedDataset

    parser = argparse.ArgumentParser(description="Evaluate PRS-Med Model (paper-faithful)")

    # Core model / data args
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.pth file)",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Root directory of the dataset",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate on (default: test)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for evaluation (default: 8)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="Number of data loader workers (default: 2)",
    )
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
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save evaluation results (optional)",
    )
    parser.add_argument(
        "--specific_dataset",
        type=str,
        default=None,
        help="Evaluate on specific dataset only (optional)",
    )

    # LLM judges (Qwen 3 and Llama 3.1, from HuggingFace)
    parser.add_argument(
        "--qwen_model_name",
        type=str,
        default="Qwen/Qwen3-8B",
        help="HuggingFace model name for Qwen 3 judge (default: Qwen/Qwen3-8B)",
    )
    parser.add_argument(
        "--llama_model_name",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="HuggingFace model name for Llama 3.1 judge (default: meta-llama/Llama-3.1-8B-Instruct)",
    )
    parser.add_argument(
        "--judge_device",
        type=str,
        default=None,
        help="Device for judge models (e.g., 'cuda', 'cuda:1', 'cpu'). "
             "Defaults to same as main device if not set.",
    )

    args = parser.parse_args()

    # Set device for PRS-Med
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using device: CUDA ({torch.cuda.get_device_name(0)})")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using device: MPS")
    else:
        device = torch.device("cpu")
        print("Using device: CPU")

    # Judge device
    if args.judge_device is not None:
        judge_device = torch.device(args.judge_device)
    else:
        judge_device = device

    # Load PRS-Med model
    model = load_model_from_checkpoint(args.checkpoint, args, device)
    model.eval()

    # Load dataset
    print(f"\nLoading {args.split} dataset from {args.data_root}...")
    dataset = PRSMedDataset(
        split=args.split,
        data_root=args.data_root,
        specific_dataset=args.specific_dataset,
    )
    print(f"Dataset size: {len(dataset)} samples")
    if args.specific_dataset:
        print(f"Evaluating on dataset: {args.specific_dataset}")

    # DataLoader
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device.type == "cuda" else False,
    )

    # Run evaluation
    print("\n" + "=" * 60)
    print(f"Starting evaluation on {args.split} split (PRS-Med protocol)")
    print("=" * 60 + "\n")

    metrics = evaluate_prs_med(
        model,
        data_loader,
        device=device,
        qwen_model_name=args.qwen_model_name,
        llama_model_name=args.llama_model_name,
        judge_device=judge_device,
    )

    # Save results if requested
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_name = Path(args.checkpoint).stem
        results_file = output_dir / f"results_{args.split}_{checkpoint_name}.json"

        results = {
            "checkpoint": args.checkpoint,
            "split": args.split,
            "dataset_size": len(dataset),
            "specific_dataset": args.specific_dataset,
            "metrics": metrics,
            "qwen_model_name": args.qwen_model_name,
            "llama_model_name": args.llama_model_name,
        }

        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n✓ Results saved to: {results_file}")

    # Summary
    print("\n" + "=" * 60)
    print("Evaluation Summary (PRS-Med)")
    print("=" * 60)
    print(f"Checkpoint:   {args.checkpoint}")
    print(f"Split:        {args.split}")
    print(f"Dataset size: {len(dataset)}")
    print("\nSegmentation Metrics:")
    print(f"  mDice: {metrics['mDice']:.4f}")
    print(f"  mIoU:  {metrics['mIoU']:.4f}")
    print("\nPosition Reasoning Benchmark:")
    print(f"  Qwen 3:     {metrics['qwen_acc']:.3f} ({metrics['qwen_std']:.3f})")
    print(f"  Llama 3.1:  {metrics['llama_acc']:.3f} ({metrics['llama_std']:.3f})")
    print(f"  Final Acc:  {metrics['final_reasoning_acc']:.3f}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
