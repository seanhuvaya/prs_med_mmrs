import numpy as np
import torch
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from tqdm import tqdm

from evaluation.metrics import (
    dice_coefficient,
    iou_score,
    hausdorff_distance_per_sample,
    evaluate_text_reasoning
)

load_dotenv()



def evaluate_prs_med(model, data_loader, device, verbose=True, use_llm_eval=True):
    """
    Evaluation of PRS-Med model following standard medical segmentation practices.
    Metrics are computed per-sample and aggregated across the entire dataset.
    
    Args:
        model: PRS-Med model
        data_loader: DataLoader with test data
        device: Device to run evaluation on
        verbose: If True, print detailed metrics. If False, only return metrics dict.
        use_llm_eval: If True, use ensemble LLM evaluation (paper method). If False, use keyword matching.
    
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
        for batch in tqdm(data_loader, desc="Evaluating", disable=not verbose):
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
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Segmentation Metrics (computed per-sample, aggregated across dataset):")
        print(f"  mDice:  {mdice:.4f}")
        print(f"  mIoU:   {miou:.4f}")
        print(f"  mHD95:  {mhd95:.2f}" if mhd95 != float('inf') else f"  mHD95:  inf")
        print(f"{'='*60}")

        # Position reasoning metrics (paper method: ensemble LLM evaluation)
        text_metrics = evaluate_text_reasoning(pred_texts, gt_texts, use_llm=use_llm_eval)
        
        print(f"\nPosition Reasoning Metrics (Ensemble LLM Evaluation):")
        if 'ensemble_accuracy' in text_metrics:
            print(f"  Ensemble Accuracy:     {text_metrics.get('ensemble_accuracy', 0):.4f}")
            print(f"  Agent 1 (Qwen) Acc:     {text_metrics.get('agent1_accuracy', 0):.4f} ± {text_metrics.get('agent1_std', 0):.4f}")
            print(f"  Agent 2 (LLaMA) Acc:    {text_metrics.get('agent2_accuracy', 0):.4f} ± {text_metrics.get('agent2_std', 0):.4f}")
        else:
            # Fallback to keyword matching if LLM not available
            print(f"  Exact Match Accuracy:   {text_metrics.get('exact_match_acc', 0):.4f}")
            print(f"  Keyword Match Accuracy: {text_metrics.get('keyword_match_acc', 0):.4f}")
        print(f"{'='*60}\n")
    else:
        text_metrics = evaluate_text_reasoning(pred_texts, gt_texts, use_llm=use_llm_eval)

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
    from models import PRSMedModel
    from utils import load_checkpoint
    
    print(f"Loading model from checkpoint: {checkpoint_path}")
    
    # Initialize model
    model = PRSMedModel(args, device)
    
    # Load checkpoint using utility function
    checkpoint_info = load_checkpoint(checkpoint_path, model, device=device)
    
    print(f"✓ Model loaded successfully (epoch {checkpoint_info.get('epoch', 'unknown')})")
    return model


def main():
    import argparse
    import json
    from pathlib import Path
    from data.dataset import PRSMedDataset, PRSMedDataLoader
    
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
    parser.add_argument('--use_llm_eval', action='store_true', default=True,
                       help='Use ensemble LLM evaluation for position reasoning (paper method, default: True)')
    parser.add_argument('--no_llm_eval', dest='use_llm_eval', action='store_false',
                       help='Disable LLM evaluation and use keyword matching fallback')
    
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
    
    # Helper to format and print a results table
    def _print_results_table(rows, title="Evaluation Results by Dataset"):
        # rows: list of dicts with keys: dataset, n, mDice, mIoU, mHD95, ensemble_accuracy (or fallback metrics)
        # Check if we have ensemble metrics (paper method) or fallback metrics
        has_ensemble = any('ensemble_accuracy' in r for r in rows)
        
        if has_ensemble:
            headers = [
                "Dataset", "N", "mDice", "mIoU", "mHD95", "EnsembleAcc", "Agent1Acc", "Agent2Acc"
            ]
            col_widths = {
                "Dataset": max(len("Dataset"), max(len(str(r["dataset"])) for r in rows) if rows else 7),
                "N": max(len("N"), max(len(str(r["n"])) for r in rows) if rows else 1),
                "mDice": 7, "mIoU": 7, "mHD95": 7, "EnsembleAcc": 12, "Agent1Acc": 11, "Agent2Acc": 11
            }
            def fmt_row(r):
                hd95_str = "   inf" if r['mHD95'] == float('inf') else f"{r['mHD95']:>7.2f}"
                return (
                    f"{r['dataset']:<{col_widths['Dataset']}}  "
                    f"{r['n']:>{col_widths['N']}}  "
                    f"{r['mDice']:>7.4f}  "
                    f"{r['mIoU']:>7.4f}  "
                    f"{hd95_str}  "
                    f"{r.get('ensemble_accuracy', 0.0):>12.4f}  "
                    f"{r.get('agent1_accuracy', 0.0):>11.4f}  "
                    f"{r.get('agent2_accuracy', 0.0):>11.4f}"
                )
            header_line = (
                f"{headers[0]:<{col_widths['Dataset']}}  "
                f"{headers[1]:>{col_widths['N']}}  "
                f"{headers[2]:>7}  {headers[3]:>7}  {headers[4]:>7}  {headers[5]:>12}  {headers[6]:>11}  {headers[7]:>11}"
            )
        else:
            # Fallback to keyword matching metrics
            headers = [
                "Dataset", "N", "mDice", "mIoU", "mHD95", "ExactAcc", "KeywordAcc"
            ]
            col_widths = {
                "Dataset": max(len("Dataset"), max(len(str(r["dataset"])) for r in rows) if rows else 7),
                "N": max(len("N"), max(len(str(r["n"])) for r in rows) if rows else 1),
                "mDice": 7, "mIoU": 7, "mHD95": 7, "ExactAcc": 9, "KeywordAcc": 11
            }
            def fmt_row(r):
                hd95_str = "   inf" if r['mHD95'] == float('inf') else f"{r['mHD95']:>7.2f}"
                return (
                    f"{r['dataset']:<{col_widths['Dataset']}}  "
                    f"{r['n']:>{col_widths['N']}}  "
                    f"{r['mDice']:>7.4f}  "
                    f"{r['mIoU']:>7.4f}  "
                    f"{hd95_str}  "
                    f"{r.get('exact_match_acc', 0.0):>9.4f}  "
                    f"{r.get('keyword_match_acc', 0.0):>11.4f}"
                )
            header_line = (
                f"{headers[0]:<{col_widths['Dataset']}}  "
                f"{headers[1]:>{col_widths['N']}}  "
                f"{headers[2]:>7}  {headers[3]:>7}  {headers[4]:>7}  {headers[5]:>9}  {headers[6]:>11}"
            )
        
        print(f"\n{'='*80}")
        print(f"{title:^80}")
        print(f"{'='*80}")
        print(header_line)
        print("-" * len(header_line))
        for r in rows:
            print(fmt_row(r))
        print("-" * len(header_line))
        print(f"{'='*80}\n")

    # Two modes:
    # 1) If a specific dataset is provided, evaluate it only (legacy behavior + table row)
    # 2) Otherwise, build per-dataset test loaders and report a table across datasets

    if args.specific_dataset is not None:
        # Load dataset (single)
        print(f"\nLoading {args.split} dataset from {args.data_root}...")
        test_dataset = PRSMedDataset(
            split=args.split,
            data_root=args.data_root,
            specific_dataset=args.specific_dataset
        )
        print(f"Dataset size: {len(test_dataset)} samples")
        print(f"Evaluating on dataset: {args.specific_dataset}")

        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True if device.type == 'cuda' else False
        )

        print(f"\n{'='*60}")
        print(f"Starting evaluation on {args.split} split")
        print(f"{'='*60}\n")

        metrics = evaluate_prs_med(model, test_loader, device, verbose=True, use_llm_eval=args.use_llm_eval)

        # Prepare single-row table output (paper metrics)
        rows = [{
            "dataset": args.specific_dataset,
            "n": len(test_dataset),
            "mDice": metrics.get("mDice", 0.0),
            "mIoU": metrics.get("mIoU", 0.0),
            "mHD95": metrics.get("mHD95", float('inf')),
            "ensemble_accuracy": metrics.get("ensemble_accuracy", metrics.get("exact_match_acc", 0.0)),
            "agent1_accuracy": metrics.get("agent1_accuracy", 0.0),
            "agent2_accuracy": metrics.get("agent2_accuracy", 0.0),
            # Fallback metrics if LLM not available
            "exact_match_acc": metrics.get("exact_match_acc", 0.0),
            "keyword_match_acc": metrics.get("keyword_match_acc", 0.0),
        }]
        _print_results_table(rows, title=f"Evaluation Results: {args.specific_dataset}")

        # Save results if output directory specified
        if args.output_dir:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_name = Path(args.checkpoint).stem
            results_file = output_dir / f"results_{args.split}_{checkpoint_name}_{args.specific_dataset}.json"
            # Add metadata
            results = {
                'checkpoint': args.checkpoint,
                'split': args.split,
                'dataset_size': len(test_dataset),
                'specific_dataset': args.specific_dataset,
                'metrics': metrics
            }
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            # Also save a small CSV table (paper metrics)
            csv_path = output_dir / f"table_{args.split}_{checkpoint_name}_{args.specific_dataset}.csv"
            with open(csv_path, 'w') as f:
                if 'ensemble_accuracy' in rows[0]:
                    f.write("Dataset,N,mDice,mIoU,mHD95,EnsembleAcc,Agent1Acc,Agent1Std,Agent2Acc,Agent2Std\n")
                    r = rows[0]
                    f.write(f"{r['dataset']},{r['n']},{r['mDice']:.6f},{r['mIoU']:.6f},{r['mHD95']},{r.get('ensemble_accuracy', 0.0):.6f},{r.get('agent1_accuracy', 0.0):.6f},{metrics.get('agent1_std', 0.0):.6f},{r.get('agent2_accuracy', 0.0):.6f},{metrics.get('agent2_std', 0.0):.6f}\n")
                else:
                    f.write("Dataset,N,mDice,mIoU,mHD95,ExactAcc,KeywordAcc\n")
                    r = rows[0]
                    f.write(f"{r['dataset']},{r['n']},{r['mDice']:.6f},{r['mIoU']:.6f},{r['mHD95']},{r['exact_match_acc']:.6f},{r['keyword_match_acc']:.6f}\n")
            print(f"\n✓ Results saved to: {results_file}\n✓ Table saved to: {csv_path}")

        # Print summary remains for convenience
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
        if 'ensemble_accuracy' in metrics:
            print(f"  Ensemble Accuracy:     {metrics.get('ensemble_accuracy', 0):.4f}")
            print(f"  Agent 1 (Qwen) Acc:     {metrics.get('agent1_accuracy', 0):.4f} ± {metrics.get('agent1_std', 0):.4f}")
            print(f"  Agent 2 (LLaMA) Acc:    {metrics.get('agent2_accuracy', 0):.4f} ± {metrics.get('agent2_std', 0):.4f}")
        else:
            print(f"  Exact Match Accuracy:   {metrics.get('exact_match_acc', 0):.4f}")
            print(f"  Keyword Match Accuracy: {metrics.get('keyword_match_acc', 0):.4f}")
        print(f"{'='*60}\n")

    else:
        # Build per-dataset loaders for the requested split. For now, leverage testing helper.
        # If evaluating train/val, we still partition by datasets present in that split.
        # We emulate get_testing_dataloaders behavior across the chosen split by discovering
        # available datasets in the split and constructing loaders one by one.
        print(f"\nDiscovering datasets for split='{args.split}' under {args.data_root}...")
        # Discover available datasets by reading the split dataframe
        split_df = PRSMedDataset(split=args.split, data_root=args.data_root)
        available = split_df.get_available_datasets()
        print(f"Found datasets: {', '.join(available) if available else '(none)'}")

        rows = []
        per_dataset_metrics = {}

        print(f"\n{'='*80}")
        print(f"Evaluating {len(available)} dataset(s) on {args.split} split")
        print(f"{'='*80}\n")

        for ds_name in available:
            print(f"Evaluating dataset: {ds_name}...", end=" ", flush=True)
            ds = PRSMedDataset(
                split=args.split,
                data_root=args.data_root,
                specific_dataset=ds_name,
            )
            loader = DataLoader(
                ds,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True if device.type == 'cuda' else False,
            )
            # Suppress verbose output when evaluating multiple datasets
            m = evaluate_prs_med(model, loader, device, verbose=False, use_llm_eval=args.use_llm_eval)
            per_dataset_metrics[ds_name] = m
            rows.append({
                "dataset": ds_name,
                "n": len(ds),
                "mDice": m.get("mDice", 0.0),
                "mIoU": m.get("mIoU", 0.0),
                "mHD95": m.get("mHD95", float('inf')),
                "ensemble_accuracy": m.get("ensemble_accuracy", m.get("exact_match_acc", 0.0)),
                "agent1_accuracy": m.get("agent1_accuracy", 0.0),
                "agent2_accuracy": m.get("agent2_accuracy", 0.0),
                # Fallback metrics if LLM not available
                "exact_match_acc": m.get("exact_match_acc", 0.0),
                "keyword_match_acc": m.get("keyword_match_acc", 0.0),
            })
            print(f"✓ ({len(ds)} samples)")

        # Print table across datasets
        _print_results_table(rows, title=f"Evaluation Results: {args.split} split")

        # Save aggregated results if requested
        if args.output_dir:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_name = Path(args.checkpoint).stem
            json_path = output_dir / f"results_by_dataset_{args.split}_{checkpoint_name}.json"
            with open(json_path, 'w') as f:
                json.dump({
                    "checkpoint": args.checkpoint,
                    "split": args.split,
                    "results": per_dataset_metrics,
                }, f, indent=2)
            csv_path = output_dir / f"table_by_dataset_{args.split}_{checkpoint_name}.csv"
            with open(csv_path, 'w') as f:
                if rows and 'ensemble_accuracy' in rows[0]:
                    f.write("Dataset,N,mDice,mIoU,mHD95,EnsembleAcc,Agent1Acc,Agent1Std,Agent2Acc,Agent2Std\n")
                    for r in rows:
                        ds_metrics = per_dataset_metrics.get(r['dataset'], {})
                        f.write(f"{r['dataset']},{r['n']},{r['mDice']:.6f},{r['mIoU']:.6f},{r['mHD95']},{r.get('ensemble_accuracy', 0.0):.6f},{r.get('agent1_accuracy', 0.0):.6f},{ds_metrics.get('agent1_std', 0.0):.6f},{r.get('agent2_accuracy', 0.0):.6f},{ds_metrics.get('agent2_std', 0.0):.6f}\n")
                else:
                    f.write("Dataset,N,mDice,mIoU,mHD95,ExactAcc,KeywordAcc\n")
                    for r in rows:
                        f.write(f"{r['dataset']},{r['n']},{r['mDice']:.6f},{r['mIoU']:.6f},{r['mHD95']},{r['exact_match_acc']:.6f},{r['keyword_match_acc']:.6f}\n")
            print(f"\n✓ Aggregated results saved to: {json_path}\n✓ Table saved to: {csv_path}")


if __name__ == "__main__":
    main()