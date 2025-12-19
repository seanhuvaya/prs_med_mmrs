import os
import sys
import argparse
import torch
from torch.cuda.amp import autocast
from PIL import Image
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add parent PRS-Med repo to path if needed
parent_repo = os.path.join(os.path.dirname(__file__), '../PRS-Med')
if os.path.exists(parent_repo):
    sys.path.insert(0, parent_repo)

from models.llm_seg_original import build_llm_seg
from data_utils.utils import load_image, binary_loader
from llava.mm_utils import process_images, tokenizer_image_token
from torchvision import transforms


def parse_args():
    parser = argparse.ArgumentParser(description='PRS-Med Inference (Original Model)')
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to checkpoint directory (e.g., checkpoints/llm_seg_10)'
    )
    parser.add_argument(
        '--vlm_path',
        type=str,
        required=True,
        help='Path to LLaVA-Med model (local path or Hugging Face ID like "microsoft/llava-med-v1.5-mistral-7b")'
    )
    parser.add_argument(
        '--sam_ckpt',
        type=str,
        required=True,
        help='Path to vision encoder checkpoint (TinySAM or SAM-Med2D)'
    )
    parser.add_argument(
        '--sam_model_type',
        type=str,
        default='vit_t',
        help='TinySAM model type (default: vit_t, ignored for SAM-Med2D)'
    )
    parser.add_argument(
        '--encoder_type',
        type=str,
        default='tinysam',
        choices=['tinysam', 'sam_med2d', 'sammed2d'],
        help='Vision encoder type: tinysam or sam_med2d (default: tinysam)'
    )
    parser.add_argument(
        '--image_size',
        type=int,
        default=1024,
        help='Input image size (default: 1024)'
    )
    parser.add_argument(
        '--data_root',
        type=str,
        required=True,
        help='Root directory containing data (e.g., data_v2/)'
    )
    parser.add_argument(
        '--ann_paths',
        type=str,
        required=True,
        help='Comma-separated paths to annotation CSV files'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        choices=['train', 'val', 'test'],
        help='Dataset split to use (default: test)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='inference_outputs_original',
        help='Directory to save outputs (default: inference_outputs_original)'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=None,
        help='Number of samples to process (None = all samples)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
        help='Device to use (default: cuda:0)'
    )
    parser.add_argument(
        '--load_8bit',
        action='store_true',
        help='Load model in 8-bit'
    )
    parser.add_argument(
        '--load_4bit',
        action='store_true',
        help='Load model in 4-bit'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.2,
        help='Temperature for generation (default: 0.2)'
    )
    parser.add_argument(
        '--max_new_tokens',
        type=int,
        default=512,
        help='Max new tokens for generation (default: 512)'
    )
    parser.add_argument(
        '--top_p',
        type=float,
        default=0.95,
        help='Top-p for generation (default: 0.95)'
    )
    return parser.parse_args()


def transform_for_sam(image_path):
    """Transform image for SAM encoder"""
    image_sam_transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = load_image(image_path)
    image_tensor = image_sam_transform(image)
    return image_tensor.to(torch.float32).unsqueeze(0)


def load_image_for_vlm(image_path, image_processor, config):
    """Load and process image for VLM"""
    image_pil = load_image(image_path)
    image_tensor = process_images(
        [image_pil],
        image_processor,
        config
    )
    # Keep batch dimension (1, C, H, W) for CLIP vision tower
    return image_tensor.to(torch.float16)


def save_visualization_triplet(image_path, mask_path, pred_mask_path, save_path):
    """Save a side-by-side visualization of image, GT mask, and predicted mask."""
    try:
        # Load base image
        img = Image.open(image_path).convert("RGB")

        # Load GT mask (if available)
        gt_mask = None
        if mask_path is not None and os.path.exists(mask_path):
            gt_mask = Image.open(mask_path)

        # Load predicted mask
        pred_mask = Image.open(pred_mask_path)

        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        axes[0].imshow(img)
        axes[0].set_title("Image")
        axes[0].axis("off")

        if gt_mask is not None:
            axes[1].imshow(gt_mask, cmap="gray")
            axes[1].set_title("GT Mask")
        else:
            axes[1].imshow(np.zeros((img.height, img.width)), cmap="gray")
            axes[1].set_title("GT Mask (missing)")
        axes[1].axis("off")

        axes[2].imshow(pred_mask, cmap="gray")
        axes[2].set_title("Pred Mask")
        axes[2].axis("off")

        plt.tight_layout()
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print(f"Failed to save visualization for {image_path}: {e}")


def process_prompt(prompt, tokenizer):
    """Process prompt for VLM"""
    prompt_for_vlm = "<image>\n" + f"### User: {prompt} \n"
    input_ids = tokenizer_image_token(
        prompt_for_vlm,
        tokenizer,
        -200,
        return_tensors="pt"
    )
    return input_ids.to(torch.int64).unsqueeze(0)


def process_prompt_seg(prompt, tokenizer):
    """Process prompt for segmentation"""
    prompt_for_vlm = "<image> \n" + prompt
    input_ids = tokenizer_image_token(
        prompt_for_vlm,
        tokenizer,
        -200,
        return_tensors="pt"
    )
    return input_ids.to(torch.int64).unsqueeze(0)


def load_model(args):
    """Load LLMSeg model and checkpoint"""
    print(f"Building model with VLM: {args.vlm_path}")
    model, tokenizer, image_processor, config = build_llm_seg(
        model_path=args.vlm_path,
        model_base=None,
        load_8bit=args.load_8bit,
        load_4bit=args.load_4bit,
        device=args.device,
        sam_model_type=args.sam_model_type,
        sam_checkpoint_path=args.sam_ckpt,
        encoder_type=args.encoder_type,
        image_size=args.image_size
    )
    
    print(f"Loading checkpoint from: {args.checkpoint}")
    tokenizer = model.load_model(args.checkpoint)
    model = model.to(args.device)
    model.eval()
    
    print("✓ Model loaded successfully")
    return model, tokenizer, image_processor, config


def load_dataset(args):
    """Load dataset from CSV files"""
    ann_paths = [p.strip() for p in args.ann_paths.split(',')]
    
    # Load annotations
    import pandas as pd
    dfs = []
    for ann_path in ann_paths:
        df = pd.read_csv(ann_path)
        dfs.append(df)
    
    df = pd.concat(dfs, ignore_index=True)
    df = df.dropna()
    
    # Filter by split
    if 'split' in df.columns:
        df = df[df['split'] == args.split].copy()
    else:
        print(f"Warning: No 'split' column found, using all samples")
    
    print(f"Loaded {len(df)} samples for {args.split} split")
    return df


def infer_single(
    model,
    tokenizer,
    image_processor,
    config,
    image_path,
    prompt,
    device,
    temperature=0.2,
    max_new_tokens=512,
    top_p=0.95
):
    """Run inference on a single image"""
    # Load and process images
    image_tensor = load_image_for_vlm(image_path, image_processor, config)
    image_tensor_for_sam = transform_for_sam(image_path)
    image_tensor_for_sam = image_tensor_for_sam.to(device)
    image_tensor = image_tensor.to(device)
    
    # Process prompts
    input_ids = process_prompt(prompt, tokenizer).to(device)
    input_ids_for_seg = process_prompt_seg(prompt, tokenizer).to(device)
    
    # Run inference
    model.eval()
    with autocast(dtype=torch.float16):
        with torch.no_grad():
            output_mask, output_ids = model.generate(
                input_ids=input_ids,
                input_ids_for_seg=input_ids_for_seg,
                image_tensor_for_vlm=image_tensor,
                image_tensor_for_image_enc=image_tensor_for_sam,
                attention_mask=None,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                top_p=top_p
            )
    
    # Process outputs
    mask_prob = torch.sigmoid(output_mask).cpu().numpy().squeeze()
    mask_prob = (mask_prob - mask_prob.min()) / (mask_prob.max() - mask_prob.min() + 1e-8)
    
    # Decode text
    text_output = tokenizer.decode(output_ids[0, :], skip_special_tokens=True)
    
    return mask_prob, text_output


def main():
    args = parse_args()
    
    # Generate timestamp prefix
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Setup output directory with timestamp prefix
    base_output_dir = Path(args.output_dir)
    output_dir = base_output_dir / f"inference_{timestamp}"
    masks_dir = output_dir / "pred_masks"
    triplets_dir = output_dir / "triplets"
    output_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    triplets_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model, tokenizer, image_processor, config = load_model(args)
    
    # Load dataset
    df = load_dataset(args)
    
    # Limit number of samples if specified
    if args.num_samples is not None:
        df = df.head(args.num_samples)
    
    # Process each sample
    results = []
    device = torch.device(args.device)
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        # Get image path
        if 'image_path' in row:
            image_path = row['image_path']
            if not os.path.isabs(image_path):
                image_path = os.path.join(args.data_root, image_path)
        elif 'image_name' in row:
            split = row.get('split', args.split)
            task = row.get('task', 'head_and_neck')
            image_name = row['image_name']
            image_path = os.path.join(args.data_root, task, f"{split}_images", image_name)
        else:
            print(f"Skipping row {idx}: No image_path or image_name")
            continue
        
        image_path = image_path.replace("\\", "/")
        
        # Get prompt
        prompt = row.get('question', '')
        if not prompt:
            print(f"Skipping row {idx}: No question")
            continue
        
        # Get mask path for visualization
        mask_path = None
        if 'mask_path' in row:
            mask_path = row['mask_path']
            if not os.path.isabs(mask_path):
                mask_path = os.path.join(args.data_root, mask_path)
        else:
            # Infer mask path
            import re
            mask_path = re.sub(r"/(train|test|val)_images/", r"/\1_masks/", image_path)
            if "ISIC" in mask_path:
                mask_path = mask_path.replace(".jpg", ".png")
        
        mask_path = mask_path.replace("\\", "/") if mask_path else None
        
        try:
            # Run inference
            mask_prob, text_output = infer_single(
                model=model,
                tokenizer=tokenizer,
                image_processor=image_processor,
                config=config,
                image_path=image_path,
                prompt=prompt,
                device=device,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                top_p=args.top_p
            )
            
            # Save mask
            mask_uint8 = (mask_prob * 255).astype(np.uint8)
            mask_filename = f"pred_mask_{idx:05d}.png"
            mask_save_path = masks_dir / mask_filename
            cv2.imwrite(str(mask_save_path), mask_uint8)

            # Save side-by-side visualization (image, GT mask, pred mask)
            triplet_filename = f"triplet_{idx:05d}.png"
            triplet_save_path = triplets_dir / triplet_filename
            save_visualization_triplet(
                image_path=image_path,
                mask_path=mask_path,
                pred_mask_path=str(mask_save_path),
                save_path=str(triplet_save_path),
            )
            
            # Store results
            results.append({
                'index': idx,
                'image_path': image_path,
                'mask_path': mask_path,
                'pred_mask_path': str(mask_save_path),
                'triplet_path': str(triplet_save_path),
                'question': prompt,
                'answer': row.get('answer', ''),
                'pred_answer': text_output,
            })
            
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save results CSV with timestamp prefix
    results_df = pd.DataFrame(results)
    csv_path = output_dir / f"results_{args.split}_{timestamp}.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\n✓ Results saved to: {csv_path}")
    print(f"✓ Processed {len(results)} samples")
    print(f"✓ Masks saved to: {masks_dir}")
    print(f"✓ Output directory: {output_dir}")


if __name__ == "__main__":
    main()
