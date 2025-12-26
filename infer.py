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
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    image_sam_transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    try:
        image = load_image(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Validate image dimensions
        if image.size[0] == 0 or image.size[1] == 0:
            raise ValueError(f"Image has invalid dimensions: {image.size} for {image_path}")
        
        # Ensure image is in RGB mode
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image_tensor = image_sam_transform(image)
        return image_tensor.to(torch.float32).unsqueeze(0)
    except (IOError, OSError) as e:
        raise RuntimeError(f"Error loading image file {image_path}: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Error processing image {image_path} (size: {image.size if 'image' in locals() else 'unknown'}, mode: {image.mode if 'image' in locals() else 'unknown'}): {e}") from e


def load_image_for_vlm(image_path, image_processor, config):
    """Load and process image for VLM"""
    image_pil = load_image(image_path)
    image_tensor = process_images(
        [image_pil],
        image_processor,
        config
    )
    # Keep batch dimension (1, C, H, W) for CLIP vision tower
    # Note: dtype will be matched to model dtype in generate() method
    return image_tensor


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
    
    # Verify checkpoint directory exists and has required files
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint directory not found: {args.checkpoint}")
    
    mask_decoder_path = os.path.join(args.checkpoint, "mask_decoder.pth")
    if not os.path.exists(mask_decoder_path):
        print(f"⚠ WARNING: mask_decoder.pth not found in checkpoint directory!")
        print(f"   Expected: {mask_decoder_path}")
        print(f"   This may cause NaN outputs if mask decoder weights are not loaded.")
        print(f"   Checkpoint directory contents:")
        if os.path.isdir(args.checkpoint):
            for f in os.listdir(args.checkpoint):
                print(f"     - {f}")
    else:
        print(f"✓ Found mask_decoder.pth: {mask_decoder_path}")
    
    tokenizer = model.load_model(args.checkpoint)
    model = model.to(args.device)
    model.eval()
    
    # Verify model components are loaded
    print("\n✓ Model loaded successfully")
    print(f"  - Image encoder: {'✓' if model.image_encoder is not None else '✗'}")
    print(f"  - Mask decoder: {'✓' if model.mask_decoder is not None else '✗'}")
    if model.image_encoder is not None:
        # Check if image encoder has parameters (not just initialized)
        num_params = sum(p.numel() for p in model.image_encoder.parameters())
        print(f"  - Image encoder params: {num_params:,}")
    if model.mask_decoder is not None:
        num_params = sum(p.numel() for p in model.mask_decoder.parameters())
        print(f"  - Mask decoder params: {num_params:,}")
        
        # Check for NaN/Inf in mask decoder weights (indicates uninitialized or corrupted weights)
        has_nan = False
        has_inf = False
        for name, param in model.mask_decoder.named_parameters():
            if torch.isnan(param).any():
                print(f"  ⚠ WARNING: NaN detected in mask_decoder.{name}")
                has_nan = True
            if torch.isinf(param).any():
                print(f"  ⚠ WARNING: Inf detected in mask_decoder.{name}")
                has_inf = True
        
        if has_nan or has_inf:
            print(f"  ✗ ERROR: Mask decoder has NaN/Inf weights! Checkpoint may be corrupted or not loaded correctly.")
            print(f"     Checkpoint path: {args.checkpoint}")
            print(f"     Expected files: mask_decoder.pth should exist in checkpoint directory")
        else:
            # Check if weights are all zeros (uninitialized)
            all_zeros = all(torch.allclose(p, torch.zeros_like(p)) for p in model.mask_decoder.parameters())
            if all_zeros:
                print(f"  ⚠ WARNING: Mask decoder weights are all zeros! Checkpoint may not have been loaded.")
            else:
                print(f"  ✓ Mask decoder weights appear valid (no NaN/Inf, not all zeros)")
    
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
    
    # CRITICAL: Ensure image_tensor is in float16 to match model dtype
    # The vision tower converts outputs to match input dtype, so if images are float32,
    # vision tower outputs will be float32, causing mismatch with mm_projector (float16)
    # Convert to float16 early to ensure consistency
    if image_tensor.dtype != torch.float16:
        image_tensor = image_tensor.to(dtype=torch.float16)
    
    # Process prompts
    input_ids = process_prompt(prompt, tokenizer).to(device)
    input_ids_for_seg = process_prompt_seg(prompt, tokenizer).to(device)
    
    # Run inference
    model.eval()
    
    # Check for NaN/Inf in inputs before forward pass
    if torch.isnan(image_tensor).any() or torch.isinf(image_tensor).any():
        raise ValueError(f"NaN/Inf detected in image_tensor for {image_path}")
    if torch.isnan(image_tensor_for_sam).any() or torch.isinf(image_tensor_for_sam).any():
        raise ValueError(f"NaN/Inf detected in image_tensor_for_sam for {image_path}")
    
    # Ensure mask decoder uses float32 to avoid numerical instability
    # The generate() method will handle excluding mask decoder from autocast
    original_dtype = next(model.mask_decoder.parameters()).dtype
    model.mask_decoder = model.mask_decoder.float()
    model.mask_decoder.eval()  # Ensure eval mode
    
    try:
        with autocast(dtype=torch.float16):
            with torch.no_grad():
                output_mask, output_ids = model.generate(
                    input_ids=input_ids,
                    input_ids_for_seg=input_ids_for_seg,
                    image_tensor_for_vlm=image_tensor,
                    image_tensor_for_image_enc=image_tensor_for_sam,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                    top_p=top_p
                )
    finally:
        # Restore original dtype
        model.mask_decoder = model.mask_decoder.to(dtype=original_dtype)
    
    # Check for NaN in output
    if torch.isnan(output_mask).any() or torch.isinf(output_mask).any():
        # Detailed debugging: check intermediate values in mask decoder
        print(f"\n[DEBUG NaN] Checking intermediate values for {image_path}:")
        with torch.no_grad():
            # Re-run mask decoder with debugging
            image_embedding = model.image_encoder(image_tensor_for_sam.float())
            
            # Use the same model selection logic as in generate() method
            if hasattr(model.model, 'extract_last_hidden_state'):
                model_for_embedding = model.model
            else:
                model_for_embedding = model.base_model
            
            # Ensure model is in correct dtype
            model_dtype = next(model_for_embedding.parameters()).dtype
            if model_dtype == torch.bfloat16:
                model_for_embedding = model_for_embedding.to(dtype=torch.float16)
            
            # CRITICAL: Ensure image tensor is in float16 to match model dtype
            # The vision tower converts outputs to match input dtype, so images must be float16
            image_tensor_debug = image_tensor
            target_dtype = next(model_for_embedding.parameters()).dtype
            if image_tensor_debug.dtype != target_dtype:
                image_tensor_debug = image_tensor_debug.to(dtype=target_dtype)
            
            # Also ensure vision tower is in float16
            try:
                vision_tower = model_for_embedding.get_vision_tower() if hasattr(model_for_embedding, 'get_vision_tower') else None
                if vision_tower is not None:
                    actual_tower = vision_tower.vision_tower if hasattr(vision_tower, 'vision_tower') else vision_tower
                    if hasattr(actual_tower, 'to') and hasattr(actual_tower, 'parameters'):
                        tower_dtype = next(actual_tower.parameters()).dtype
                        if tower_dtype != target_dtype:
                            actual_tower = actual_tower.to(dtype=target_dtype)
                            if hasattr(vision_tower, 'vision_tower'):
                                vision_tower.vision_tower = actual_tower
            except Exception:
                pass
            
            prompt_embedding = model_for_embedding.extract_last_hidden_state(
                input_ids=input_ids_for_seg if input_ids_for_seg is not None else input_ids,
                images=image_tensor_debug,
                do_sample=False,
                temperature=0,
                max_new_tokens=max_new_tokens,
                top_p=top_p
            )["hidden_states"][-1]
            
            print(f"  image_embedding: shape={image_embedding.shape}, has_nan={torch.isnan(image_embedding).any()}, has_inf={torch.isinf(image_embedding).any()}")
            print(f"  prompt_embedding: shape={prompt_embedding.shape}, has_nan={torch.isnan(prompt_embedding).any()}, has_inf={torch.isinf(prompt_embedding).any()}")
            
            # Check mask decoder weights
            for name, param in list(model.mask_decoder.named_parameters())[:3]:  # Check first 3 params
                has_nan = torch.isnan(param).any()
                has_inf = torch.isinf(param).any()
                if has_nan or has_inf:
                    print(f"  mask_decoder.{name}: has_nan={has_nan}, has_inf={has_inf}, mean={param.mean().item():.6f}")
        
        raise ValueError(
            f"NaN/Inf detected in output_mask for {image_path}.\n"
            "This indicates a model issue. Possible causes:\n"
            "  1. Checkpoint not loaded correctly (check checkpoint path)\n"
            "  2. Model weights are corrupted or uninitialized\n"
            "  3. Numerical instability in mask decoder\n"
            "  4. Wrong model architecture for this checkpoint\n"
            "Check the debug output above to see where NaN first appears."
        )
    
    # Process outputs
    # Debug: print raw output values for first few samples
    if hasattr(infer_single, '_debug_count'):
        infer_single._debug_count += 1
    else:
        infer_single._debug_count = 0
    
    if infer_single._debug_count < 3:
        print(f"\n[DEBUG] Sample {infer_single._debug_count}:")
        print(f"  output_mask shape: {output_mask.shape}")
        print(f"  output_mask min/max/mean: {output_mask.min().item():.4f} / {output_mask.max().item():.4f} / {output_mask.mean().item():.4f}")
        print(f"  output_mask std: {output_mask.std().item():.4f}")
    
    # Apply sigmoid to get probabilities
    mask_prob = torch.sigmoid(output_mask).cpu().numpy().squeeze()
    
    if infer_single._debug_count < 3:
        print(f"  After sigmoid - min/max/mean: {mask_prob.min():.4f} / {mask_prob.max():.4f} / {mask_prob.mean():.4f}")
        print(f"  Non-zero pixels: {(mask_prob > 0.1).sum()} / {mask_prob.size}")
    
    # Don't normalize after sigmoid - sigmoid already gives [0,1] range
    # The min-max normalization was destroying the signal if all values were similar
    # Instead, just clip to [0,1] to ensure valid range
    mask_prob = np.clip(mask_prob, 0.0, 1.0)
    
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
            # Fix: if image_path points to a mask directory, correct it
            if '_masks' in image_path:
                image_path = image_path.replace('_masks', '_images')
                # Also fix file extension if needed (masks are often .png, images might be .jpg)
                if image_path.endswith('.png') and not os.path.exists(image_path):
                    # Try .jpg extension
                    image_path_jpg = image_path.replace('.png', '.jpg')
                    if os.path.exists(image_path_jpg):
                        image_path = image_path_jpg
        elif 'image_name' in row:
            split = row.get('split', args.split)
            task = row.get('task', 'head_and_neck')
            image_name = row['image_name']
            image_path = os.path.join(args.data_root, task, f"{split}_images", image_name)
        else:
            print(f"Skipping row {idx}: No image_path or image_name")
            continue
        
        image_path = image_path.replace("\\", "/")
        
        # Debug: print image path for first few samples
        if idx < 3:
            print(f"Processing image {idx}: {image_path}")
            if not os.path.exists(image_path):
                print(f"  WARNING: Image file does not exist!")
            elif '_masks' in image_path:
                print(f"  WARNING: Image path still contains '_masks' - this might be wrong!")
        
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
            # Validate image path exists
            if not os.path.exists(image_path):
                print(f"Skipping row {idx}: Image file not found: {image_path}")
                continue
            
            # Run inference
            try:
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
            except ValueError as e:
                if "NaN/Inf detected" in str(e):
                    print(f"⚠ Skipping row {idx} due to NaN in model output (checkpoint issue)")
                    print(f"   Image: {image_path}")
                    print(f"   This suggests the checkpoint may be corrupted or incompatible.")
                    continue
                else:
                    raise
            
            # Save mask
            # Ensure mask_prob is 2D (H, W) for saving
            if mask_prob.ndim > 2:
                mask_prob = mask_prob.squeeze()
            if mask_prob.ndim == 1:
                # If somehow 1D, reshape (this shouldn't happen)
                print(f"Warning: mask_prob is 1D with shape {mask_prob.shape}, skipping save")
                continue
            
            # Debug first few masks
            if idx < 3:
                print(f"  Saving mask {idx}: shape={mask_prob.shape}, min={mask_prob.min():.4f}, max={mask_prob.max():.4f}, mean={mask_prob.mean():.4f}")
                print(f"  Pixels > 0.5: {(mask_prob > 0.5).sum()} / {mask_prob.size}")
            
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
