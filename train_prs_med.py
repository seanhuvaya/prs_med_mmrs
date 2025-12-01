import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
from datetime import datetime
import random
import numpy as np
import traceback
import sys

# Import all components including vision backbone
from data.dataset import PRSMedDataLoader  # may be unused, kept for compatibility
from models.vision_backbone.tiny_sam_encoder import TinySAMVisionBackbone
from models.mllm.llava_med_lora_adapter import LLavaMedWithLoRA
from models.decoder.fusion_module import PromptMaskFusionModule
from models.decoder.mask_prediction_module import MaskPredictionModule
from models.loss.objective_function import PRSMedLoss


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Enable deterministic algorithms (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set Python hash seed for reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)

    print(f"✓ Random seed set to {seed} for reproducibility")


def parse_args():
    parser = argparse.ArgumentParser(description='PRS-Med Training (Single GPU)')
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--tinysam_checkpoint', type=str, default='weights/tinysam_42.3.pth',
                       help='Path to TinySAM checkpoint')
    parser.add_argument('--lora_rank', type=int, default=16)
    parser.add_argument('--lora_alpha', type=int, default=16)
    parser.add_argument('--lora_dropout', type=float, default=0.05)
    parser.add_argument('--lambda_seg', type=float, default=1.0)
    parser.add_argument('--lambda_txt', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--image_size', type=int, default=1024)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--deterministic', action='store_true',
                       help='Enable fully deterministic training (may be slower)')
    # Memory optimization arguments
    parser.add_argument('--use_amp', action='store_true', default=False,
                       help='Use Automatic Mixed Precision (AMP) to reduce memory usage')
    parser.add_argument('--no-use_amp', dest='use_amp', action='store_false',
                       help='Disable Automatic Mixed Precision (AMP)')
    parser.set_defaults(use_amp=True)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                       help='Gradient accumulation steps (effective batch size = batch_size * grad_accum_steps)')
    parser.add_argument('--gradient_checkpointing', action='store_true', default=False,
                       help='Use gradient checkpointing to trade compute for memory')
    parser.add_argument('--compile_model', action='store_true', default=False,
                       help='Compile model with torch.compile (PyTorch 2.0+)')
    return parser.parse_args()


class PRSMedModel(nn.Module):
    """
    Complete PRS-Med model with explicit dtype handling.
    Supports joint seg + text loss training by passing answers through the MLLM.
    """
    def __init__(self, args, device):
        super().__init__()
        self.device = device
        self.image_size = args.image_size

        # Vision backbone (TinySAM)
        self.vision_backbone = TinySAMVisionBackbone(
            checkpoint_path=args.tinysam_checkpoint,
            image_size=args.image_size,
            device=str(device)
        )
        self.vision_backbone = self.vision_backbone.to(device)

        # Multimodal LLM with LoRA
        self.mllm = LLavaMedWithLoRA(
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            freeze_llm=True,
            device=str(device)
        )
        self.mllm = self.mllm.to(device)

        # Fusion and mask modules - explicitly set to float32
        self.fusion_module = PromptMaskFusionModule().to(device).float()
        self.mask_predictor = MaskPredictionModule().to(device).float()

    def preprocess_images(self, images):
        """
        Preprocess images and ensure float32.
        """
        if isinstance(images, torch.Tensor):
            B, C, H, W = images.shape

            # Resize if necessary
            if H != self.image_size or W != self.image_size:
                images = torch.nn.functional.interpolate(
                    images,
                    size=(self.image_size, self.image_size),
                    mode='bilinear',
                    align_corners=False
                )

            # Normalize if needed (check if already normalized)
            if images.max() > 2.0:  # Likely unnormalized [0, 255]
                images = images / 255.0
                mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
                images = (images - mean) / std

        return images.float()

    def forward(self, images, text_prompts, answers=None, compute_lm_loss=False):
        """
        Forward pass with seg + optional text (LM) loss.

        Args:
            images: Tensor (B, C, H, W)
            text_prompts: list[str] questions
            answers: list[str] ground-truth answers (optional, used when compute_lm_loss=True)
            compute_lm_loss: whether to compute LM loss inside MLLM
        """
        # Ensure input images are on the correct device
        if isinstance(images, torch.Tensor) and images.device != self.device:
            images = images.to(self.device)

        processed_images = self.preprocess_images(images)
        if processed_images.device != self.device:
            processed_images = processed_images.to(self.device)

        # 1. Vision backbone features
        z_image = self.vision_backbone(processed_images).float()  # (B, 256, 16, 16)
        if z_image.device != self.device:
            z_image = z_image.to(self.device)

        # 2. MLLM (with optional LM loss)
        mllm_output = self.mllm(
            processed_images,
            text_prompts,
            answers=answers if compute_lm_loss and answers is not None else None,
            return_projected=True,
            compute_lm_loss=compute_lm_loss and answers is not None,
        )
        z_emb = mllm_output["z_emb"].float()
        z_txt_logits = mllm_output["z_txt"].float()
        pred_ids = mllm_output["pred_ids"]
        lm_loss = mllm_output.get("lm_loss", None)

        if z_emb.device != self.device:
            z_emb = z_emb.to(self.device)
        if z_txt_logits.device != self.device:
            z_txt_logits = z_txt_logits.to(self.device)

        # 3. Fuse visual and multimodal features
        z_fused = self.fusion_module(z_image, z_emb)  # (B, 256, 16, 16)
        if z_fused.device != self.device:
            z_fused = z_fused.to(self.device)

        # 4. Generate segmentation mask
        z_mask = self.mask_predictor(z_fused)  # (B, 1, 1024, 1024)
        if z_mask.device != self.device:
            z_mask = z_mask.to(self.device)

        return {
            "z_mask": z_mask,              # Segmentation logits
            "z_txt_logits": z_txt_logits,  # Text logits (not used for loss now, but kept)
            "pred_ids": pred_ids,          # Predicted token IDs
            "lm_loss": lm_loss,            # Scalar LM loss or None
        }


def check_disk_space(path, required_gb=5.0):
    """Check if there's enough disk space available"""
    try:
        import shutil
        stat = shutil.disk_usage(path)
        free_gb = stat.free / (1024 ** 3)
        if free_gb < required_gb:
            print(f"WARNING: Low disk space: {free_gb:.2f} GB free (need at least {required_gb} GB)")
            return False
        return True
    except Exception as e:
        print(f"WARNING: Could not check disk space: {e}")
        return True


def save_checkpoint(epoch, model, optimizer, checkpoint_dir, is_best=False, max_retries=3):
    """Save complete model checkpoint with atomic writes and retry logic (single GPU)"""
    if checkpoint_dir is None:
        print("ERROR: checkpoint_dir is None, cannot save checkpoint")
        return False

    if not os.path.exists(checkpoint_dir):
        print(f"ERROR: Checkpoint directory does not exist: {checkpoint_dir}")
        try:
            os.makedirs(checkpoint_dir, exist_ok=True)
            print(f"Created checkpoint directory: {checkpoint_dir}")
        except Exception as e:
            print(f"ERROR: Failed to create checkpoint directory: {e}")
            return False

    if not check_disk_space(checkpoint_dir, required_gb=10.0):
        print("ERROR: Insufficient disk space to save checkpoint")
        return False

    model_to_save = model  # no DDP wrapping

    if is_best:
        filename = f'best_model_epoch_{epoch+1}.pth'
    else:
        filename = f'checkpoint_epoch_{epoch+1}.pth'

    checkpoint_path = os.path.join(checkpoint_dir, filename)
    temp_path = checkpoint_path + '.tmp'

    try:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        print(f"ERROR: Failed to prepare checkpoint data: {e}")
        traceback.print_exc()
        return False

    for attempt in range(max_retries):
        try:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass

            print(f"  Saving checkpoint (attempt {attempt + 1}/{max_retries})...")
            torch.save(checkpoint, temp_path)

            sys.stdout.flush()
            if hasattr(os, 'sync'):
                try:
                    os.sync()
                except:
                    pass

            if not os.path.exists(temp_path):
                raise RuntimeError(f"Temporary checkpoint file was not created: {temp_path}")

            temp_size = os.path.getsize(temp_path)
            if temp_size == 0:
                raise RuntimeError(f"Temporary checkpoint file is empty: {temp_path}")

            os.rename(temp_path, checkpoint_path)

            if not os.path.exists(checkpoint_path):
                raise RuntimeError(f"Checkpoint file was not created after rename: {checkpoint_path}")

            final_size = os.path.getsize(checkpoint_path)
            if final_size != temp_size:
                raise RuntimeError(f"File size mismatch after rename: {final_size} != {temp_size}")

            file_size_mb = final_size / (1024 * 1024)
            print(f"✓ Checkpoint saved to {checkpoint_path}")
            print(f"  File size: {file_size_mb:.2f} MB")
            return True

        except Exception as e:
            print(f"ERROR: Failed to save checkpoint on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                print(f"  Retrying in 2 seconds...")
                import time as _time
                _time.sleep(2)
                continue
            else:
                print(f"ERROR: Failed to save checkpoint after {max_retries} attempts")
                traceback.print_exc()
                return False

    if os.path.exists(temp_path):
        try:
            os.remove(temp_path)
        except:
            pass

    return False


def main():
    args = parse_args()

    # Disable tokenizer parallelism noise
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    # Device (single GPU or CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Seed
    set_seed(args.seed)

    torch.set_default_dtype(torch.float32)

    if args.deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        print("✓ Deterministic algorithms enabled")

    # Checkpoint directory
    checkpoint_dir = None
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_dir = os.path.join(args.checkpoint_dir, f'training_{timestamp}')
    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"✓ Checkpoint directory created: {checkpoint_dir}")
    print(f"  Absolute path: {os.path.abspath(checkpoint_dir)}")

    if os.path.exists(checkpoint_dir) and os.access(checkpoint_dir, os.W_OK):
        print(f"  Directory is writable")
    else:
        print(f"  WARNING: Directory may not be writable!")

    try:
        test_file = os.path.join(checkpoint_dir, '.test_write')
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
        print(f"  Write test: SUCCESS")
    except Exception as e:
        print(f"  Write test: FAILED - {e}")

    # Data
    print(f"Loading data from {args.data_root}...")
    from data.dataset import PRSMedDataset
    train_dataset = PRSMedDataset(split='train', data_root=args.data_root)
    val_dataset = PRSMedDataset(split='val', data_root=args.data_root)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")

    # Model
    model = PRSMedModel(args, device).to(device)

    # Optional gradient checkpointing
    if args.gradient_checkpointing:
        if hasattr(model.vision_backbone.encoder, 'use_checkpoint'):
            for module in model.vision_backbone.encoder.modules():
                if hasattr(module, 'use_checkpoint'):
                    module.use_checkpoint = True
        if hasattr(model.mllm.model, 'gradient_checkpointing_enable'):
            model.mllm.model.gradient_checkpointing_enable()
        print("✓ Gradient checkpointing enabled")

    # Optional compile
    if args.compile_model and hasattr(torch, 'compile'):
        print("Compiling model with torch.compile...")
        model = torch.compile(model, mode='reduce-overhead')
        print("✓ Model compiled")

    # Optimizer & loss
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        weight_decay=0.01,
        betas=(0.9, 0.999)
    )
    criterion = PRSMedLoss(lambda_seg=args.lambda_seg, lambda_txt=args.lambda_txt)

    scaler = None
    if args.use_amp and torch.cuda.is_available():
        scaler = torch.cuda.amp.GradScaler()
        print("✓ Mixed precision training (AMP) enabled")

    best_val_loss = float('inf')

    print("Starting training...")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
    print(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")

    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss_total = 0.0
        epoch_loss_seg = 0.0
        epoch_loss_txt = 0.0
        epoch_start_time = time.time()

        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            images = batch['image'].to(device, non_blocking=True)
            masks = batch['mask'].to(device, non_blocking=True)
            questions = batch['question']
            answers = batch['answer']

            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(images, questions, answers=answers, compute_lm_loss=True)
                    pred_masks = outputs["z_mask"]
                    lm_loss = outputs["lm_loss"]

                    loss_dict = criterion(
                        z_mask=pred_masks,
                        y_mask=masks,
                        lm_loss=lm_loss
                    )

                    loss_total = loss_dict["loss_total"] / args.gradient_accumulation_steps

                scaler.scale(loss_total).backward()
            else:
                outputs = model(images, questions, answers=answers, compute_lm_loss=True)
                pred_masks = outputs["z_mask"]
                lm_loss = outputs["lm_loss"]

                loss_dict = criterion(
                    z_mask=pred_masks,
                    y_mask=masks,
                    lm_loss=lm_loss
                )

                loss_total = loss_dict["loss_total"] / args.gradient_accumulation_steps
                loss_total.backward()

            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                    optimizer.step()

                optimizer.zero_grad()

            epoch_loss_total += loss_total.item() * args.gradient_accumulation_steps
            epoch_loss_seg += loss_dict["loss_seg"].item() * args.gradient_accumulation_steps
            epoch_loss_txt += loss_dict["loss_txt"].item() * args.gradient_accumulation_steps

            if batch_idx % 10 == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                print(f'Epoch {epoch+1}, Batch {batch_idx}, '
                      f'Total Loss: {loss_total.item() * args.gradient_accumulation_steps:.4f}, '
                      f'Seg Loss: {loss_dict["loss_seg"].item():.4f}, '
                      f'Text Loss: {loss_dict["loss_txt"].item():.4f}')

        if len(train_loader) % args.gradient_accumulation_steps != 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
            optimizer.zero_grad()

        avg_train_loss_total = epoch_loss_total / len(train_loader)
        avg_train_loss_seg = epoch_loss_seg / len(train_loader)
        avg_train_loss_txt = epoch_loss_txt / len(train_loader)
        epoch_time = time.time() - epoch_start_time

        print(f'Epoch {epoch+1} - TRAIN - '
              f'Total Loss: {avg_train_loss_total:.4f}, '
              f'Seg Loss: {avg_train_loss_seg:.4f}, '
              f'Text Loss: {avg_train_loss_txt:.4f}, '
              f'Time: {epoch_time:.2f}s')

        # Validation
        model.eval()
        val_loss_total = 0.0
        val_loss_seg = 0.0
        val_loss_txt = 0.0

        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device, non_blocking=True)
                masks = batch['mask'].to(device, non_blocking=True)
                questions = batch['question']
                answers = batch['answer']

                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        outputs = model(images, questions, answers=answers, compute_lm_loss=True)
                        pred_masks = outputs["z_mask"]
                        lm_loss = outputs["lm_loss"]

                        loss_dict = criterion(
                            z_mask=pred_masks,
                            y_mask=masks,
                            lm_loss=lm_loss
                        )
                else:
                    outputs = model(images, questions, answers=answers, compute_lm_loss=True)
                    pred_masks = outputs["z_mask"]
                    lm_loss = outputs["lm_loss"]

                    loss_dict = criterion(
                        z_mask=pred_masks,
                        y_mask=masks,
                        lm_loss=lm_loss
                    )

                val_loss_total += loss_dict["loss_total"].item()
                val_loss_seg += loss_dict["loss_seg"].item()
                val_loss_txt += loss_dict["loss_txt"].item()

        avg_val_loss_total = val_loss_total / len(val_loader)
        avg_val_loss_seg = val_loss_seg / len(val_loader)
        avg_val_loss_txt = val_loss_txt / len(val_loader)

        print(f'Epoch {epoch+1} - VALIDATION - '
              f'Total Loss: {avg_val_loss_total:.4f}, '
              f'Seg Loss: {avg_val_loss_seg:.4f}, '
              f'Text Loss: {avg_val_loss_txt:.4f}')

        # Save checkpoints
        if checkpoint_dir is not None:
            if avg_val_loss_total < best_val_loss:
                best_val_loss = avg_val_loss_total
                success = save_checkpoint(epoch, model, optimizer, checkpoint_dir, is_best=True)
                if success:
                    print(f"✓ New best model saved with val_loss: {avg_val_loss_total:.4f}")
                else:
                    print(f"✗ Failed to save best model!")

            if (epoch + 1) % 5 == 0:
                success = save_checkpoint(epoch, model, optimizer, checkpoint_dir)
                if success:
                    print(f"✓ Periodic checkpoint saved at epoch {epoch+1}")
                else:
                    print(f"✗ Failed to save periodic checkpoint!")

            if epoch == 0:
                success = save_checkpoint(epoch, model, optimizer, checkpoint_dir)
                if success:
                    print(f"✓ Initial checkpoint saved at epoch {epoch+1}")
                else:
                    print(f"✗ Failed to save initial checkpoint!")
        else:
            print("ERROR: Cannot save checkpoint - checkpoint_dir is None!")

    # Final checkpoint summary
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    if checkpoint_dir is not None:
        print(f"Final checkpoints saved in: {checkpoint_dir}")
        if os.path.exists(checkpoint_dir):
            checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
            if checkpoint_files:
                print(f"Saved {len(checkpoint_files)} checkpoint(s):")
                for f in sorted(checkpoint_files):
                    file_path = os.path.join(checkpoint_dir, f)
                    file_size = os.path.getsize(file_path) / (1024 * 1024)
                    print(f"  - {f} ({file_size:.2f} MB)")
            else:
                print("WARNING: No checkpoint files found in directory!")
    else:
        print("WARNING: checkpoint_dir was None, no checkpoints were saved!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"ERROR: Training failed with exception:")
        print(f"{'='*60}")
        print(f"Exception type: {type(e).__name__}")
        print(f"Exception message: {str(e)}")
        print(f"\nFull traceback:")
        traceback.print_exc()
        print(f"{'='*60}\n")
        sys.exit(1)
