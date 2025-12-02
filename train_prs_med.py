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

# Import components
from data.dataset import PRSMedDataLoader  # optional, kept for compatibility
from data.dataset import PRSMedDataset
from models.vision_backbone.tiny_sam_encoder import TinySAMVisionBackbone
from models.mllm.llava_med_lora_adapter import LLavaMedWithLoRA
from models.decoder.fusion_module import PromptMaskFusionModule
from models.decoder.mask_prediction_module import MaskPredictionModule
from models.loss.objective_function import PRSMedLoss


# ------------------------------------------------------------- #
# Utils
# ------------------------------------------------------------- #
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed)

    print(f"✓ Random seed set to {seed}")


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
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--image_size', type=int, default=1024)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--deterministic', action='store_true',
                        help='Enable deterministic training (slower)')
    parser.add_argument('--use_amp', action='store_true', default=True,
                        help='Use AMP (mixed precision)')
    parser.add_argument('--no-use_amp', dest='use_amp', action='store_false')
    parser.set_defaults(use_amp=True)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--gradient_checkpointing', action='store_true', default=False)
    parser.add_argument('--compile_model', action='store_true', default=False)
    return parser.parse_args()


def prepare_text_targets(full_texts, tokenizer, max_length=512):
    """
    Tokenize the EXACT same text used as input to MLLM:
      "USER: <image>\\n{question}\\nASSISTANT: {answer}"
    so CE(ŷ_txt, z_txt) is valid (Eq. 7).
    """
    tokenized = tokenizer(
        full_texts,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )
    return tokenized.input_ids


def check_disk_space(path, required_gb=5.0):
    try:
        import shutil
        stat = shutil.disk_usage(path)
        free_gb = stat.free / (1024 ** 3)
        if free_gb < required_gb:
            print(f"WARNING: Low disk space: {free_gb:.2f} GB free (need {required_gb} GB)")
            return False
        return True
    except Exception as e:
        print(f"WARNING: Could not check disk space: {e}")
        return True


def save_checkpoint(epoch, model, optimizer, checkpoint_dir, is_best=False, max_retries=3):
    if checkpoint_dir is None:
        print("ERROR: checkpoint_dir is None")
        return False

    if not os.path.exists(checkpoint_dir):
        try:
            os.makedirs(checkpoint_dir, exist_ok=True)
        except Exception as e:
            print(f"ERROR: Failed to create checkpoint_dir: {e}")
            return False

    if not check_disk_space(checkpoint_dir, required_gb=10.0):
        print("ERROR: Not enough disk space to save checkpoint")
        return False

    model_to_save = model

    if is_best:
        filename = f'best_model_epoch_{epoch+1}.pth'
    else:
        filename = f'checkpoint_epoch_{epoch+1}.pth'

    ckpt_path = os.path.join(checkpoint_dir, filename)
    tmp_path = ckpt_path + '.tmp'

    try:
        ckpt = {
            'epoch': epoch,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'timestamp': datetime.now().isoformat(),
        }
    except Exception as e:
        print(f"ERROR preparing checkpoint data: {e}")
        traceback.print_exc()
        return False

    for attempt in range(max_retries):
        try:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except:
                    pass

            print(f"  Saving checkpoint (attempt {attempt+1}/{max_retries})...")
            torch.save(ckpt, tmp_path)

            sys.stdout.flush()
            if hasattr(os, "sync"):
                try:
                    os.sync()
                except:
                    pass

            if not os.path.exists(tmp_path):
                raise RuntimeError("Temp checkpoint file not created")

            tmp_size = os.path.getsize(tmp_path)
            if tmp_size == 0:
                raise RuntimeError("Temp checkpoint file is empty")

            os.rename(tmp_path, ckpt_path)

            if not os.path.exists(ckpt_path):
                raise RuntimeError("Checkpoint not found after rename")

            final_size = os.path.getsize(ckpt_path)
            if final_size != tmp_size:
                raise RuntimeError("Checkpoint file size mismatch")

            size_mb = final_size / (1024 * 1024)
            print(f"✓ Checkpoint saved to {ckpt_path} ({size_mb:.2f} MB)")
            return True

        except Exception as e:
            print(f"ERROR saving checkpoint attempt {attempt+1}: {e}")
            if attempt < max_retries - 1:
                import time as _time
                print("  Retrying in 2 seconds...")
                _time.sleep(2)
            else:
                print("ERROR: Failed to save checkpoint after retries")
                traceback.print_exc()
                return False

    if os.path.exists(tmp_path):
        try:
            os.remove(tmp_path)
        except:
            pass

    return False


# ------------------------------------------------------------- #
# Model wrapper
# ------------------------------------------------------------- #
class PRSMedModel(nn.Module):
    """
    Complete PRS-Med model:
      - Vision backbone: TinySAM
      - MLLM: LLaVA-Med + LoRA (LLavaMedWithLoRA)
      - Fusion: PromptMaskFusionModule
      - Seg head: MaskPredictionModule

    Text branch follows PRS-Med:
      z_emb = F_mllm(X_image, X_txt)
      z_txt_logits = p(z_txt | X_image, X_txt)
    """
    def __init__(self, args, device):
        super().__init__()
        self.device = device
        self.image_size = args.image_size

        # Vision backbone
        self.vision_backbone = TinySAMVisionBackbone(
            checkpoint_path=args.tinysam_checkpoint,
            image_size=args.image_size,
            device=str(device),
        ).to(device)

        # MLLM with LoRA
        self.mllm = LLavaMedWithLoRA(
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            freeze_llm=True,
            device=str(device),
        ).to(device)

        # Fusion + mask predictor
        self.fusion_module = PromptMaskFusionModule().to(device).float()
        self.mask_predictor = MaskPredictionModule().to(device).float()

    def preprocess_images(self, images: torch.Tensor):
        if isinstance(images, torch.Tensor):
            B, C, H, W = images.shape

            if H != self.image_size or W != self.image_size:
                images = torch.nn.functional.interpolate(
                    images,
                    size=(self.image_size, self.image_size),
                    mode='bilinear',
                    align_corners=False,
                )

            if images.max() > 2.0:
                images = images / 255.0
                mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
                images = (images - mean) / std

        return images.float()

    def forward(
        self,
        images: torch.Tensor,
        questions: list,
        answers: list,
        training_text: bool = True,
    ):
        # Images to device
        if images.device != self.device:
            images = images.to(self.device)

        processed_images = self.preprocess_images(images)
        if processed_images.device != self.device:
            processed_images = processed_images.to(self.device)

        # 1. Vision backbone
        z_image = self.vision_backbone(processed_images).float()
        if z_image.device != self.device:
            z_image = z_image.to(self.device)

        # 2. MLLM
        mllm_out = self.mllm(
            processed_images,
            questions,
            answers=answers,
            training_text=training_text,
            return_projected=True,
        )
        z_emb = mllm_out["z_emb"].float()
        z_txt_logits = mllm_out["z_txt"].float()
        pred_ids = mllm_out["pred_ids"]

        if z_emb.device != self.device:
            z_emb = z_emb.to(self.device)
        if z_txt_logits.device != self.device:
            z_txt_logits = z_txt_logits.to(self.device)

        # 3. Fusion
        z_fused = self.fusion_module(z_image, z_emb)
        if z_fused.device != self.device:
            z_fused = z_fused.to(self.device)

        # 4. Segmentation
        z_mask = self.mask_predictor(z_fused)
        if z_mask.device != self.device:
            z_mask = z_mask.to(self.device)

        return {
            "z_mask": z_mask,
            "z_txt_logits": z_txt_logits,
            "pred_ids": pred_ids,
        }


# ------------------------------------------------------------- #
# Main training
# ------------------------------------------------------------- #
def main():
    args = parse_args()

    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    set_seed(args.seed)
    torch.set_default_dtype(torch.float32)

    if args.deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        print("✓ Deterministic algorithms enabled")

    # Checkpoint dir
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_dir = os.path.join(args.checkpoint_dir, f"training_{timestamp}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"✓ Checkpoint directory: {checkpoint_dir}")
    print(f"  Absolute path: {os.path.abspath(checkpoint_dir)}")

    try:
        test_file = os.path.join(checkpoint_dir, ".test_write")
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
        print("  Write test: SUCCESS")
    except Exception as e:
        print(f"  Write test: FAILED - {e}")

    # Data
    print(f"Loading data from {args.data_root}...")
    train_dataset = PRSMedDataset(split='train', data_root=args.data_root)
    val_dataset = PRSMedDataset(split='val', data_root=args.data_root)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")

    # Model
    model = PRSMedModel(args, device).to(device)

    # Gradient checkpointing
    if args.gradient_checkpointing:
        if hasattr(model.vision_backbone.encoder, "use_checkpoint"):
            for module in model.vision_backbone.encoder.modules():
                if hasattr(module, "use_checkpoint"):
                    module.use_checkpoint = True
        if hasattr(model.mllm.model, "gradient_checkpointing_enable"):
            model.mllm.model.gradient_checkpointing_enable()
        print("✓ Gradient checkpointing enabled")

    # Compile model
    if args.compile_model and hasattr(torch, "compile"):
        print("Compiling model with torch.compile...")
        model = torch.compile(model, mode="reduce-overhead")
        print("✓ Model compiled")

    # Optimizer & loss
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        weight_decay=0.01,
        betas=(0.9, 0.999),
    )
    criterion = PRSMedLoss(lambda_seg=args.lambda_seg, lambda_txt=args.lambda_txt)

    scaler = None
    if args.use_amp and torch.cuda.is_available():
        scaler = torch.cuda.amp.GradScaler()
        print("✓ Mixed precision (AMP) enabled")

    best_val_loss = float('inf')

    print("Starting training...")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Grad accumulation: {args.gradient_accumulation_steps}")

    # Training loop
    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss_total = 0.0
        epoch_loss_seg = 0.0
        epoch_loss_txt = 0.0
        epoch_start = time.time()

        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            questions = batch["question"]
            answers = batch["answer"]

            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(
                        images,
                        questions,
                        answers,
                        training_text=True,
                    )
                    pred_masks = outputs["z_mask"]
                    text_logits = outputs["z_txt_logits"]

                    # Build full text sequences to match MLLM inputs
                    full_texts = [
                        f"USER: <image>\n{q}\nASSISTANT: {a}"
                        for q, a in zip(questions, answers)
                    ]
                    tokenizer = model.mllm.processor.tokenizer
                    text_targets = prepare_text_targets(full_texts, tokenizer).to(device)

                    loss_dict = criterion(
                        z_mask=pred_masks,
                        y_mask=masks,
                        z_txt=text_logits,
                        y_txt=text_targets,
                    )

                    loss_total = loss_dict["loss_total"] / args.gradient_accumulation_steps

                scaler.scale(loss_total).backward()
            else:
                outputs = model(
                    images,
                    questions,
                    answers,
                    training_text=True,
                )
                pred_masks = outputs["z_mask"]
                text_logits = outputs["z_txt_logits"]

                full_texts = [
                    f"USER: <image>\n{q}\nASSISTANT: {a}"
                    for q, a in zip(questions, answers)
                ]
                tokenizer = model.mllm.processor.tokenizer
                text_targets = prepare_text_targets(full_texts, tokenizer).to(device)

                loss_dict = criterion(
                    z_mask=pred_masks,
                    y_mask=masks,
                    z_txt=text_logits,
                    y_txt=text_targets,
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
                print(
                    f"Epoch {epoch+1}, Batch {batch_idx}, "
                    f"Total: {loss_total.item() * args.gradient_accumulation_steps:.4f}, "
                    f"Seg: {loss_dict['loss_seg'].item():.4f}, "
                    f"Text: {loss_dict['loss_txt'].item():.4f}"
                )

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

        avg_train_total = epoch_loss_total / len(train_loader)
        avg_train_seg = epoch_loss_seg / len(train_loader)
        avg_train_txt = epoch_loss_txt / len(train_loader)
        epoch_time = time.time() - epoch_start

        print(
            f"Epoch {epoch+1} - TRAIN - "
            f"Total: {avg_train_total:.4f}, "
            f"Seg: {avg_train_seg:.4f}, "
            f"Text: {avg_train_txt:.4f}, "
            f"Time: {epoch_time:.2f}s"
        )

        # ------------------------- Validation ------------------------- #
        model.eval()
        val_total = 0.0
        val_seg = 0.0
        val_txt = 0.0

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device, non_blocking=True)
                masks = batch["mask"].to(device, non_blocking=True)
                questions = batch["question"]
                answers = batch["answer"]

                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        outputs = model(
                            images,
                            questions,
                            answers,
                            training_text=True,
                        )
                        pred_masks = outputs["z_mask"]
                        text_logits = outputs["z_txt_logits"]

                        full_texts = [
                            f"USER: <image>\n{q}\nASSISTANT: {a}"
                            for q, a in zip(questions, answers)
                        ]
                        tokenizer = model.mllm.processor.tokenizer
                        text_targets = prepare_text_targets(full_texts, tokenizer).to(device)

                        loss_dict = criterion(
                            z_mask=pred_masks,
                            y_mask=masks,
                            z_txt=text_logits,
                            y_txt=text_targets,
                        )
                else:
                    outputs = model(
                        images,
                        questions,
                        answers,
                        training_text=True,
                    )
                    pred_masks = outputs["z_mask"]
                    text_logits = outputs["z_txt_logits"]

                    full_texts = [
                        f"USER: <image>\n{q}\nASSISTANT: {a}"
                        for q, a in zip(questions, answers)
                    ]
                    tokenizer = model.mllm.processor.tokenizer
                    text_targets = prepare_text_targets(full_texts, tokenizer).to(device)

                    loss_dict = criterion(
                        z_mask=pred_masks,
                        y_mask=masks,
                        z_txt=text_logits,
                        y_txt=text_targets,
                    )

                val_total += loss_dict["loss_total"].item()
                val_seg += loss_dict["loss_seg"].item()
                val_txt += loss_dict["loss_txt"].item()

        avg_val_total = val_total / len(val_loader)
        avg_val_seg = val_seg / len(val_loader)
        avg_val_txt = val_txt / len(val_loader)

        print(
            f"Epoch {epoch+1} - VALIDATION - "
            f"Total: {avg_val_total:.4f}, "
            f"Seg: {avg_val_seg:.4f}, "
            f"Text: {avg_val_txt:.4f}"
        )

        # Checkpoints
        if checkpoint_dir is not None:
            if avg_val_total < best_val_loss:
                best_val_loss = avg_val_total
                ok = save_checkpoint(epoch, model, optimizer, checkpoint_dir, is_best=True)
                if ok:
                    print(f"✓ New best model saved (val_total={avg_val_total:.4f})")
            if (epoch + 1) % 5 == 0 or epoch == 0:
                ok = save_checkpoint(epoch, model, optimizer, checkpoint_dir, is_best=False)
                if ok:
                    print(f"✓ Periodic checkpoint saved at epoch {epoch+1}")
        else:
            print("WARNING: checkpoint_dir is None, not saving checkpoints")

    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    if checkpoint_dir is not None:
        print(f"Checkpoints in: {checkpoint_dir}")
        files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth")]
        if files:
            print(f"Saved {len(files)} checkpoint(s):")
            for f in sorted(files):
                size = os.path.getsize(os.path.join(checkpoint_dir, f)) / (1024 * 1024)
                print(f"  - {f} ({size:.2f} MB)")
        else:
            print("WARNING: No checkpoint files found")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n" + "=" * 60)
        print("ERROR: Training failed with exception:")
        print("=" * 60)
        print(f"Type: {type(e).__name__}")
        print(f"Message: {str(e)}")
        print("\nTraceback:")
        traceback.print_exc()
        print("=" * 60 + "\n")
        sys.exit(1)
