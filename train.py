import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast
from tqdm import tqdm
import logging
import torch.nn.functional as F
from pathlib import Path

from models.llm_seg_original import build_llm_seg
from data.dataset import PromptSegmentDataset, collate_fn
from models.loss.original_loss import structure_loss, dice_score, BceDiceLoss
from llava.utils import disable_torch_init


def count_train_parameters(model):
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    num_params = sum([p.numel() for p in trainable_params])
    print(f"Number of trainable parameters: {num_params / 1e6:.2f}M")
    return num_params


def evaluate(model, val_loader, device="cuda:0"):
    dice_score_list = []
    print(f"Number of val samples: {len(val_loader)}")
    for batch in tqdm(val_loader, desc="Evaluating"):
        model.eval()
        model.to(device)
        input_ids = batch['input_ids'].to(device)
        image_tensor = batch['image_tensor'].to(device)
        mask_tensor = batch['mask_tensor'].to(device)
        image_sam_tensor = batch['image_sam'].to(device)
        attention_mask = batch['attention_masks'].to(device)
        answers_ids = batch['answers_ids'].to(device)
        
        with torch.no_grad():
            outputs, _ = model(
                input_ids=input_ids,
                image_tensor_for_vlm=image_tensor,
                image_tensor_for_image_enc=image_sam_tensor,
                attention_mask=attention_mask,
                answers=answers_ids
            )
            dice_score_value = dice_score(outputs, mask_tensor)
            dice_score_list.append(dice_score_value.item())
    
    mean_dice = sum(dice_score_list) / len(dice_score_list) if dice_score_list else 0.0
    return mean_dice


def train(
    model,
    full_loader,
    optimizer,
    num_epochs=10,
    device="cuda:0",
    save_dir="./checkpoints",
    cls_loss_weight=0.5,
    cls_loss_epochs=5
):
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=1e-6
    )

    bce_dice_loss = BceDiceLoss()
    os.makedirs(save_dir, exist_ok=True)

    dataloader = full_loader["train"]
    val_dataloader = full_loader["val"]
    
    best_dice = 0.0
    
    for epoch in range(num_epochs):
        model.train()
        model.to(device)
        ep_loss = 0
        total_llm_loss = 0
        total_segment_loss = 0
        total_cls_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for batch in progress_bar:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            image_tensor = batch['image_tensor'].to(device)
            mask_tensor = batch['mask_tensor'].to(device)
            image_sam_tensor = batch['image_sam'].to(device)
            attention_mask = batch['attention_masks'].to(device)
            answers_ids = batch['answers_ids'].to(device)
            labels = batch['label'].to(device)
            
            with autocast(dtype=torch.bfloat16, device_type=device.split(':')[0]):
                outputs_mask, output_cls, logit_loss = model(
                    input_ids=input_ids,
                    image_tensor_for_vlm=image_tensor,
                    image_tensor_for_image_enc=image_sam_tensor,
                    attention_mask=attention_mask,
                    answers=answers_ids
                )
            
            outputs_mask = F.interpolate(outputs_mask, size=(1024, 1024), mode='bilinear', align_corners=False)
            cls_loss = nn.CrossEntropyLoss()(output_cls, labels)
            segment_loss = structure_loss(outputs_mask, mask_tensor)
            
            if epoch < cls_loss_epochs:
                loss = segment_loss + cls_loss_weight * cls_loss + logit_loss
            else:
                loss = segment_loss + logit_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            ep_loss += loss.item()
            avg_loss = ep_loss / (progress_bar.n + 1)
            total_llm_loss += logit_loss.item()
            total_segment_loss += segment_loss.item()
            total_cls_loss += cls_loss.item()
            avg_llm_loss = total_llm_loss / (progress_bar.n + 1)
            avg_segment_loss = total_segment_loss / (progress_bar.n + 1)
            avg_cls_loss = total_cls_loss / (progress_bar.n + 1)
            
            if progress_bar.n % 100 == 0:
                logging.info(
                    f"Epoch [{epoch+1}/{num_epochs}], Step [{progress_bar.n}], "
                    f"Loss: {avg_loss:.4f}, LLM Loss: {avg_llm_loss:.4f}, "
                    f"Segment Loss: {avg_segment_loss:.4f}, Cls Loss: {avg_cls_loss:.4f}"
                )
            progress_bar.set_postfix(
                loss=f"{avg_loss:.4f}",
                llm_loss=f"{avg_llm_loss:.4f}",
                seg_loss=f"{avg_segment_loss:.4f}",
                cls_loss=f"{avg_cls_loss:.4f}"
            )
        
        scheduler.step()
        model.eval()
        ep_loss /= len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {ep_loss:.4f}")
        logging.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {ep_loss:.4f}")
        
        mean_dice = evaluate(model, val_dataloader, device=device)
        print(f"Epoch [{epoch+1}/{num_epochs}], Val mean Dice Score: {mean_dice:.4f}")
        logging.info(f"Epoch [{epoch+1}/{num_epochs}], Val mean Dice Score: {mean_dice:.4f}")
        
        # Save checkpoint
        checkpoint_dir = Path(save_dir) / f"llm_seg_{epoch+1}"
        model.save_model(str(checkpoint_dir))
        
        # Save best model
        if mean_dice > best_dice:
            best_dice = mean_dice
            best_checkpoint_dir = Path(save_dir) / "best_model"
            model.save_model(str(best_checkpoint_dir))
            print(f"Saved best model with Dice: {best_dice:.4f}")


def create_dataloader(
    data_path,
    annotation_paths,
    data_config,
    image_processor,
    tokenizer,
    batch_size=2,
    mode="train",
    num_workers=4
):
    """Create dataloaders from annotation paths (can be single path or comma-separated string)"""
    if isinstance(annotation_paths, str):
        annotation_paths = [p.strip() for p in annotation_paths.split(',')]
    
    train_datasets = []
    val_datasets = []
    
    for ann_path in annotation_paths:
        train_dataset = PromptSegmentDataset(
            data_path=data_path,
            annotation_path=ann_path,
            data_config=data_config,
            image_processor=image_processor,
            tokenizer=tokenizer,
            mode="train"
        )
        val_dataset = PromptSegmentDataset(
            data_path=data_path,
            annotation_path=ann_path,
            data_config=data_config,
            image_processor=image_processor,
            tokenizer=tokenizer,
            mode="val"
        )
        train_datasets.append(train_dataset)
        val_datasets.append(val_dataset)
    
    if len(train_datasets) == 1:
        train_dataset = train_datasets[0]
        val_dataset = val_datasets[0]
    else:
        train_dataset = torch.utils.data.ConcatDataset(train_datasets)
        val_dataset = torch.utils.data.ConcatDataset(val_datasets)
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return {
        "train": train_dataloader,
        "val": val_dataloader
    }


def parse_args():
    parser = argparse.ArgumentParser(description='PRS-Med Training (Original Implementation)')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory containing data (e.g., data_v2/)')
    parser.add_argument('--ann_paths', type=str, required=True,
                        help='Comma-separated paths to annotation CSV files')
    parser.add_argument('--vlm_path', type=str, required=True,
                        help='Path to LLaVA-Med model (local path or Hugging Face ID)')
    parser.add_argument('--sam_ckpt', type=str, required=True,
                        help='Path to vision encoder checkpoint (TinySAM or SAM-Med2D)')
    parser.add_argument('--sam_model_type', type=str, default='vit_t',
                        help='TinySAM model type (default: vit_t, ignored for SAM-Med2D)')
    parser.add_argument('--encoder_type', type=str, default='tinysam',
                        choices=['tinysam', 'sam_med2d', 'sammed2d'],
                        help='Vision encoder type: tinysam or sam_med2d (default: tinysam)')
    parser.add_argument('--image_size', type=int, default=1024,
                        help='Input image size (default: 1024)')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size (default: 4)')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs (default: 20)')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay (default: 1e-5)')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use (default: cuda:0)')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints (default: ./checkpoints)')
    parser.add_argument('--load_8bit', action='store_true',
                        help='Load model in 8-bit')
    parser.add_argument('--load_4bit', action='store_true',
                        help='Load model in 4-bit')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loader workers (default: 4)')
    parser.add_argument('--cls_loss_weight', type=float, default=0.5,
                        help='Classification loss weight (default: 0.5)')
    parser.add_argument('--cls_loss_epochs', type=int, default=5,
                        help='Number of epochs to use classification loss (default: 5)')
    parser.add_argument('--log_file', type=str, default='logs/training.log',
                        help='Log file path (default: logs/training.log)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup logging
    os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
    logging.basicConfig(
        filename=args.log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logging.info("Starting PRS-Med training")
    logging.info(f"Arguments: {args}")
    
    # Build model
    disable_torch_init()
    print(f"Building model with VLM: {args.vlm_path}")
    print(f"Vision encoder: {args.encoder_type} from {args.sam_ckpt}")
    
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
    
    # Create dataloaders
    print(f"Loading data from: {args.data_root}")
    print(f"Annotation paths: {args.ann_paths}")
    
    dataloader = create_dataloader(
        data_path=args.data_root,
        annotation_paths=args.ann_paths,
        data_config=config,
        image_processor=image_processor,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        mode="train",
        num_workers=args.num_workers
    )
    
    # Setup optimizer
    model.to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        eps=1e-6
    )
    
    # Count trainable parameters
    train_params = count_train_parameters(model)
    print(f"Trainable parameters: {train_params / 1e6:.2f}M")
    logging.info(f"Trainable parameters: {train_params / 1e6:.2f}M")
    
    # Train
    train(
        model=model,
        full_loader=dataloader,
        optimizer=optimizer,
        num_epochs=args.epochs,
        device=args.device,
        save_dir=args.save_dir,
        cls_loss_weight=args.cls_loss_weight,
        cls_loss_epochs=args.cls_loss_epochs
    )
    
    print("Training completed!")
    logging.info("Training completed!")


if __name__ == "__main__":
    main()
