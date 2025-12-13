import os
import sys
import argparse
import torch 
import torch.nn as nn 
from torch.utils.data import DataLoader
from torch.amp import autocast
from tqdm import tqdm
import logging
import torch.nn.functional as F

# Add parent PRS-Med repo to path
parent_repo = os.path.join(os.path.dirname(__file__), '../PRS-Med')
if os.path.exists(parent_repo):
    sys.path.insert(0, parent_repo)

from models.llm_seg_original import build_llm_seg
from data.dataset_original import create_dataloader
from models.loss.original_loss import structure_loss, dice_score, BceDiceLoss

# Setup logging - will be configured in main() with proper log file path
logger = logging.getLogger(__name__)

def count_train_parameters(model):
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    num_params = sum([p.numel() for p in trainable_params])
    print(f"Number of trainable parameters: {num_params / 1e6:.2f}M")
    return num_params

def evaluate(model, val_loader, device="cuda:0"): 
    dice_score_list = []
    print("Number of val sample", len(val_loader))
    for batch in tqdm(val_loader, desc="Evaluating"):
        model.eval()
        model.to(device)
        input_ids = batch['input_ids'].to(device)
        image_tensor = batch['image_tensor'].to(device)
        mask_tensor = batch['mask_tensor'].to(device)
        image_sam_tensor = batch['image_sam'].to(device)
        with torch.no_grad():
            outputs, _, _ = model(
                input_ids=input_ids,
                image_tensor_for_vlm=image_tensor,
                image_tensor_for_image_enc=image_sam_tensor,
                attention_mask=batch['attention_masks'].to(device),
                answers=batch['answers_ids'].to(device)
            )
            dice_score_value = dice_score(outputs, mask_tensor)
            dice_score_list.append(dice_score_value.item())
    mean_dice = sum(dice_score_list) / len(dice_score_list)
    return mean_dice
        
def train(
    model,
    full_loader,
    optimizer,
    num_epochs=10,
    device="cuda:0",
    save_dir="./checkpoints"
):
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=20,
        eta_min=1e-6
    )

    bce_dice_loss = BceDiceLoss()

    dataloader = full_loader["train"]
    val_dataloader = full_loader["val"]
    
    os.makedirs(save_dir, exist_ok=True)
    
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
            
            with autocast(dtype=torch.bfloat16, device_type=device):
                outputs_mask, output_cls, logit_loss = model(
                    input_ids = input_ids, 
                    image_tensor_for_vlm = image_tensor, 
                    image_tensor_for_image_enc = image_sam_tensor, 
                    attention_mask = attention_mask,
                    answers = answers_ids)
            
            outputs_mask = F.interpolate(outputs_mask, size=(1024, 1024), mode='bilinear', align_corners=False)
            cls_loss = nn.CrossEntropyLoss()(output_cls, labels)
            segment_loss = structure_loss(outputs_mask, mask_tensor)
            
            if epoch < 5:
                loss = segment_loss + 0.5 * cls_loss + logit_loss 
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
            
            # Log every batch to file
            batch_num = progress_bar.n + 1
            global_step = epoch * len(dataloader) + batch_num
            logging.info(
                f"Epoch {epoch+1}/{num_epochs} | Batch {batch_num} | Step {global_step} | "
                f"Total Loss: {loss.item():.6f} | "
                f"Seg Loss: {segment_loss.item():.6f} | "
                f"LM Loss: {logit_loss.item():.6f} | "
                f"CLS Loss: {cls_loss.item():.6f} | "
                f"Avg Total: {avg_loss:.6f} | "
                f"Avg Seg: {avg_segment_loss:.6f} | "
                f"Avg LM: {avg_llm_loss:.6f} | "
                f"Avg CLS: {avg_cls_loss:.6f}"
            )
            
            progress_bar.set_postfix(loss=avg_loss, llm_loss=avg_llm_loss, segment_loss=avg_segment_loss, cls_loss=avg_cls_loss)
        
        scheduler.step()
        model.eval()
        ep_loss /= len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {ep_loss}")
        logging.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {ep_loss}")
        model.eval()
        mean_dice = evaluate(model, val_dataloader, device=device)
        print(f"Epoch [{epoch+1}/{num_epochs}], Val mean Dice Score: {mean_dice}")
        logging.info(f"Epoch [{epoch+1}/{num_epochs}], Val mean Dice Score: {mean_dice}")
        
        # Save checkpoint
        checkpoint_path = os.path.join(save_dir, f"llm_seg_{epoch+1}")
        model.save_model(checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")
        logging.info(f"Checkpoint saved to: {checkpoint_path}")


def parse_args():
    parser = argparse.ArgumentParser(description='PRS-Med Training (Original Implementation)')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory containing data (e.g., data_v2/)')
    parser.add_argument('--ann_paths', type=str, required=True,
                        help='Comma-separated paths to annotation CSV files')
    parser.add_argument('--vlm_path', type=str, required=True,
                        help='Path to LLaVA-Med model (local path or Hugging Face ID like "microsoft/llava-med-v1.5-mistral-7b")')
    parser.add_argument('--sam_ckpt', type=str, required=True,
                        help='Path to TinySAM checkpoint')
    parser.add_argument('--sam_model_type', type=str, default='vit_t',
                        help='TinySAM model type (default: vit_t)')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--load_8bit', action='store_true',
                        help='Load model in 8-bit')
    parser.add_argument('--load_4bit', action='store_true',
                        help='Load model in 4-bit')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    os.makedirs('logs', exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Setup logging with batch loss file
    log_file = os.path.join(args.save_dir, 'batch_losses.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ],
        force=True  # Override any existing config
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Training started. Batch losses will be logged to: {log_file}")
    logger.info(f"Arguments: {vars(args)}")
    
    device = args.device
    
    # Build model
    model, tokenizer, image_processor, config = build_llm_seg(
        model_path=args.vlm_path,
        model_base=None,
        load_8bit=args.load_8bit,
        load_4bit=args.load_4bit,
        device=device,
        sam_model_type=args.sam_model_type,
        sam_checkpoint_path=args.sam_ckpt
    )

    # Parse annotation paths
    ann_paths = [p.strip() for p in args.ann_paths.split(',')]
    
    # Create dataloader
    dataloader = create_dataloader(
        data_path=args.data_root,
        annotation_path=ann_paths,
        data_config=config,
        image_processor=image_processor,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        mode="train"
    )

    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-5,
        eps=1e-6
    )

    train_params = count_train_parameters(model)
    print("Trainable parameters:", train_params)
    
    train(
        model=model,
        full_loader=dataloader,
        optimizer=optimizer,
        num_epochs=args.epochs,
        device=device,
        save_dir=args.save_dir
    )

