"""Main training script for PRS-Med model."""

import os
import sys
import torch
import torch.optim as optim

from datetime import datetime

from utils.config import get_config
from data.dataset import PRSMedDataLoader
from models import PRSMedModel
from models.loss.objective_function import PRSMedLoss
from utils import set_seed, logging
from training import PRSMedTrainer

logger = logging.get_logger(__name__)


def main():
    """Main training function."""
    args = get_config()

    # Set tokenizers parallelism to avoid warnings when forking
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    # Choose device (single GPU/CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Set random seed for reproducibility
    set_seed(args.seed)

    # Set default tensor type to float32
    torch.set_default_dtype(torch.float32)

    # Enable deterministic algorithms if requested
    if args.deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        logger.info("Deterministic algorithms enabled")

    # Create checkpoint directory
    os.makedirs(args.ckpt_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_dir = os.path.join(args.ckpt_dir, f'training_{timestamp}')
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger.info(f"Checkpoint directory created: {checkpoint_dir}")

    # Initialize data loaders
    logger.info(f"Loading data from {args.data_dir}...")
    train_loader, val_loader = PRSMedDataLoader.get_training_dataloaders(batch_size=args.batch_size,
                                                                         num_workers=args.num_workers,
                                                                         data_root=args.data_dir)

    # Initialize complete PRS-Med model
    model = PRSMedModel(args, device)
    model.to(device)

    # Setup optimizer and loss
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    # Use AdamW optimizer with weight decay (from paper)
    optimizer = optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        weight_decay=0.01,  # From paper hyperparameters
        betas=(0.9, 0.999)
    )
    criterion = PRSMedLoss(lambda_seg=args.lambda_seg, lambda_txt=args.lambda_txt)

    # Setup mixed precision training (AMP)
    scaler = None
    if args.amp and torch.cuda.is_available():
        scaler = torch.cuda.amp.GradScaler()
        logger.info("Mixed precision training (AMP) enabled")

    logger.info("Starting training...")

    # Create trainer
    trainer = PRSMedTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        args=args,
        scaler=scaler
    )

    # Train model
    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        checkpoint_dir=checkpoint_dir
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Training failed with exception: {str(e)}")
        sys.exit(1)
