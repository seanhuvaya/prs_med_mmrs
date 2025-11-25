"""Training loop and validation logic for PRS-Med (single GPU/CPU only)."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
from typing import Dict, Optional, Callable

from utils.checkpoint import save_checkpoint
from utils.text import prepare_text_targets


class PRSMedTrainer:
    """Trainer class for PRS-Med model."""
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        args,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
        callbacks: Optional[list] = None
    ):
        """
        Initialize trainer.
        
        Args:
            model: PRS-Med model
            optimizer: Optimizer
            criterion: Loss function
            device: Device to train on
            args: Training arguments
            scaler: Optional gradient scaler for mixed precision
            callbacks: Optional list of callback functions
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.args = args
        self.scaler = scaler
        self.callbacks = callbacks or []
        
        # Model reference for saving
        self.model_for_saving = model
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch number
        
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        self.current_epoch = epoch
        
        epoch_loss_total = 0.0
        epoch_loss_seg = 0.0
        epoch_loss_txt = 0.0
        epoch_start_time = time.time()
        
        # Initialize gradients at the start of epoch
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(train_loader):
            # Move data to device
            images = batch['image'].to(self.device, non_blocking=True)
            masks = batch['mask'].to(self.device, non_blocking=True)
            questions = batch['question']
            answers = batch['answer']
            
            # Forward pass with mixed precision
            if self.scaler is not None:
                # Mixed precision forward pass
                with torch.cuda.amp.autocast():
                    outputs = self.model(images, questions)
                    pred_masks = outputs["z_mask"]
                    text_logits = outputs["z_txt_logits"]
                    
                    # Prepare text targets - access tokenizer from the underlying model
                    tokenizer = self.model_for_saving.mllm.processor.tokenizer
                    text_targets = prepare_text_targets(answers, tokenizer)
                    text_targets = text_targets.to(self.device)
                    
                    # Calculate loss
                    loss_dict = self.criterion(
                        z_mask=pred_masks,
                        y_mask=masks,
                        z_txt=text_logits,  # Use text logits, not embeddings
                        y_txt=text_targets
                    )
                    
                    loss_total = loss_dict["loss_total"]
                    # Scale loss for gradient accumulation
                    loss_total = loss_total / self.args.gradient_accumulation_steps
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss_total).backward()
            else:
                # Standard precision forward pass
                outputs = self.model(images, questions)
                pred_masks = outputs["z_mask"]
                text_logits = outputs["z_txt_logits"]
                
                # Prepare text targets - access tokenizer from the underlying model
                tokenizer = self.model_for_saving.mllm.processor.tokenizer
                text_targets = prepare_text_targets(answers, tokenizer)
                text_targets = text_targets.to(self.device)
                
                # Calculate loss
                loss_dict = self.criterion(
                    z_mask=pred_masks,
                    y_mask=masks,
                    z_txt=text_logits,  # Use text logits, not embeddings
                    y_txt=text_targets
                )
                
                loss_total = loss_dict["loss_total"]
                # Scale loss for gradient accumulation
                loss_total = loss_total / self.args.gradient_accumulation_steps
                
                # Backward pass
                loss_total.backward()
            
            # Gradient accumulation: only update weights every N steps
            if (batch_idx + 1) % self.args.gradient_accumulation_steps == 0:
                if self.scaler is not None:
                    # Unscale gradients before clipping
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad],
                        max_norm=1.0
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad],
                        max_norm=1.0
                    )
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
            
            # Accumulate losses (multiply by accumulation steps to get true loss)
            epoch_loss_total += loss_total.item() * self.args.gradient_accumulation_steps
            epoch_loss_seg += loss_dict["loss_seg"].item() * self.args.gradient_accumulation_steps
            epoch_loss_txt += loss_dict["loss_txt"].item() * self.args.gradient_accumulation_steps
            
            if batch_idx % 10 == 0:
                # Clear cache periodically to free memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                print(f'Epoch {epoch+1}, Batch {batch_idx}, '
                      f'Total Loss: {loss_total.item() * self.args.gradient_accumulation_steps:.4f}, '
                      f'Seg Loss: {loss_dict["loss_seg"].item():.4f}, '
                      f'Text Loss: {loss_dict["loss_txt"].item():.4f}')
        
        # Handle remaining gradients if batch doesn't divide evenly
        if len(train_loader) % self.args.gradient_accumulation_steps != 0:
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    max_norm=1.0
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    max_norm=1.0
                )
                self.optimizer.step()
            self.optimizer.zero_grad()
        
        # Calculate epoch averages
        avg_train_loss_total = epoch_loss_total / len(train_loader)
        avg_train_loss_seg = epoch_loss_seg / len(train_loader)
        avg_train_loss_txt = epoch_loss_txt / len(train_loader)
        epoch_time = time.time() - epoch_start_time
        
        return {
            'loss_total': avg_train_loss_total,
            'loss_seg': avg_train_loss_seg,
            'loss_txt': avg_train_loss_txt,
            'epoch_time': epoch_time
        }
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate model on validation set.
        
        Args:
            val_loader: Validation data loader
        
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        val_loss_total = 0.0
        val_loss_seg = 0.0
        val_loss_txt = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(self.device, non_blocking=True)
                masks = batch['mask'].to(self.device, non_blocking=True)
                questions = batch['question']
                answers = batch['answer']
                
                # Use mixed precision for validation if enabled
                if self.scaler is not None:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(images, questions)
                        pred_masks = outputs["z_mask"]
                        text_logits = outputs["z_txt_logits"]
                        
                        # Access tokenizer from the underlying model
                        tokenizer = self.model_for_saving.mllm.processor.tokenizer
                        text_targets = prepare_text_targets(answers, tokenizer)
                        text_targets = text_targets.to(self.device)
                        
                        loss_dict = self.criterion(
                            z_mask=pred_masks,
                            y_mask=masks,
                            z_txt=text_logits,
                            y_txt=text_targets
                        )
                else:
                    outputs = self.model(images, questions)
                    pred_masks = outputs["z_mask"]
                    text_logits = outputs["z_txt_logits"]
                    
                    # Access tokenizer from the underlying model
                    tokenizer = self.model_for_saving.mllm.processor.tokenizer
                    text_targets = prepare_text_targets(answers, tokenizer)
                    text_targets = text_targets.to(self.device)
                    
                    loss_dict = self.criterion(
                        z_mask=pred_masks,
                        y_mask=masks,
                        z_txt=text_logits,
                        y_txt=text_targets
                    )
                
                val_loss_total += loss_dict["loss_total"].item()
                val_loss_seg += loss_dict["loss_seg"].item()
                val_loss_txt += loss_dict["loss_txt"].item()
        
        total_val_batches = len(val_loader)
        
        # Calculate validation averages (across all GPUs if distributed)
        avg_val_loss_total = val_loss_total / total_val_batches
        avg_val_loss_seg = val_loss_seg / total_val_batches
        avg_val_loss_txt = val_loss_txt / total_val_batches
        
        return {
            'loss_total': avg_val_loss_total,
            'loss_seg': avg_val_loss_seg,
            'loss_txt': avg_val_loss_txt
        }
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        checkpoint_dir: Optional[str] = None
    ):
        """
        Train model for multiple epochs.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs to train
            checkpoint_dir: Directory to save checkpoints
        """
        print("Starting training...")
        print(f"Learning rate: {self.args.learning_rate}")
        print(f"Gradient accumulation steps: {self.args.gradient_accumulation_steps}")
        
        for epoch in range(num_epochs):
            # Set epoch for sampler if supported
            if hasattr(train_loader, 'sampler') and hasattr(train_loader.sampler, 'set_epoch'):
                train_loader.sampler.set_epoch(epoch)
            
            # Training phase
            train_metrics = self.train_epoch(train_loader, epoch)
            
            print(f'Epoch {epoch+1} - TRAIN - '
                  f'Total Loss: {train_metrics["loss_total"]:.4f}, '
                  f'Seg Loss: {train_metrics["loss_seg"]:.4f}, '
                  f'Text Loss: {train_metrics["loss_txt"]:.4f}, '
                  f'Time: {train_metrics["epoch_time"]:.2f}s')
            
            # Validation phase
            val_metrics = self.validate(val_loader)
            
            print(f'Epoch {epoch+1} - VALIDATION - '
                  f'Total Loss: {val_metrics["loss_total"]:.4f}, '
                  f'Seg Loss: {val_metrics["loss_seg"]:.4f}, '
                  f'Text Loss: {val_metrics["loss_txt"]:.4f}')
            
            # Save checkpoints
            if checkpoint_dir is not None:
                # Save best model if validation improved
                if val_metrics["loss_total"] < self.best_val_loss:
                    self.best_val_loss = val_metrics["loss_total"]
                    success = save_checkpoint(
                        epoch,
                        self.model,
                        self.optimizer,
                        checkpoint_dir,
                        is_best=True
                    )
                    if success:
                        print(f"✓ New best model saved with val_loss: {val_metrics['loss_total']:.4f}")
                
                # Save periodic checkpoints every 5 epochs
                if (epoch + 1) % 5 == 0:
                    success = save_checkpoint(
                        epoch,
                        self.model,
                        self.optimizer,
                        checkpoint_dir
                    )
                    if success:
                        print(f"✓ Periodic checkpoint saved at epoch {epoch+1}")
                
                # Save checkpoint at end of epoch 1 (for debugging/verification)
                if epoch == 0:
                    success = save_checkpoint(
                        epoch,
                        self.model,
                        self.optimizer,
                        checkpoint_dir
                    )
                    if success:
                        print(f"✓ Initial checkpoint saved at epoch {epoch+1}")
        
        # Save final checkpoint
        if checkpoint_dir is not None:
            print("Saving final checkpoint...")
            save_checkpoint(
                num_epochs - 1,
                self.model,
                self.optimizer,
                checkpoint_dir
            )
        
        print("Training completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")

