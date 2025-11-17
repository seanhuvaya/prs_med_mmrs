import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import time
from datetime import datetime, timedelta
import random
import numpy as np
import traceback
import sys

# Import all components including vision backbone
from data.dataset import PRSMedDataLoader
from models.vision_backbone.tiny_sam_encoder import TinySAMVisionBackbone  # Add this
from models.mllm.llava_med_lora_adapter import LLavaMedWithLoRA
from models.decoder.fusion_module import PromptMaskFusionModule
from models.decoder.mask_prediction_module import MaskPredictionModule
from models.loss.objective_function import PRSMedLoss

def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value (default: 42)
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
    parser = argparse.ArgumentParser(description='PRS-Med Training')
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
                       help='Batch size per GPU (total batch size = batch_size * num_gpus)')
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--image_size', type=int, default=1024)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--deterministic', action='store_true', 
                       help='Enable fully deterministic training (may be slower)')
    # Distributed training arguments
    parser.add_argument('--local_rank', type=int, default=-1,
                       help='Local rank for distributed training (set automatically by torchrun)')
    parser.add_argument('--world_size', type=int, default=-1,
                       help='Number of GPUs (set automatically by torchrun)')
    parser.add_argument('--dist_url', type=str, default='env://',
                       help='URL used to set up distributed training')
    # Memory optimization arguments
    parser.add_argument('--use_amp', action='store_true', default=False,
                       help='Use Automatic Mixed Precision (AMP) to reduce memory usage')
    parser.add_argument('--no-use_amp', dest='use_amp', action='store_false',
                       help='Disable Automatic Mixed Precision (AMP)')
    # Set default to True if neither flag is specified
    parser.set_defaults(use_amp=True)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                       help='Number of gradient accumulation steps (effective batch size = batch_size * gradient_accumulation_steps)')
    parser.add_argument('--gradient_checkpointing', action='store_true', default=False,
                       help='Use gradient checkpointing to trade compute for memory')
    parser.add_argument('--compile_model', action='store_true', default=False,
                       help='Compile model with torch.compile for better memory efficiency (PyTorch 2.0+)')
    return parser.parse_args()

class PRSMedModel(nn.Module):
    """
    Complete PRS-Med model with explicit dtype handling
    """
    def __init__(self, args, device):
        super().__init__()
        self.device = device
        self.image_size = args.image_size
        
        # Vision backbone (TinySAM)
        # Note: We'll move it to device after creation to ensure all submodules are on the same device
        self.vision_backbone = TinySAMVisionBackbone(
            checkpoint_path=args.tinysam_checkpoint,
            image_size=args.image_size,
            device=str(device)
        )
        # Explicitly move to device to ensure all submodules are on the correct device
        self.vision_backbone = self.vision_backbone.to(device)
        
        # Multimodal LLM with LoRA
        self.mllm = LLavaMedWithLoRA(
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            freeze_llm=True,
            device=str(device)
        )
        # Explicitly move to device to ensure all submodules are on the correct device
        self.mllm = self.mllm.to(device)
        
        # Fusion and mask modules - explicitly set to float32
        self.fusion_module = PromptMaskFusionModule().to(device).float()
        self.mask_predictor = MaskPredictionModule().to(device).float()
        
    def preprocess_images(self, images):
        """
        Preprocess images and ensure float32
        """
        # If images are already tensors, ensure they're the right size
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
        
        # Ensure float32
        return images.float()
        
    def forward(self, images, text_prompts):
        """
        Forward pass with explicit dtype conversion and device checking
        """
        # Ensure input images are on the correct device
        if isinstance(images, torch.Tensor) and images.device != self.device:
            images = images.to(self.device)
        
        # Preprocess images and ensure float32
        processed_images = self.preprocess_images(images)
        
        # Ensure processed images are on the correct device
        if processed_images.device != self.device:
            processed_images = processed_images.to(self.device)
        
        # 1. Extract visual features using TinySAM (ensure float32)
        z_image = self.vision_backbone(processed_images).float()  # (B, 256, 16, 16)
        if z_image.device != self.device:
            z_image = z_image.to(self.device)
        
        # 2. Get multimodal embeddings from LLaVA-Med and convert to float32
        mllm_output = self.mllm(processed_images, text_prompts, return_projected=True)
        z_emb = mllm_output["z_emb"].float()      # Convert to float32
        z_txt_logits = mllm_output["z_txt"].float() # Convert to float32
        pred_ids = mllm_output["pred_ids"]
        
        # Ensure all outputs are on the correct device
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
            "z_mask": z_mask,        # Segmentation logits
            "z_txt_logits": z_txt_logits,  # Text generation logits
            "pred_ids": pred_ids,    # Predicted token IDs
        }

def init_distributed(args):
    """
    Initialize distributed training environment.
    Returns True if distributed training is enabled, False otherwise.
    """
    # Check if we're in a distributed environment
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        # Using torchrun or torch.distributed.run
        args.rank = int(os.environ['RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.local_rank = int(os.environ.get('LOCAL_RANK', os.environ.get('RANK', 0)))
    elif 'LOCAL_RANK' in os.environ:
        # Fallback: check LOCAL_RANK directly
        args.local_rank = int(os.environ['LOCAL_RANK'])
        args.rank = int(os.environ.get('RANK', args.local_rank))
        args.world_size = int(os.environ.get('WORLD_SIZE', 1))
    elif args.local_rank != -1:
        # Using torch.distributed.launch (legacy)
        args.rank = int(os.environ.get('RANK', args.local_rank))
        args.world_size = int(os.environ.get('WORLD_SIZE', 1))
        args.local_rank = args.local_rank
    else:
        # Single GPU or CPU
        args.rank = 0
        args.world_size = 1
        args.local_rank = -1
        return False
    
    # Initialize the process group
    try:
        dist.init_process_group(
            backend='nccl' if torch.cuda.is_available() else 'gloo',
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
            timeout=timedelta(seconds=1800)  # 30 minute timeout
        )
    except Exception as e:
        print(f"Error initializing process group: {e}")
        print(f"RANK={args.rank}, WORLD_SIZE={args.world_size}, LOCAL_RANK={args.local_rank}")
        raise
    
    # Set the device for this process
    if torch.cuda.is_available():
        torch.cuda.set_device(args.local_rank)
    
    return True

def cleanup_distributed():
    """Clean up distributed training environment"""
    if dist.is_initialized():
        dist.destroy_process_group()

def is_main_process():
    """Check if this is the main process (rank 0)"""
    return not dist.is_initialized() or dist.get_rank() == 0

def ensure_model_on_device(model, device):
    """
    Ensure all model parameters and buffers are on the specified device.
    This is critical for DDP which requires all parameters on the same device.
    """
    # First, move the entire model to the device (this should move all submodules)
    model = model.to(device)
    
    # Double-check all parameters are on the correct device
    wrong_device_params = []
    for name, param in model.named_parameters():
        if param.device != device:
            wrong_device_params.append((name, param.device))
            # Try to move the parameter
            with torch.no_grad():
                param.data = param.data.to(device)
            # Also move the parameter itself if it's a leaf
            if param.is_leaf:
                param.data = param.data.to(device)
    
    # Also check buffers
    wrong_device_buffers = []
    for name, buffer in model.named_buffers():
        if buffer.device != device:
            wrong_device_buffers.append((name, buffer.device))
            buffer.data = buffer.data.to(device)
    
    # Print warnings if any were found
    if wrong_device_params:
        print(f"WARNING: Found {len(wrong_device_params)} parameters on wrong device:")
        for name, dev in wrong_device_params[:10]:  # Print first 10
            print(f"  {name}: {dev}")
        if len(wrong_device_params) > 10:
            print(f"  ... and {len(wrong_device_params) - 10} more")
    
    if wrong_device_buffers:
        print(f"WARNING: Found {len(wrong_device_buffers)} buffers on wrong device:")
        for name, dev in wrong_device_buffers[:10]:  # Print first 10
            print(f"  {name}: {dev}")
        if len(wrong_device_buffers) > 10:
            print(f"  ... and {len(wrong_device_buffers) - 10} more")
    
    # Final verification - recursively check all submodules
    for name, module in model.named_modules():
        if hasattr(module, 'to'):
            try:
                module = module.to(device)
            except:
                pass
    
    return model

def prepare_text_targets(answers, tokenizer, max_length=512):
    """
    Properly tokenize answers for text generation loss
    """
    # Tokenize answers with the same tokenizer used in LLaVA-Med
    tokenized = tokenizer(
        answers,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )
    
    return tokenized.input_ids

def main():
    args = parse_args()
    
    # Print environment info for debugging (only on main process after init)
    if 'RANK' in os.environ or 'LOCAL_RANK' in os.environ:
        print(f"Distributed environment detected:")
        print(f"  RANK: {os.environ.get('RANK', 'N/A')}")
        print(f"  WORLD_SIZE: {os.environ.get('WORLD_SIZE', 'N/A')}")
        print(f"  LOCAL_RANK: {os.environ.get('LOCAL_RANK', 'N/A')}")
        print(f"  MASTER_ADDR: {os.environ.get('MASTER_ADDR', 'N/A')}")
        print(f"  MASTER_PORT: {os.environ.get('MASTER_PORT', 'N/A')}")
    
    # Initialize distributed training
    is_distributed = init_distributed(args)
    
    # Set device based on distributed setup
    # Note: torch.cuda.set_device is already called in init_distributed
    if is_distributed:
        device = torch.device(f'cuda:{args.local_rank}')
        if is_main_process():
            print(f"Initialized distributed training with {args.world_size} GPUs")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if is_main_process():
        print(f"Using device: {device}")
        if is_distributed:
            print(f"Local rank: {args.local_rank}, Global rank: {args.rank}")
            print(f"Current CUDA device: {torch.cuda.current_device()}")
    
    # Set random seed for reproducibility (must be done before any other operations)
    # Add rank to seed to ensure different random states across processes
    set_seed(args.seed + (args.rank if is_distributed else 0))

    # Set default tensor type to float32
    torch.set_default_dtype(torch.float32)
    
    # Enable deterministic algorithms if requested
    if args.deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if is_main_process():
            print("✓ Deterministic algorithms enabled")
    
    # Create checkpoint directory (only on main process)
    if is_main_process():
        # Ensure base checkpoint directory exists
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        checkpoint_dir = os.path.join(args.checkpoint_dir, f'training_{timestamp}')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        print(f"✓ Checkpoint directory created: {checkpoint_dir}")
        print(f"  Absolute path: {os.path.abspath(checkpoint_dir)}")
        
        # Verify directory exists and is writable
        if os.path.exists(checkpoint_dir) and os.access(checkpoint_dir, os.W_OK):
            print(f"  Directory is writable")
        else:
            print(f"  WARNING: Directory may not be writable!")
            
        # Test write permissions by creating a test file
        try:
            test_file = os.path.join(checkpoint_dir, '.test_write')
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
            print(f"  Write test: SUCCESS")
        except Exception as e:
            print(f"  Write test: FAILED - {e}")
    else:
        checkpoint_dir = None
    
    # Initialize data loaders
    if is_main_process():
        print(f"Loading data from {args.data_root}...")
    
    # Create datasets
    from data.dataset import PRSMedDataset
    train_dataset = PRSMedDataset(split='train', data_root=args.data_root)
    val_dataset = PRSMedDataset(split='val', data_root=args.data_root)
    
    # Create distributed samplers if using distributed training
    if is_distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=args.world_size,
            rank=args.rank,
            shuffle=True,
            seed=args.seed
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=args.world_size,
            rank=args.rank,
            shuffle=False
        )
        shuffle = False  # Shuffle is handled by DistributedSampler
    else:
        train_sampler = None
        val_sampler = None
        shuffle = True
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True  # Drop last incomplete batch for distributed training
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    if is_main_process():
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        print(f"Effective batch size: {args.batch_size * args.world_size if is_distributed else args.batch_size}")
    
    # CRITICAL: Set the default CUDA device before creating the model
    # This ensures all new tensors/modules are created on the correct device
    if is_distributed and torch.cuda.is_available():
        torch.cuda.set_device(args.local_rank)
        # Verify the default device is set correctly
        if torch.cuda.current_device() != args.local_rank:
            raise RuntimeError(f"Failed to set CUDA device to {args.local_rank}. Current device: {torch.cuda.current_device()}")
    
    # Initialize complete PRS-Med model
    model = PRSMedModel(args, device)
    
    # Ensure all model parameters are on the correct device
    # This is critical for DDP - all parameters must be on the same device
    model = ensure_model_on_device(model, device)
    
    # Verify all parameters are on the correct device (for debugging)
    if is_main_process():
        param_devices = set()
        for name, param in model.named_parameters():
            param_devices.add(param.device)
        if len(param_devices) > 1:
            print(f"ERROR: Model parameters are still on multiple devices: {param_devices}")
            print("This should not happen after ensure_model_on_device. Attempting to fix...")
            model = ensure_model_on_device(model, device)
            # Check again
            param_devices = set()
            for name, param in model.named_parameters():
                param_devices.add(param.device)
            if len(param_devices) > 1:
                raise RuntimeError(f"Failed to move all parameters to {device}. Parameters still on: {param_devices}")
        else:
            print(f"✓ All model parameters are on device: {list(param_devices)[0]}")
    
    # Wrap model with DDP if using distributed training
    if is_distributed:
        # Final check: verify all parameters are on the correct device before DDP
        param_devices = set()
        for name, param in model.named_parameters():
            param_devices.add(param.device)
        
        if len(param_devices) > 1:
            print(f"ERROR before DDP: Model parameters are on multiple devices: {param_devices}")
            print("Attempting to fix by moving all parameters to device:", device)
            # Try one more time to move everything
            model = ensure_model_on_device(model, device)
            # Check again
            param_devices = set()
            for name, param in model.named_parameters():
                param_devices.add(param.device)
            if len(param_devices) > 1:
                raise RuntimeError(
                    f"Cannot wrap model with DDP: parameters are on multiple devices {param_devices}. "
                    f"Expected all on {device}. This usually means some model components are creating "
                    f"tensors on the default CUDA device instead of the assigned device."
                )
        
        if is_main_process():
            print(f"✓ All parameters verified on device: {list(param_devices)[0]}")
            print(f"Wrapping model with DDP on device {args.local_rank}...")
        
        # For DDP, we need to ensure the model is on the correct device
        # and pass the device_ids parameter correctly
        # When model is already on device, we can use device_ids=[args.local_rank]
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=False)
        model_for_saving = model.module  # Access underlying model for saving
    else:
        model_for_saving = model
    
    # Enable gradient checkpointing if requested
    if args.gradient_checkpointing:
        if hasattr(model, 'module'):  # DDP wrapped
            # Enable checkpointing for vision backbone
            if hasattr(model.module.vision_backbone.encoder, 'use_checkpoint'):
                # Set use_checkpoint for all layers in TinySAM encoder
                for module in model.module.vision_backbone.encoder.modules():
                    if hasattr(module, 'use_checkpoint'):
                        module.use_checkpoint = True
            # Enable checkpointing for MLLM
            if hasattr(model.module.mllm.model, 'gradient_checkpointing_enable'):
                model.module.mllm.model.gradient_checkpointing_enable()
        else:
            # Enable checkpointing for vision backbone
            if hasattr(model.vision_backbone.encoder, 'use_checkpoint'):
                # Set use_checkpoint for all layers in TinySAM encoder
                for module in model.vision_backbone.encoder.modules():
                    if hasattr(module, 'use_checkpoint'):
                        module.use_checkpoint = True
            # Enable checkpointing for MLLM
            if hasattr(model.mllm.model, 'gradient_checkpointing_enable'):
                model.mllm.model.gradient_checkpointing_enable()
        if is_main_process():
            print("✓ Gradient checkpointing enabled")
    
    # Compile model if requested (PyTorch 2.0+)
    if args.compile_model and hasattr(torch, 'compile'):
        if is_main_process():
            print("Compiling model with torch.compile...")
        model = torch.compile(model, mode='reduce-overhead')
        if is_main_process():
            print("✓ Model compiled")
    
    # Setup optimizer and loss
    trainable_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append(param)
    
    # Use AdamW optimizer with weight decay (from paper)
    # Scale learning rate by world size for distributed training
    effective_lr = args.learning_rate * (args.world_size if is_distributed else 1)
    optimizer = optim.AdamW(
        trainable_params, 
        lr=effective_lr,
        weight_decay=0.01,  # From paper hyperparameters
        betas=(0.9, 0.999)
    )
    criterion = PRSMedLoss(lambda_seg=args.lambda_seg, lambda_txt=args.lambda_txt)
    
    # Setup mixed precision training (AMP)
    scaler = None
    if args.use_amp and torch.cuda.is_available():
        scaler = torch.cuda.amp.GradScaler()
        if is_main_process():
            print("✓ Mixed precision training (AMP) enabled")
    
    # Training parameters
    best_val_loss = float('inf')
    
    if is_main_process():
        print("Starting training...")
        print(f"Learning rate: {effective_lr} (base: {args.learning_rate}, scaled by world_size: {args.world_size if is_distributed else 1})")
        print(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
        print(f"Effective batch size per GPU: {args.batch_size * args.gradient_accumulation_steps}")
        if is_distributed:
            print(f"Total effective batch size: {args.batch_size * args.gradient_accumulation_steps * args.world_size}")
    
    for epoch in range(args.num_epochs):
        # Set epoch for DistributedSampler to ensure proper shuffling
        if is_distributed:
            train_sampler.set_epoch(epoch)
        # Training phase
        model.train()
        epoch_loss_total = 0.0
        epoch_loss_seg = 0.0
        epoch_loss_txt = 0.0
        epoch_start_time = time.time()
        
        # Initialize gradients at the start of epoch
        optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(train_loader):
            # Move data to device
            images = batch['image'].to(device, non_blocking=True)
            masks = batch['mask'].to(device, non_blocking=True)
            questions = batch['question']
            answers = batch['answer']
            
            # Forward pass with mixed precision
            if scaler is not None:
                # Mixed precision forward pass
                with torch.cuda.amp.autocast():
                    outputs = model(images, questions)
                    pred_masks = outputs["z_mask"]
                    text_logits = outputs["z_txt_logits"]
                    
                    # Prepare text targets - access tokenizer from the underlying model
                    tokenizer = model_for_saving.mllm.processor.tokenizer
                    text_targets = prepare_text_targets(answers, tokenizer)
                    text_targets = text_targets.to(device)
                    
                    # Calculate loss
                    loss_dict = criterion(
                        z_mask=pred_masks,
                        y_mask=masks,
                        z_txt=text_logits,  # Use text logits, not embeddings
                        y_txt=text_targets
                    )
                    
                    loss_total = loss_dict["loss_total"]
                    # Scale loss for gradient accumulation
                    loss_total = loss_total / args.gradient_accumulation_steps
                
                # Backward pass with gradient scaling
                scaler.scale(loss_total).backward()
            else:
                # Standard precision forward pass
                outputs = model(images, questions)
                pred_masks = outputs["z_mask"]
                text_logits = outputs["z_txt_logits"]
                
                # Prepare text targets - access tokenizer from the underlying model
                tokenizer = model_for_saving.mllm.processor.tokenizer
                text_targets = prepare_text_targets(answers, tokenizer)
                text_targets = text_targets.to(device)
                
                # Calculate loss
                loss_dict = criterion(
                    z_mask=pred_masks,
                    y_mask=masks,
                    z_txt=text_logits,  # Use text logits, not embeddings
                    y_txt=text_targets
                )
                
                loss_total = loss_dict["loss_total"]
                # Scale loss for gradient accumulation
                loss_total = loss_total / args.gradient_accumulation_steps
                
                # Backward pass
                loss_total.backward()
            
            # Gradient accumulation: only update weights every N steps
            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                if scaler is not None:
                    # Unscale gradients before clipping
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                    optimizer.step()
                
                optimizer.zero_grad()
            
            # Accumulate losses (multiply by accumulation steps to get true loss)
            epoch_loss_total += loss_total.item() * args.gradient_accumulation_steps
            epoch_loss_seg += loss_dict["loss_seg"].item() * args.gradient_accumulation_steps
            epoch_loss_txt += loss_dict["loss_txt"].item() * args.gradient_accumulation_steps
            
            if is_main_process() and batch_idx % 10 == 0:
                # Clear cache periodically to free memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                print(f'Epoch {epoch+1}, Batch {batch_idx}, '
                      f'Total Loss: {loss_total.item() * args.gradient_accumulation_steps:.4f}, '
                      f'Seg Loss: {loss_dict["loss_seg"].item():.4f}, '
                      f'Text Loss: {loss_dict["loss_txt"].item():.4f}')
        
        # Handle remaining gradients if batch doesn't divide evenly
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
        
        # Calculate epoch averages
        avg_train_loss_total = epoch_loss_total / len(train_loader)
        avg_train_loss_seg = epoch_loss_seg / len(train_loader)
        avg_train_loss_txt = epoch_loss_txt / len(train_loader)
        epoch_time = time.time() - epoch_start_time
        
        if is_main_process():
            print(f'Epoch {epoch+1} - TRAIN - '
                  f'Total Loss: {avg_train_loss_total:.4f}, '
                  f'Seg Loss: {avg_train_loss_seg:.4f}, '
                  f'Text Loss: {avg_train_loss_txt:.4f}, '
                  f'Time: {epoch_time:.2f}s')
        
        # Validation phase
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
                
                # Use mixed precision for validation if enabled
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        outputs = model(images, questions)
                        pred_masks = outputs["z_mask"]
                        text_logits = outputs["z_txt_logits"]
                        
                        # Access tokenizer from the underlying model
                        tokenizer = model_for_saving.mllm.processor.tokenizer
                        text_targets = prepare_text_targets(answers, tokenizer)
                        text_targets = text_targets.to(device)
                        
                        loss_dict = criterion(
                            z_mask=pred_masks,
                            y_mask=masks,
                            z_txt=text_logits,
                            y_txt=text_targets
                        )
                else:
                    outputs = model(images, questions)
                    pred_masks = outputs["z_mask"]
                    text_logits = outputs["z_txt_logits"]
                    
                    # Access tokenizer from the underlying model
                    tokenizer = model_for_saving.mllm.processor.tokenizer
                    text_targets = prepare_text_targets(answers, tokenizer)
                    text_targets = text_targets.to(device)
                    
                    loss_dict = criterion(
                        z_mask=pred_masks,
                        y_mask=masks,
                        z_txt=text_logits,
                        y_txt=text_targets
                    )
                
                val_loss_total += loss_dict["loss_total"].item()
                val_loss_seg += loss_dict["loss_seg"].item()
                val_loss_txt += loss_dict["loss_txt"].item()
        
        # Aggregate validation losses across all GPUs
        if is_distributed:
            # Create tensors for aggregation
            val_loss_tensor = torch.tensor([val_loss_total, val_loss_seg, val_loss_txt], device=device)
            dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
            val_loss_total = val_loss_tensor[0].item()
            val_loss_seg = val_loss_tensor[1].item()
            val_loss_txt = val_loss_tensor[2].item()
            
            # Get total number of validation batches across all processes
            num_val_batches = torch.tensor(len(val_loader), device=device)
            dist.all_reduce(num_val_batches, op=dist.ReduceOp.SUM)
            total_val_batches = num_val_batches.item()
        else:
            total_val_batches = len(val_loader)
        
        # Calculate validation averages (across all GPUs if distributed)
        avg_val_loss_total = val_loss_total / total_val_batches
        avg_val_loss_seg = val_loss_seg / total_val_batches
        avg_val_loss_txt = val_loss_txt / total_val_batches
        
        if is_main_process():
            print(f'Epoch {epoch+1} - VALIDATION - '
                  f'Total Loss: {avg_val_loss_total:.4f}, '
                  f'Seg Loss: {avg_val_loss_seg:.4f}, '
                  f'Text Loss: {avg_val_loss_txt:.4f}')
        
        # Save checkpoints (only on main process)
        if is_main_process():
            if checkpoint_dir is None:
                print("WARNING: checkpoint_dir is None, creating default checkpoint directory")
                checkpoint_dir = os.path.join(args.checkpoint_dir, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
                os.makedirs(checkpoint_dir, exist_ok=True)
            
            if avg_val_loss_total < best_val_loss:
                best_val_loss = avg_val_loss_total
                save_checkpoint(epoch, model_for_saving, optimizer, checkpoint_dir, is_best=True)
                print(f"New best model saved with val_loss: {avg_val_loss_total:.4f}")
            
            # Save periodic checkpoints every 5 epochs
            if (epoch + 1) % 5 == 0:
                save_checkpoint(epoch, model_for_saving, optimizer, checkpoint_dir)
                print(f"Periodic checkpoint saved at epoch {epoch+1}")
            
            # Save checkpoint at end of epoch 1 (for debugging/verification)
            if epoch == 0:
                save_checkpoint(epoch, model_for_saving, optimizer, checkpoint_dir)
                print(f"Initial checkpoint saved at epoch {epoch+1}")
    
    if is_main_process():
        # Save final checkpoint
        if checkpoint_dir is not None:
            print("Saving final checkpoint...")
            save_checkpoint(args.num_epochs - 1, model_for_saving, optimizer, checkpoint_dir)
        
        print("Training completed!")
        print(f"Best validation loss: {best_val_loss:.4f}")
        if checkpoint_dir is not None:
            print(f"Final checkpoints saved in: {checkpoint_dir}")
            # List all checkpoint files
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
    
    # Clean up distributed training
    cleanup_distributed()

def save_checkpoint(epoch, model, optimizer, checkpoint_dir, is_best=False):
    """Save complete model checkpoint"""
    if checkpoint_dir is None:
        print("WARNING: checkpoint_dir is None, cannot save checkpoint")
        return
    
    try:
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'timestamp': datetime.now().isoformat()
        }
        
        if is_best:
            filename = f'best_model_epoch_{epoch+1}.pth'
        else:
            filename = f'checkpoint_epoch_{epoch+1}.pth'
        
        checkpoint_path = os.path.join(checkpoint_dir, filename)
        torch.save(checkpoint, checkpoint_path)
        print(f"✓ Checkpoint saved to {checkpoint_path}")
        
        # Verify file was actually written
        if os.path.exists(checkpoint_path):
            file_size = os.path.getsize(checkpoint_path) / (1024 * 1024)  # Size in MB
            print(f"  File size: {file_size:.2f} MB")
        else:
            print(f"ERROR: Checkpoint file was not created at {checkpoint_path}")
    except Exception as e:
        print(f"ERROR: Failed to save checkpoint: {e}")
        import traceback
        traceback.print_exc()


def debug_full_pipeline(model, train_loader, device):
    """Debug the entire pipeline step by step"""
    print("=== Debugging Full Pipeline ===")
    model.eval()
    
    with torch.no_grad():
        test_batch = next(iter(train_loader))
        images = test_batch['image'].to(device)
        masks = test_batch['mask'].to(device)
        questions = test_batch['question']
        
        print(f"1. Input images: {images.shape}")
        print(f"2. Input masks: {masks.shape}")
        
        # Step 1: Image preprocessing
        processed_images = model.preprocess_images(images)
        print(f"3. Processed images: {processed_images.shape}")
        
        # Step 2: Vision backbone
        z_image = model.vision_backbone(processed_images)
        print(f"4. Vision backbone output: {z_image.shape}")
        
        # Step 3: MLLM
        mllm_output = model.mllm(processed_images, questions, return_projected=True)
        z_emb = mllm_output["z_emb"]
        print(f"5. MLLM z_emb: {z_emb.shape}")
        
        # Step 4: Fusion module
        z_fused = model.fusion_module(z_image, z_emb)
        print(f"6. Fusion output: {z_fused.shape}")
        
        # Step 5: Mask prediction
        z_mask = model.mask_predictor(z_fused)
        print(f"7. Mask prediction: {z_mask.shape}")
        
        # Step 6: Compare with ground truth
        print(f"8. Ground truth masks: {masks.shape}")
        
        # Check if shapes match
        if z_mask.shape == masks.shape:
            print("✅ ALL SHAPES MATCH!")
        else:
            print(f"❌ SHAPE MISMATCH: Predicted {z_mask.shape} vs Target {masks.shape}")
            
            # Check which dimension is wrong
            for i, (pred_dim, target_dim) in enumerate(zip(z_mask.shape, masks.shape)):
                if pred_dim != target_dim:
                    print(f"   Dimension {i}: Predicted {pred_dim} vs Target {target_dim}")
    
    print("=== Pipeline Debug Complete ===")
    return z_mask.shape, masks.shape


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
        
        # Clean up distributed training if it was initialized
        try:
            cleanup_distributed()
        except:
            pass
        
        sys.exit(1)
    # torch.set_float32_matmul_precision('high')  # For faster float32 ops
    # torch.set_default_dtype(torch.float32)
    # torch.backends.cuda.matmul.allow_tf32 = True  # For faster float32 ops

    # args = parse_args()
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(f"Using device: {device}")

    # # Set default tensor type to float32
    # torch.set_default_dtype(torch.float32)
    
    # # Create checkpoint directory
    # timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # checkpoint_dir = os.path.join(args.checkpoint_dir, f'training_{timestamp}')
    # os.makedirs(checkpoint_dir, exist_ok=True)
    
    # # Initialize data loaders
    # print(f"Loading data from {args.data_root}...")
    # data_loader = PRSMedDataLoader(
    #     batch_size=args.batch_size,
    #     num_workers=args.num_workers,
    #     data_root=args.data_root
    # )
    
    # train_loader = data_loader.get_dataloader('train', shuffle=True)
    # val_loader = data_loader.get_dataloader('val', shuffle=False)
    

    # model = PRSMedModel(args, device)
    
    # debug_full_pipeline(model, train_loader, device)