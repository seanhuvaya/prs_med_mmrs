import torch

def build_optimizer(model, lr: float = 1e-4, weight_decay: float = 1e-2):
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

def build_scheduler(optimizer, num_warmup: int = 1000, num_training_steps: int = 20000):
    from transformers import get_cosine_schedule_with_warmup
    return get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup, num_training_steps=num_training_steps
    )