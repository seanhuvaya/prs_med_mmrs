"""Device management utilities."""

import torch
from typing import Union


def get_device(device_id: Union[int, str, None] = None) -> torch.device:
    """
    Get the appropriate device for training.
    
    Args:
        device_id: Device ID (int), device string ('cuda', 'cpu', 'mps'), or None for auto-detect
    
    Returns:
        torch.device instance
    """
    if device_id is not None:
        if isinstance(device_id, int):
            return torch.device(f'cuda:{device_id}')
        elif isinstance(device_id, str):
            return torch.device(device_id)
    
    # Auto-detect
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')
