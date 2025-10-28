import torch
import torch.nn as nn
import torch.nn.functional as F 

class SegmentationLoss(nn.Module):
    def __init__(self, bce_weights: float = 1.0, dice_weights: float = 1.0):
        super().__init__()
        self.bce = nn.BCELoss()
        self.bce_weight = bce_weights
        self.dice_weight = dice_weights
        
    def forward(self, pred, target):
        bce = self.bce(pred, target)
        dice = self._dice_loss(pred, target)
        return self.bce_weight * bce + self.dice_weight * dice

    @staticmethod
    def _dice_loss(pred, target, eps: float = 1e-6):
        pred, target = pred.flatten(), target.flatten()
        inter = (pred * target).sum()   
        return 1 - (2. * inter + eps) / (pred.sum() + target.sum() + eps)

class TextLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, targets):
        # Handle dimension mismatch between logits and targets
        # logits: [batch_size, seq_len, vocab_size]
        # targets: [batch_size, target_seq_len]
        
        batch_size = targets.shape[0]
        target_seq_len = targets.shape[1]
        
        # Ensure logits sequence length matches target sequence length
        if logits.shape[1] != target_seq_len:
            # Truncate or pad logits to match target length
            if logits.shape[1] > target_seq_len:
                logits = logits[:, :target_seq_len, :]
            else:
                # Pad logits if needed (though this shouldn't happen with our setup)
                padding_size = target_seq_len - logits.shape[1]
                padding = torch.zeros(batch_size, padding_size, logits.shape[2], 
                                    device=logits.device, dtype=logits.dtype)
                logits = torch.cat([logits, padding], dim=1)
        
        # Compute cross-entropy loss
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))