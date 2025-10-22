import torch
import torch.nn.functional as F

def dice_loss(pred, target, eps=1e-6):
    pred = torch.sigmoid(pred)
    num = 2 * (pred * target).sum(dim=(2,3)) + eps
    den = pred.sum(dim=(2,3)) + target.sum(dim=(2,3)) + eps
    loss = 1 - (num / den)
    return loss.mean()

def bce_dice_loss(mask_logits, mask_gt):
    bce = F.binary_cross_entropy_with_logits(mask_logits, mask_gt)
    dc = dice_loss(mask_logits, mask_gt)
    return bce + dc

def text_ce_loss(logits, labels, ignore_index=-100):
    # logits: [B,L,V], labels: [B,L]
    return F.cross_entropy(logits.transpose(1,2), labels, ignore_index=ignore_index)
