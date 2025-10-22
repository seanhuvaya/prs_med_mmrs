import torch

def threshold(pred, thr=0.5): return (pred > thr).float()

def dice_iou(mask_logits, mask_gt, thr=0.5, eps=1e-6):
    pred = torch.sigmoid(mask_logits)
    pred = threshold(pred, thr)
    inter = (pred*mask_gt).sum(dim=(2,3))
    union = pred.sum(dim=(2,3)) + mask_gt.sum(dim=(2,3)) - inter
    dice = (2*inter + eps) / (pred.sum(dim=(2,3)) + mask_gt.sum(dim=(2,3)) + eps)
    iou  = (inter + eps) / (union + eps)
    return dice.mean().item(), iou.mean().item()
