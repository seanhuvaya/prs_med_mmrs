import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Computes soft Dice loss for binary segmentation."""
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        """
        pred:   (B, 1, H, W) - raw logits or probabilities
        target: (B, 1, H, W) - binary masks (0 or 1)
        """
        pred = torch.sigmoid(pred)        # convert logits → [0,1]
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1).float()

        intersection = (pred_flat * target_flat).sum(1)
        dice_score = (2. * intersection + self.smooth) / (
            pred_flat.sum(1) + target_flat.sum(1) + self.smooth
        )
        dice_loss = 1 - dice_score.mean()
        return dice_loss


class PRSMedLoss(nn.Module):
    """
    Fixed implementation: L_total = λ_seg * (BCE + Dice) + λ_txt * CE
    """
    def __init__(self, lambda_seg: float = 1.0, lambda_txt: float = 0.5, ignore_index: int = -100):
        super().__init__()
        self.lambda_seg = lambda_seg
        self.lambda_txt = lambda_txt
        self.ignore_index = ignore_index
        
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(
        self,
        z_mask: torch.Tensor,
        y_mask: torch.Tensor,
        z_txt: torch.Tensor,
        y_txt: torch.Tensor,
    ) -> dict:
        # ---- Segmentation loss ---- #
        loss_bce = self.bce(z_mask, y_mask)
        loss_dice = self.dice(z_mask, y_mask)
        loss_seg = loss_bce + loss_dice

        # ---- Text reasoning loss ---- #
        # FIX: Properly handle text logits and targets
        # z_txt: (B, L, V) where B=batch, L=sequence length, V=vocab size
        # y_txt: (B, L) where each element is a token ID
        
        B, L, V = z_txt.shape
        B_target, L_target = y_txt.shape
        
        print(f"Text loss debug: z_txt {z_txt.shape}, y_txt {y_txt.shape}")
        
        # Ensure batch sizes match
        if B != B_target:
            print(f"⚠️  Batch size mismatch: z_txt {B} vs y_txt {B_target}")
            # Use the smaller batch size
            min_batch = min(B, B_target)
            z_txt = z_txt[:min_batch]
            y_txt = y_txt[:min_batch]
            print(f"   Using batch size: {min_batch}")
        
        # Ensure sequence lengths match
        if L != L_target:
            print(f"⚠️  Sequence length mismatch: z_txt {L} vs y_txt {L_target}")
            # Use the minimum sequence length
            min_seq = min(L, L_target)
            z_txt = z_txt[:, :min_seq, :]
            y_txt = y_txt[:, :min_seq]
            print(f"   Using sequence length: {min_seq}")
        
        # Reshape for cross entropy: (B * L, V) and (B * L)
        z_txt_flat = z_txt.reshape(-1, V)
        y_txt_flat = y_txt.reshape(-1)
        
        print(f"Text loss final: z_txt_flat {z_txt_flat.shape}, y_txt_flat {y_txt_flat.shape}")
        
        loss_txt = self.ce(z_txt_flat, y_txt_flat)

        # ---- Weighted sum ---- #
        loss_total = self.lambda_seg * loss_seg + self.lambda_txt * loss_txt

        return {
            "loss_total": loss_total,
            "loss_seg": loss_seg.detach(),
            "loss_txt": loss_txt.detach(),
        }


if __name__ == "__main__":

    B, L, V, H, W = 2, 595, 32064, 1024, 1024

    z_mask = torch.randn(B, 1, H, W)
    y_mask = (torch.rand(B, 1, H, W) > 0.5).float()

    z_txt = torch.randn(B, L, V)
    y_txt = torch.randint(0, V, (B, L))

    criterion = PRSMedLoss(lambda_seg=1.0, lambda_txt=0.5)
    out = criterion(z_mask, y_mask, z_txt, y_txt)

    print(
        f"loss_total={out['loss_total']:.4f}, "
        f"loss_seg={out['loss_seg']:.4f}, "
        f"loss_txt={out['loss_txt']:.4f}"
    )
