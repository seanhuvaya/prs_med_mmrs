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
    L_total = λ_seg * (BCE + Dice) + λ_txt * lm_loss

    NOTE:
      - We no longer compute CE over tokens here.
      - `lm_loss` is computed inside the LLaVA-Med MLLM (LLavaMedMLLM)
        using proper masking of question tokens.
    """
    def __init__(self, lambda_seg: float = 1.0, lambda_txt: float = 0.5):
        super().__init__()
        self.lambda_seg = lambda_seg
        self.lambda_txt = lambda_txt

        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(
        self,
        z_mask: torch.Tensor,
        y_mask: torch.Tensor,
        lm_loss: torch.Tensor,
    ) -> dict:
        # ---- Segmentation loss ---- #
        loss_bce = self.bce(z_mask, y_mask)
        loss_dice = self.dice(z_mask, y_mask)
        loss_seg = loss_bce + loss_dice

        # ---- Total loss ---- #
        loss_total = self.lambda_seg * loss_seg
        loss_txt_value = torch.tensor(0.0, device=z_mask.device)

        if lm_loss is not None:
            loss_total = loss_total + self.lambda_txt * lm_loss
            loss_txt_value = lm_loss

        return {
            "loss_total": loss_total,
            "loss_seg": loss_seg.detach(),
            "loss_txt": loss_txt_value.detach(),
        }


if __name__ == "__main__":

    B, H, W = 2, 1024, 1024

    z_mask = torch.randn(B, 1, H, W)
    y_mask = (torch.rand(B, 1, H, W) > 0.5).float()
    lm_loss = torch.tensor(4.0)

    criterion = PRSMedLoss(lambda_seg=1.0, lambda_txt=0.5)
    out = criterion(z_mask=z_mask, y_mask=y_mask, lm_loss=lm_loss)

    print(
        f"loss_total={out['loss_total']:.4f}, "
        f"loss_seg={out['loss_seg']:.4f}, "
        f"loss_txt={out['loss_txt']:.4f}"
    )
