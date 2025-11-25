import unittest
import torch

from models.loss.objective_function import DiceLoss, PRSMedLoss


class TestDiceLoss(unittest.TestCase):
    def test_dice_perfect_match(self):
        # Create a small binary mask and logits that perfectly match it
        B, H, W = 2, 8, 8
        target = torch.zeros(B, 1, H, W)
        target[:, :, :4, :4] = 1.0

        # Logits: large positive where target=1, large negative where target=0
        logits = torch.full_like(target, -10.0)
        logits[target == 1.0] = 10.0

        loss = DiceLoss(smooth=1e-6)(logits, target)
        self.assertLess(loss.item(), 1e-3)

    def test_dice_complete_mismatch(self):
        B, H, W = 1, 6, 6
        target = torch.zeros(B, 1, H, W)
        target[:, :, :3, :3] = 1.0
        # Invert target for logits
        logits = torch.full_like(target, -10.0)
        logits[target == 0.0] = 10.0

        loss = DiceLoss(smooth=1e-6)(logits, target)
        # Not exactly 1 due to sigmoid + smooth, but should be high
        self.assertGreater(loss.item(), 0.9)


class TestPRSMedLoss(unittest.TestCase):
    def test_loss_outputs_and_weighting(self):
        torch.manual_seed(0)
        B, L1, L2, V, H, W = 2, 7, 5, 23, 16, 16

        # Segmentation inputs
        z_mask = torch.randn(B, 1, H, W)
        y_mask = (torch.rand(B, 1, H, W) > 0.5).float()

        # Text logits and targets with mismatched sequence lengths
        z_txt = torch.randn(B, L1, V)
        y_txt = torch.randint(0, V, (B, L2))

        # Include some ignore_index tokens
        y_txt[0, 0] = -100

        crit = PRSMedLoss(lambda_seg=1.0, lambda_txt=0.5)
        out = crit(z_mask, y_mask, z_txt, y_txt)

        # Keys present
        self.assertIn("loss_total", out)
        self.assertIn("loss_seg", out)
        self.assertIn("loss_txt", out)

        # Finite and non-negative
        self.assertTrue(torch.isfinite(out["loss_total"]))
        self.assertTrue(torch.isfinite(out["loss_seg"]))
        self.assertTrue(torch.isfinite(out["loss_txt"]))
        self.assertGreaterEqual(out["loss_total"].item(), 0.0)

        # Weighting effect: increasing lambda_txt should increase total (same inputs)
        crit2 = PRSMedLoss(lambda_seg=1.0, lambda_txt=1.5)
        out2 = crit2(z_mask, y_mask, z_txt, y_txt)
        self.assertGreaterEqual(out2["loss_total"].item(), out["loss_total"].item())

    def test_shapes_align_when_batch_or_seq_mismatch(self):
        # Different batch sizes and sequence lengths should be clipped to min
        Bz, By, Lz, Ly, V = 3, 2, 6, 4, 11
        z_mask = torch.randn(By, 1, 8, 8)
        y_mask = (torch.rand(By, 1, 8, 8) > 0.5).float()
        z_txt = torch.randn(Bz, Lz, V)
        y_txt = torch.randint(0, V, (By, Ly))

        crit = PRSMedLoss(lambda_seg=0.0, lambda_txt=1.0)
        out = crit(z_mask, y_mask, z_txt, y_txt)
        # With lambda_seg=0, total should equal text loss (up to float precision)
        self.assertAlmostEqual(out["loss_total"].item(), out["loss_txt"].item(), places=6)


if __name__ == "__main__":
    unittest.main()
