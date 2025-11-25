import unittest
import torch

from models.decoder.fusion_module import PromptMaskFusionModule
from models.decoder.mask_prediction_module import MaskPredictionModule


class TestPromptMaskFusionModule(unittest.TestCase):
    def test_forward_shapes_and_dtype(self):
        torch.manual_seed(0)
        B, C, H, W = 2, 256, 16, 16
        L, H_emb = 7, 4096

        # z_image float32, z_emb float16 to test internal casting
        z_image = torch.randn(B, C, H, W, dtype=torch.float32)
        z_emb = torch.randn(B, L, H_emb, dtype=torch.float16)

        module = PromptMaskFusionModule(img_dim=C, emb_dim=H_emb, fused_dim=256, num_heads=4)
        module = module.float()  # ensure module params are float32 on CPU

        out = module(z_image, z_emb)

        # Output shape and dtype
        self.assertEqual(out.shape, (B, 256, H, W))
        self.assertEqual(out.dtype, torch.float32)

        # Check values are finite
        self.assertTrue(torch.isfinite(out).all())

    def test_backward(self):
        B, C, H, W = 1, 256, 16, 16
        L, H_emb = 5, 4096
        z_image = torch.randn(B, C, H, W, dtype=torch.float32, requires_grad=True)
        z_emb = torch.randn(B, L, H_emb, dtype=torch.float32, requires_grad=True)

        module = PromptMaskFusionModule(img_dim=C, emb_dim=H_emb, fused_dim=256, num_heads=2)
        module = module.float()

        out = module(z_image, z_emb)
        loss = out.mean()
        loss.backward()

        # Gradients should be populated
        self.assertIsNotNone(z_image.grad)
        self.assertIsNotNone(z_emb.grad)
        self.assertTrue(torch.isfinite(z_image.grad).all())
        self.assertTrue(torch.isfinite(z_emb.grad).all())


class TestMaskPredictionModule(unittest.TestCase):
    def test_forward_shape(self):
        torch.manual_seed(0)
        B = 2
        x = torch.randn(B, 256, 16, 16)
        module = MaskPredictionModule()
        out = module(x)
        self.assertEqual(out.shape, (B, 1, 1024, 1024))
        self.assertTrue(torch.isfinite(out).all())

    def test_backward(self):
        x = torch.randn(1, 256, 16, 16, requires_grad=True)
        module = MaskPredictionModule()
        out = module(x)
        # Simple loss on a small crop to keep memory reasonable
        loss = out[:, :, :8, :8].mean()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())


if __name__ == "__main__":
    unittest.main()
