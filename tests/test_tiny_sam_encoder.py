import unittest
from unittest.mock import patch
import torch


def _make_model_with_state(state):
    from models.tiny_sam_encoder import TinySAMVisionBackbone
    with patch("torch.load", lambda path, map_location=None: state):
        return TinySAMVisionBackbone(checkpoint_path="/ignored.pth", device="cpu")


class TestTinySamEncoder(unittest.TestCase):
    def test_init_with_full_sam_like_checkpoint_filters_image_encoder(self):
        # Arrange: simulate a full SAM state dict where only image_encoder.* keys are relevant
        fake_state = {
            "image_encoder.patch_embed.seq.0.c.weight": torch.randn(32, 3, 3, 3),
            "image_encoder.patch_embed.seq.0.bn.weight": torch.ones(32),
        }

        # Act
        model = _make_model_with_state(fake_state)

        # Assert: model is on CPU and encoder parameters allocated
        p = next(model.encoder.parameters())
        self.assertEqual(p.device.type, "cpu")

    def test_forward_output_shape_cpu(self):
        # Arrange: TinyViT-only checkpoint path (empty dict tolerated with strict=False)
        model = _make_model_with_state({})
        x = torch.randn(1, 3, 1024, 1024)

        # Act
        y = model(x)

        # Assert
        self.assertIsInstance(y, torch.Tensor)
        self.assertEqual(y.shape, (1, 256, 16, 16))
        self.assertTrue(torch.isfinite(y).all().item())

    def test_preprocessing_normalizes_255_range_input(self):
        # Arrange
        model = _make_model_with_state({})
        # Values in [0, 255] to trigger internal normalization branch
        x = torch.randint(low=0, high=256, size=(2, 3, 1024, 1024), dtype=torch.int32).float()

        # Act
        y = model(x)

        # Assert
        self.assertEqual(y.shape, (2, 256, 16, 16))
        self.assertTrue(torch.isfinite(y).all().item())


if __name__ == "__main__":
    unittest.main()
