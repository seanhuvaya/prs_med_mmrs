import unittest
from unittest.mock import patch
import types
import torch


class DummyProcessor:
    def __init__(self):
        self.last_text = None
        self.last_images = None

    def __call__(self, images, text, return_tensors="pt", padding=True):
        self.last_images = images
        self.last_text = text
        # Return minimal inputs that mimic HF processor output
        batch = len(text)
        return types.SimpleNamespace(
            to=lambda device: {
                "input_ids": torch.randint(0, 10, (batch, 5)),
                "attention_mask": torch.ones(batch, 5, dtype=torch.long),
                "pixel_values": torch.randn(batch, 3, 224, 224),
            }
        )


class DummyOutputs:
    def __init__(self, batch, seq_len=5, hidden=4096, vocab=100):
        # Create 4 layers to test indexing, ensure deterministic shapes
        self.hidden_states = [torch.randn(batch, seq_len, hidden) for _ in range(4)]
        self.logits = torch.randn(batch, seq_len, vocab)


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = types.SimpleNamespace(hidden_size=4096)
        # Register a dummy parameter to carry dtype/device
        self.param = torch.nn.Parameter(torch.zeros(()))

    def forward(self, input_ids=None, attention_mask=None, pixel_values=None, output_hidden_states=True):
        batch = input_ids.shape[0]
        return DummyOutputs(batch=batch)

    # Mimic HF API used by save_pretrained in the wrapper
    def save_pretrained(self, save_directory: str):
        # Touch a file to simulate saving
        import os
        with open(os.path.join(save_directory, "dummy_model.bin"), "wb") as f:
            f.write(b"ok")


class TestLLavaMedMLLM(unittest.TestCase):
    @patch("models.mllm.llava_med_mllm.AutoProcessor.from_pretrained")
    @patch("models.mllm.llava_med_mllm.LlavaForConditionalGeneration.from_pretrained")
    def test_forward_and_projection_with_mock(self, mock_model_from_pretrained, mock_proc_from_pretrained):
        dummy_proc = DummyProcessor()
        mock_proc_from_pretrained.return_value = dummy_proc
        dummy_model = DummyModel()
        mock_model_from_pretrained.return_value = dummy_model

        from models.mllm.llava_med_mllm import LLavaMedMLLM

        m = LLavaMedMLLM(device="cpu", freeze_llm=True, paper_preset=True)
        m.eval()

        # Two dummy PIL images are not needed; we can pass tensors and the wrapper will convert
        img = torch.rand(3, 32, 32)
        images = [img, img]
        questions = ["Where is the lesion?", "Describe the location."]

        with torch.no_grad():
            out = m(images, questions, return_projected=True)

        # Shapes
        self.assertIn("z_emb", out)
        self.assertIn("z_txt", out)
        self.assertIn("pred_ids", out)
        self.assertIn("z_emb_proj", out)

        z_emb, z_txt, pred = out["z_emb"], out["z_txt"], out["pred_ids"]
        z_proj = out["z_emb_proj"]
        self.assertEqual(z_emb.shape[-1], 4096)
        self.assertEqual(z_proj.shape[-1], 256)
        self.assertEqual(z_txt.ndim, 3)
        self.assertEqual(pred.shape, z_txt.argmax(-1).shape)

        # Prompt formatting: ensure template used and <image> token present
        self.assertIsNotNone(dummy_proc.last_text)
        self.assertEqual(len(dummy_proc.last_text), len(questions))
        for q, t in zip(questions, dummy_proc.last_text):
            self.assertIn("<image>", t)
            self.assertIn(q, t)
            self.assertTrue(t.strip().endswith("ASSISTANT:"))

    @patch("models.mllm.llava_med_mllm.AutoProcessor.from_pretrained")
    @patch("models.mllm.llava_med_mllm.LlavaForConditionalGeneration.from_pretrained")
    def test_hidden_state_layer_and_pooling(self, mock_model_from_pretrained, mock_proc_from_pretrained):
        # Arrange
        dummy_proc = DummyProcessor()
        mock_proc_from_pretrained.return_value = dummy_proc
        dummy_model = DummyModel()
        mock_model_from_pretrained.return_value = dummy_model

        from models.mllm.llava_med_mllm import LLavaMedMLLM

        # Use specific layer index and mean pooling
        m = LLavaMedMLLM(device="cpu", freeze_llm=True, hidden_state_layer=-2, visual_pooling="mean")
        m.eval()

        img = torch.rand(3, 32, 32)
        images = [img]
        questions = ["Test?"]

        with torch.no_grad():
            out = m(images, questions, return_projected=True)

        # Asserts: selected layer shape and pooled embedding
        self.assertIn("z_emb", out)
        self.assertIn("z_emb_pooled", out)
        self.assertIn("z_emb_proj", out)
        self.assertIn("z_emb_proj_pooled", out)
        z_emb = out["z_emb"]
        z_pooled = out["z_emb_pooled"]
        self.assertEqual(z_emb.ndim, 3)
        self.assertEqual(z_pooled.ndim, 2)
        # Projection dims
        self.assertEqual(out["z_emb_proj"].shape[-1], 256)
        self.assertEqual(out["z_emb_proj_pooled"].shape[-1], 256)

    @patch("models.mllm.llava_med_mllm.AutoProcessor.from_pretrained")
    @patch("models.mllm.llava_med_mllm.LlavaForConditionalGeneration.from_pretrained")
    def test_save_and_load_projectors(self, mock_model_from_pretrained, mock_proc_from_pretrained):
        # Arrange mocks
        dummy_proc = DummyProcessor()
        mock_proc_from_pretrained.return_value = dummy_proc
        dummy_model = DummyModel()
        mock_model_from_pretrained.return_value = dummy_model

        from models.mllm.llava_med_mllm import LLavaMedMLLM

        m1 = LLavaMedMLLM(device="cpu", freeze_llm=True, paper_preset=True)
        # Do a tiny forward to initialize any lazy buffers
        img = torch.rand(3, 16, 16)
        with torch.no_grad():
            _ = m1([img], ["q"], return_projected=True)

        # Save to a temporary directory
        import tempfile, os
        tmpdir = tempfile.mkdtemp()
        m1.save_pretrained(tmpdir)

        # Capture projector state before load
        pre_state = {k: v.clone() for k, v in m1.to_seg_channels.state_dict().items()}

        # Now mock calls again for from_pretrained constructor path
        dummy_proc2 = DummyProcessor()
        mock_proc_from_pretrained.return_value = dummy_proc2
        dummy_model2 = DummyModel()
        mock_model_from_pretrained.return_value = dummy_model2

        # Load back
        m2 = LLavaMedMLLM.from_pretrained(tmpdir, device="cpu", freeze_llm=True, paper_preset=True)

        # Check projector weights restored
        for (k1, v1), (k2, v2) in zip(pre_state.items(), m2.to_seg_channels.state_dict().items()):
            self.assertEqual(k1, k2)
            self.assertTrue(torch.allclose(v1, v2))

        # And config restored (prompt template contains sentinel)
        self.assertIn("<image>", m2.prompt_template)


if __name__ == "__main__":
    unittest.main()
