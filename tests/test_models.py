import torch
import pytest
from models.image_encoder import TinyVisionEncoder
from models.fusion_module import CrossAttentionFusion
from models.mask_decoder import MaskDecoder
from models.prs_med_model import PRSMedModel

@pytest.mark.parametrize("b,h,w", [(2,224,224)])
def test_image_encoder_shape(b,h,w):
    enc = TinyVisionEncoder(pretrained=False)
    x = torch.randn(b,3,h,w)
    out = enc(x)
    assert out.shape[1] == 256
    assert out.ndim == 4

def test_fusion_module_shapes():
    B,H,W,L = 2,16,16,32
    z_img = torch.randn(B,256,H,W)
    z_txt = torch.randn(B,L,1024)  # Updated to match text encoder output
    fusion = CrossAttentionFusion()
    out = fusion(z_img,z_txt)
    assert out.shape == z_img.shape

def test_mask_decoder_output():
    dec = MaskDecoder()
    x = torch.randn(2,256,14,14)
    out = dec(x)
    assert out.shape == (2,1,224,224)  # Updated to match new target size
    assert (0 <= out).all() and (out <= 1).all()

def test_full_model_forward():
    # stub text encoder to speed test
    class DummyLMHead(torch.nn.Module):
        def forward(self, x): return torch.randn(2,32,30522)
    
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lm_head = DummyLMHead()
    
    class DummyText(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = DummyModel()
        
        def forward(self, texts): return torch.randn(2,32,1024)  # Updated to match text encoder output
    
    m = PRSMedModel()
    m.text_encoder = DummyText()
    out = m(torch.randn(2,3,224,224), ["a","b"])
    assert out["mask"].shape == (2,1,224,224)  # Updated to match new target size
    assert "logits" in out
    loss = out["mask"].mean()
    loss.backward()