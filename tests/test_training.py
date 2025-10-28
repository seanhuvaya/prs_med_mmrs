import torch
from models.prs_med_model import PRSMedModel
from training.losses import SegmentationLoss, TextLoss
from training.optimizer import build_optimizer
from training.train_loop import Trainer


def test_losses_basic():
    pred = torch.sigmoid(torch.randn(2,1,16,16))
    target = torch.randint(0,2,(2,1,16,16)).float()
    l = SegmentationLoss()(pred,target)
    assert l > 0
    logits = torch.randn(2,4,10)
    labels = torch.randint(0,10,(2,4))
    assert TextLoss()(logits,labels) > 0

def test_optimizer_step():
    m = torch.nn.Linear(10,1)
    opt = build_optimizer(m)
    out = m(torch.randn(5,10)).mean()
    out.backward(); opt.step()
    assert True

def test_trainer_step(monkeypatch):
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(10, 1)  # Add trainable parameters
            
        def forward(self, imgs, texts):
            return {
                "mask": torch.sigmoid(torch.randn(imgs.size(0),1,16,16,requires_grad=True)),
                "logits": torch.randn(imgs.size(0),4,10,requires_grad=True),
            }
    dummy = DummyModel()
    trainer = Trainer(dummy, device="cpu")
    
    # Create individual samples instead of batches
    sample1 = {
        "image": torch.randn(3,224,224),
        "mask": torch.randint(0,2,(1,16,16)).float(),
        "question": "a",
        "answer_ids": torch.randint(0,10,(4,))
    }
    sample2 = {
        "image": torch.randn(3,224,224),
        "mask": torch.randint(0,2,(1,16,16)).float(),
        "question": "b",
        "answer_ids": torch.randint(0,10,(4,))
    }
    ds = [sample1, sample2]
    trainer.fit(ds, epochs=1, batch_size=1)
