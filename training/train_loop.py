import torch

from torch.utils.data import DataLoader
from tqdm import tqdm
from .losses import SegmentationLoss, TextLoss
from .optimizer import build_optimizer


class Trainer:
    def __init__(self, model, lambda_seg=1.0, lambda_text=1.0, lr=1e-4, device="mps"):
        self.model = model.to(device)
        self.device = device
        self.seg_loss = SegmentationLoss()
        self.text_loss = TextLoss()
        self.optimizer = build_optimizer(model, lr=lr)
        self.scheduler = None
        self.lambda_seg = lambda_seg
        self.lambda_text = lambda_text

    def fit(self, train_ds, val_ds=None, epochs: int = 20, batch_size: int = 8):
        loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        for epoch in range(1, epochs + 1):
            self.model.train()
            total_loss = 0.0
            for batch in tqdm(loader, desc=f"Epoch {epoch}/{epochs}"):
                imgs = batch["image"].to(self.device)
                masks = batch["mask"].to(self.device)
                texts = batch["question"]
                labels = batch["answer_ids"].to(self.device)

                out = self.model(imgs, texts)
                seg_loss = self.seg_loss(out["mask"], masks)
                text_loss = self.text_loss(out["logits"], labels)
                loss = self.lambda_seg * seg_loss + self.lambda_text * text_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
            print(f"Epoch {epoch}/{epochs} - Loss: {total_loss / len(loader):.4f}")

            if val_ds:
                self.evaluate(val_ds, batch_size=batch_size)

    @torch.no_grad()
    def evaluate(self, val_ds, batch_size: int = 8):
        loader = DataLoader(val_ds, batch_size=batch_size)
        total_dice = 0.0
        for batch in tqdm(loader, desc="Evaluating"):
            imgs = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device)
            texts = batch["question"]
            
            out = self.model(imgs, texts)
            pred = (out["mask"] > 0.5).float()
            dice = 1 - SegmentationLoss._dice_loss(pred, masks)
            total_dice += dice.item()

        print(f"Validation Dice: {total_dice / len(loader):.4f}")
                