import os, yaml, torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from src.data.transforms import build_transforms
from src.data.dataset import PRSMedCSVDataset
from src.models.vision_backbone import TinyVisionBackbone
from src.models.llm_backbone import MLLMWithLoRA
from src.models.fusion_decoder import PRSMedModel
from src.utils.losses import bce_dice_loss, text_ce_loss
from src.utils.metrics import dice_iou
from src.utils.train_utils import seed_everything, save_ckpt

def main(cfg_path="configs/prs_med.yaml"):
    cfg = yaml.safe_load(open(cfg_path))
    seed_everything(cfg["seed"])
    accelerator = Accelerator(mixed_precision=cfg["precision"])

    img_tf, mask_tf = build_transforms(cfg["img_size"])

    train_set = PRSMedCSVDataset(cfg["train_csv"], cfg["image_root"], cfg["mask_root"], img_tf, mask_tf)
    val_set   = PRSMedCSVDataset(cfg["val_csv"],   cfg["image_root"], cfg["mask_root"], img_tf, mask_tf)

    train_loader = DataLoader(train_set, batch_size=cfg["train"]["batch_size"], shuffle=True,
                              num_workers=cfg["num_workers"], pin_memory=True)
    val_loader   = DataLoader(val_set, batch_size=cfg["train"]["batch_size"], shuffle=False,
                              num_workers=cfg["num_workers"], pin_memory=True)

    vision = TinyVisionBackbone(cfg["vision"]["name"], cfg["vision"]["pretrained"], out_dim=cfg["vision"]["out_dim"])
    mllm   = MLLMWithLoRA(cfg["llm"]["model_id"], cfg["llm"]["lora"])
    model  = PRSMedModel(vision, mllm, proj_dim=cfg["fusion"]["proj_dim"], n_heads=cfg["fusion"]["n_heads"],
                         decoder_channels=tuple(cfg["decoder"]["up_channels"]))

    optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                              lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])

    model, optim, train_loader, val_loader = accelerator.prepare(model, optim, train_loader, val_loader)

    best_val = -1.0
    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        for step, batch in enumerate(train_loader):
            pixel_values, mask_gt = batch["pixel_values"], batch["mask"]
            question, answer_gt = batch["question"], batch["answer_gt"]  # (strings list)

            # Build label ids for text CE
            # For simplicity we compare next-token prediction against the ground-truth answer text.
            # In practice you should build tokenizer labels aligned with the processor's prompt template.
            proc = model.mllm.processor(text=list(answer_gt), images=[p for p in pixel_values], return_tensors="pt", padding=True)
            labels = proc["input_ids"]
            labels[labels == model.mllm.processor.tokenizer.pad_token_id] = -100
            proc = {k: v.to(pixel_values.device) for k, v in proc.items()}

            out = model(pixel_values, question=question)

            seg_loss = bce_dice_loss(out["mask_logits"], mask_gt)
            txt_loss = text_ce_loss(out["logits"], labels.to(out["logits"].device))
            loss = cfg["loss"]["lambda_seg"] * seg_loss + cfg["loss"]["lambda_txt"] * txt_loss

            accelerator.backward(loss)
            optim.step(); optim.zero_grad()

            if (step + 1) % cfg["train"]["log_every"] == 0 and accelerator.is_main_process:
                dice, iou = dice_iou(out["mask_logits"].detach(), mask_gt)
                print(f"epoch {epoch} step {step+1}: loss={loss.item():.4f} seg={seg_loss.item():.4f} txt={txt_loss.item():.4f} d={dice:.3f} iou={iou:.3f}")

        # validation
        model.eval()
        dices, ious = [], []
        with torch.no_grad():
            for batch in val_loader:
                out = model(batch["pixel_values"], question=batch["question"])
                d, i = dice_iou(out["mask_logits"], batch["mask"])
                dices.append(d); ious.append(i)
        mean_d, mean_i = sum(dices)/len(dices), sum(ious)/len(ious)
        if accelerator.is_main_process:
            print(f"[val] epoch {epoch}: dice={mean_d:.4f} iou={mean_i:.4f}")
            if mean_d > best_val:
                best_val = mean_d
                path = save_ckpt({"epoch": epoch, "model": accelerator.get_state_dict(model)}, cfg["train"]["ckpt_dir"], "best.pt")
                print(f"saved {path}")

if __name__ == "__main__":
    main()
