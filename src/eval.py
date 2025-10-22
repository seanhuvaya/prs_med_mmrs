import yaml, torch
from torch.utils.data import DataLoader
from src.data.transforms import build_transforms
from src.data.dataset import PRSMedCSVDataset
from src.models.vision_backbone import TinyVisionBackbone
from src.models.llm_backbone import MLLMWithLoRA
from src.models.fusion_decoder import PRSMedModel
from src.utils.metrics import dice_iou

def main(cfg_path="configs/prs_med.yaml", ckpt="checkpoints/best.pt"):
    cfg = yaml.safe_load(open(cfg_path))
    img_tf, mask_tf = build_transforms(cfg["img_size"])
    val_set = PRSMedCSVDataset(cfg["val_csv"], cfg["image_root"], cfg["mask_root"], img_tf, mask_tf)
    loader = DataLoader(val_set, batch_size=1, shuffle=False)
    device = cfg.get("device","cuda")

    vision = TinyVisionBackbone(cfg["vision"]["name"], cfg["vision"]["pretrained"], out_dim=cfg["vision"]["out_dim"]).to(device)
    mllm   = MLLMWithLoRA(cfg["llm"]["model_id"], cfg["llm"]["lora"]).to(device)
    model  = PRSMedModel(vision, mllm, proj_dim=cfg["fusion"]["proj_dim"], n_heads=cfg["fusion"]["n_heads"],
                         decoder_channels=tuple(cfg["decoder"]["up_channels"])).to(device)

    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["model"], strict=False)

    model.eval()
    dices, ious = [], []
    with torch.no_grad():
        for batch in loader:
            pixel_values = batch["pixel_values"].to(device)
            out = model(pixel_values, question=batch["question"])
            d, i = dice_iou(out["mask_logits"], batch["mask"].to(device))
            dices.append(d); ious.append(i)
    print(f"Dice={sum(dices)/len(dices):.4f} IoU={sum(ious)/len(ious):.4f}")

if __name__ == "__main__":
    main()
