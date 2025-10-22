import yaml, torch
from PIL import Image
from src.data.transforms import build_transforms
from src.models.vision_backbone import TinyVisionBackbone
from src.models.llm_backbone import MLLMWithLoRA
from src.models.fusion_decoder import PRSMedModel

def main(img_path: str, question: str, cfg_path="configs/prs_med.yaml", ckpt="checkpoints/best.pt"):
    cfg = yaml.safe_load(open(cfg_path))
    device = cfg.get("device","cuda")
    img_tf, _ = build_transforms(cfg["img_size"])

    vision = TinyVisionBackbone(cfg["vision"]["name"], cfg["vision"]["pretrained"], out_dim=cfg["vision"]["out_dim"]).to(device)
    mllm   = MLLMWithLoRA(cfg["llm"]["model_id"], cfg["llm"]["lora"]).to(device)
    model  = PRSMedModel(vision, mllm, proj_dim=cfg["fusion"]["proj_dim"], n_heads=cfg["fusion"]["n_heads"],
                         decoder_channels=tuple(cfg["decoder"]["up_channels"])).to(device)

    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["model"], strict=False)
    model.eval()

    img = Image.open(img_path).convert("RGB")
    x = img_tf(img).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(x, question=question)
        text = mllm.generate(x[0], question, cfg["gen"])
        mask = torch.sigmoid(out["mask_logits"]).cpu()[0,0].numpy()

    print("Answer:", text)
    # Save or visualize mask as needed
    # (avoid file I/O per your request; here we just report mean/confidence)
    print("Mask mean prob:", float(mask.mean()))

if __name__ == "__main__":
    import sys
    main(sys.argv[1], "Where is the tumour located in this image?")
