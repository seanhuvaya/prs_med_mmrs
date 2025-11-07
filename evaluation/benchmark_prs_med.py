import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()


def dice_coefficient(pred_mask: torch.Tensor, true_mask: torch.Tensor, smooth=1e-6):
    # Resize predicted mask to match ground truth resolution
    if pred_mask.shape != true_mask.shape:
        pred_mask = F.interpolate(pred_mask, size=true_mask.shape[2:], mode="bilinear", align_corners=False)
    
    pred_mask = torch.sigmoid(pred_mask)
    pred_mask = (pred_mask > 0.5).float()
    intersection = (pred_mask * true_mask).sum(dim=(1, 2, 3))
    dice = (2. * intersection + smooth) / (
        pred_mask.sum(dim=(1, 2, 3)) + true_mask.sum(dim=(1, 2, 3)) + smooth
    )
    return dice.mean().item()


def iou_score(pred_mask: torch.Tensor, true_mask: torch.Tensor, smooth=1e-6):
    # 🔧 Ensure both tensors have same spatial resolution
    if pred_mask.shape != true_mask.shape:
        pred_mask = F.interpolate(pred_mask, size=true_mask.shape[2:], mode="bilinear", align_corners=False)

    pred_mask = torch.sigmoid(pred_mask)
    pred_mask = (pred_mask > 0.5).float()
    intersection = (pred_mask * true_mask).sum(dim=(1, 2, 3))
    union = (pred_mask + true_mask - pred_mask * true_mask).sum(dim=(1, 2, 3))
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean().item()


def evaluate_text_reasoning(pred_texts: List[str], gt_texts: List[str]) -> Dict[str, float]:
    """
    Uses HF-hosted Qwen 3 & Llama 3.1 to score positional correctness.
    Expects HF_TOKEN in env or a cached login.
    """
    import os
    import numpy as np
    from tqdm import tqdm
    from huggingface_hub import InferenceClient

    HF_TOKEN = os.getenv("HF_TOKEN")  # or rely on cached login
    # Repos: pick sizes you can run via HF Inference (or swap to others you prefer)
    QWEN_MODEL  = os.getenv("QWEN_MODEL",  "Qwen/Qwen2.5-7B-Instruct")
    LLAMA_MODEL = os.getenv("LLAMA_MODEL", "meta-llama/Llama-3.1-8B-Instruct")

    # One client per model (keeps headers/session warm)
    qwen_client  = InferenceClient(model=QWEN_MODEL,  token=HF_TOKEN)
    llama_client = InferenceClient(model=LLAMA_MODEL, token=HF_TOKEN)

    SYSTEM_MSG = (
        "You are evaluating if two medical position descriptions refer to the same region. "
        "If they describe the same anatomical position (even if phrased differently), "
        "respond only with 'yes'. Otherwise, respond with 'no'."
    )

    def ask_hf(client: InferenceClient, pred: str, gt: str) -> int:
        # Use the chat.completions API for instruction-tuned chat models
        try:
            resp = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": SYSTEM_MSG},
                    {"role": "user", "content": f'Ground Truth: "{gt}"\nPredicted Answer: "{pred}"'}
                ],
                temperature=0.0,
                max_tokens=8,         # tiny, we only need "yes"/"no"
                top_p=1.0,
                seed=0,
            )
            text = resp.choices[0].message["content"].strip().lower()
            return 1 if "yes" in text and "no" not in text else 0
        except Exception as e:
            print(f"[HF error] {e}")
            return 0

    qwen_scores, llama_scores = [], []
    for pred, gt in tqdm(zip(pred_texts, gt_texts), total=len(gt_texts), desc="Evaluating text reasoning (HF)"):
        qwen_scores.append(ask_hf(qwen_client,  pred, gt))
        llama_scores.append(ask_hf(llama_client, pred, gt))

    qwen_acc = float(np.mean(qwen_scores)) if len(qwen_scores) else 0.0
    llama_acc = float(np.mean(llama_scores)) if len(llama_scores) else 0.0
    ensemble_acc = float(np.mean([qwen_acc, llama_acc]))
    return {"qwen_acc": qwen_acc, "llama_acc": llama_acc, "ensemble_acc": ensemble_acc}



def evaluate_prs_med(model, data_loader, device):
    model.to(device)
    model.eval()
    dice_scores, iou_scores = [], []
    pred_texts, gt_texts = [], []

    with torch.no_grad():
        for images, questions, gt_masks, gt_tokens, gt_answers in tqdm(data_loader, desc="Benchmarking"):
            images = images.to(device)
            gt_masks = gt_masks.to(device)
            outputs = model(images, questions)

            # segmentation metrics
            d = dice_coefficient(outputs["z_mask"], gt_masks)
            i = iou_score(outputs["z_mask"], gt_masks)
            dice_scores.append(d)
            iou_scores.append(i)

            # convert token predictions back to text (decode via processor)
            pred_ids = torch.argmax(outputs["z_txt"], dim=-1)
            pred_text_batch = model.mllm.processor.batch_decode(pred_ids, skip_special_tokens=True)
            pred_texts.extend(pred_text_batch)
            gt_texts.extend(gt_answers)

    mdice = np.mean(dice_scores)
    miou = np.mean(iou_scores)
    print(f"Segmentation mDice={mdice:.4f}, mIoU={miou:.4f}")

    text_metrics = evaluate_text_reasoning(pred_texts, gt_texts)
    print(f"Reasoning Accuracy: "
          f"Qwen={text_metrics['qwen_acc']:.3f}, "
          f"LLaMA={text_metrics['llama_acc']:.3f}, "
          f"Ensemble={text_metrics['ensemble_acc']:.3f}")

    return {
        "mDice": mdice,
        "mIoU": miou,
        **text_metrics
    }


if __name__ == "__main__":
    from train_prs_med import PRSMedModel
    dummy_model = PRSMedModel()
    
    dummy_imgs = torch.randn(2, 3, 1024, 1024)
    dummy_masks = (torch.rand(2, 1, 1024, 1024) > 0.5).float()
    dummy_tokens = torch.randint(0, 32064, (2, 595))
    dummy_questions = ["Where is the tumor?", "Locate the lesion region."]
    dummy_answers = ["top right lobe", "center of image"]

    dummy_ds = [
        (dummy_imgs[i], [dummy_questions[i]], dummy_masks[i], dummy_tokens[i], dummy_answers[i])
        for i in range(2)
    ]
    dummy_loader = DataLoader(dummy_ds, batch_size=1)

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    metrics = evaluate_prs_med(dummy_model, dummy_loader, device)
    print(metrics)