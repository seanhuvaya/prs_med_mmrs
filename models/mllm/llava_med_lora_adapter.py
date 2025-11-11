import torch
from typing import Optional
from peft import LoraConfig, get_peft_model, TaskType
from .llava_med_mllm import LLavaMedMLLM, DEFAULT_MODEL_NAME

class LLavaMedWithLoRA(LLavaMedMLLM):
    """
    LLaVA-Med MLLM with LoRA adapter.
    """
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        rank: int = 16,
        alpha:int = 16,
        dropout: float = 0.05,
        freeze_llm: bool = True,
        device: Optional[str] = None,
    ):
        super().__init__(
            model_name=model_name,
            freeze_llm=freeze_llm,
            device=device
        )

        self.lora_config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            bias="none",
            target_modules=[
                "q_proj", 
                "k_proj", 
                "v_proj", 
                "o_proj",
                "gate_proj",
                "down_proj",
                "up_proj",
            ],
            task_type=TaskType.CAUSAL_LM,
        )
        
        self.model = get_peft_model(self.model, self.lora_config)
        self.model.print_trainable_parameters()

        for name, param in self.model.named_parameters():
            if "lora_" in name:
                param.requires_grad = True

if __name__ == "__main__":
    from PIL import Image

    mllm = LLavaMedWithLoRA(
        rank=16,
        alpha=16,
        dropout=0.05,
        freeze_llm=True,
    )
    mllm.eval()

    img = Image.new("RGB", (512, 512), color=(128, 128, 128))
    batch_images = [img, img]
    batch_questions = ["Where is the tumor located?", "Describe the position of the lesion."]

    with torch.no_grad():
        out = mllm(batch_images, batch_questions, return_projected=True)

    z_emb = out["z_emb"]
    z_emb_proj = out["z_emb_proj"]
    z_txt = out["z_txt"]
    pred_ids = out["pred_ids"]

    print(f"z_emb: {tuple(z_emb.shape)}, z_emb_proj: {tuple(z_emb_proj.shape)}, z_txt: {tuple(z_txt.shape)}, pred_ids: {tuple(pred_ids.shape)}")

    mllm.model.save_pretrained("checkpoints/llava_med_lora_test")
    print(f"Saved model to checkpoints/llava_med_lora_test")
    