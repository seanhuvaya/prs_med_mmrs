import torch
from typing import Optional
from peft import LoraConfig, get_peft_model, TaskType

from .llava_med_mllm import LLavaMedMLLM, DEFAULT_MODEL_NAME


class LLavaMedWithLoRA(LLavaMedMLLM):
    """
    LLaVA-Med MLLM with LoRA adapter.

    Base LLM can be frozen; LoRA layers become trainable and used for fine-tuning
    the text branch (L_text).
    """
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        rank: int = 16,
        alpha: int = 16,
        dropout: float = 0.05,
        freeze_llm: bool = True,
        training_texts: bool = True,
        device: Optional[str] = None,
    ):
        super().__init__(
            model_name=model_name,
            freeze_llm=freeze_llm,
            device=device,
            training_texts=training_texts
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

        # Wrap base LLaVA model with LoRA
        self.model = get_peft_model(self.model, self.lora_config)

        # LoRA params must be trainable
        for name, param in self.model.named_parameters():
            if "lora_" in name:
                param.requires_grad = True

        # Optional: print summary of trainable params
        self.model.print_trainable_parameters()


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
    batch_answers = ["In the left lobe.", "In the upper right region."]

    with torch.no_grad():
        out = mllm(
            batch_images,
            batch_questions,
            answers=batch_answers,
            training_texts=True,
            return_projected=True,
        )

    z_emb = out["z_emb"]
    z_emb_proj = out["z_emb_proj"]
    z_txt = out["z_txt"]
    pred_ids = out["pred_ids"]

    print(
        f"z_emb: {tuple(z_emb.shape)}, "
        f"z_emb_proj: {tuple(z_emb_proj.shape)}, "
        f"z_txt: {tuple(z_txt.shape)}, "
        f"pred_ids: {tuple(pred_ids.shape)}"
    )
