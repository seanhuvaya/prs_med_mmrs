from typing import Optional

import torch
from peft import LoraConfig, get_peft_model, TaskType

from .llava_med_mllm import LLavaMedMLLM, DEFAULT_MODEL_NAME


class LLavaMedWithLoRA(LLavaMedMLLM):
    """
    LLaVA-Med MLLM with LoRA adapters.

    - Uses the parent LLavaMedMLLM for forward/QA/generation logic.
    - By default, base LLM weights are frozen and only LoRA + projection head
      are trainable, which is what you typically want for joint training
      with a segmentation head.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        rank: int = 16,
        alpha: int = 16,
        dropout: float = 0.05,
        freeze_llm: bool = True,
        device: Optional[str] = None,
        use_8bit: bool = False,
        use_4bit: bool = False,
        dtype: Optional[torch.dtype] = None,
        max_new_tokens: int = 64,
    ):
        super().__init__(
            model_name=model_name,
            device=device,
            use_8bit=use_8bit,
            use_4bit=use_4bit,
            dtype=dtype,
            freeze_llm=freeze_llm,
            max_new_tokens=max_new_tokens,
        )

        # LoRA configuration
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

        # Wrap the base model with LoRA
        self.model = get_peft_model(self.model, self.lora_config)

        # Make sure LoRA parameters are trainable
        for name, param in self.model.named_parameters():
            if "lora_" in name:
                param.requires_grad = True

        self.model.print_trainable_parameters()


if __name__ == "__main__":
    from PIL import Image

    mllm = LLavaMedWithLoRA(
        rank=16,
        alpha=16,
        dropout=0.05,
        freeze_llm=True,  # base frozen, LoRA + projection head trainable
    )
    mllm.eval()

    img = Image.new("RGB", (512, 512), color=(128, 128, 128))
    batch_images = [img, img]
    batch_questions = [
        "Where is the tumor located?",
        "Describe the position of the lesion.",
    ]
    batch_answers = [
        "The tumor is in the upper left region.",
        "The lesion is in the right lower lobe.",
    ]

    with torch.no_grad():
        out = mllm(
            batch_images,
            batch_questions,
            answers=batch_answers,
            compute_lm_loss=True,
            return_projected=True,
        )

    z_emb = out["z_emb"]
    z_emb_proj = out["z_emb_proj"]
    z_txt = out["z_txt"]
    pred_ids = out["pred_ids"]
    lm_loss = out["lm_loss"]

    print(
        f"z_emb: {tuple(z_emb.shape)}, "
        f"z_emb_proj: {tuple(z_emb_proj.shape)}, "
        f"z_txt: {tuple(z_txt.shape)}, "
        f"pred_ids: {tuple(pred_ids.shape)}, "
        f"lm_loss: {lm_loss}"
    )

    # Save LoRA-augmented model (for later loading)
    mllm.model.save_pretrained("checkpoints/llava_med_lora_test")
    print("Saved model to checkpoints/llava_med_lora_test")
