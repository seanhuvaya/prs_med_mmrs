import torch
import torch.nn as nn
from transformers import AutoProcessor, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

class MLLMWithLoRA(nn.Module):
    """
    Wrap an MLLM (e.g., LLaVA-Med) to:
    - produce token embeddings (last hidden states) conditioned on image+question
    - generate text logits for reasoning loss
    Trains LoRA adapters only (per paper) to inject position reasoning efficiently.  # :contentReference[oaicite:8]{index=8}
    """
    def __init__(self, model_id: str, lora_cfg: dict, device="cuda"):
        super().__init__()
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        # LoRA
        lora = LoraConfig(
            r=lora_cfg["r"],
            lora_alpha=lora_cfg["alpha"],
            target_modules=lora_cfg["target_modules"],
            lora_dropout=lora_cfg["dropout"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, lora)
        self.model.print_trainable_parameters()

    def tokenize(self, question: str, images):
        # Processor handles multimodal packing for LLaVA-style models.
        return self.processor(images=images, text=question, return_tensors="pt")

    @torch.no_grad()
    def generate(self, pixel_values, question: str, gen_cfg: dict):
        inputs = self.processor(text=question, images=[pixel_values], return_tensors="pt").to(pixel_values.device, dtype=pixel_values.dtype)
        out = self.model.generate(**inputs, max_new_tokens=gen_cfg["max_new_tokens"], temperature=gen_cfg["temperature"])
        return self.processor.batch_decode(out, skip_special_tokens=True)[0]

    def forward(self, pixel_values, question: str):
        """
        Returns:
          last_hidden_states: [B, L, D] token embeddings (conditioning)
          logits: LM logits for text (for CE loss)
        """
        B = pixel_values.size(0)
        texts = [question] * B
        proc = self.processor(images=[p for p in pixel_values], text=texts, return_tensors="pt", padding=True).to(pixel_values.device, dtype=pixel_values.dtype)
        out = self.model(**proc, output_hidden_states=True)
        last_hidden = out.hidden_states[-1]  # [B,L,D]
        return last_hidden, out.logits
