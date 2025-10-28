import torch
import torch.nn as nn

from peft import get_peft_model, LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

class MultimodalTextEncoder(nn.Module):
    def __init__(self, base_model: str = "microsoft/DialoGPT-medium", apply_lora: bool = True):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        # Add padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(base_model)
        if apply_lora:
            lora_config = LoraConfig(r=16, lora_alpha=16,  lora_dropout=0.05, target_modules=["c_attn", "c_proj"])
            self.model = get_peft_model(self.model, lora_config)

    def forward(self, input_text: list[str]) -> torch.Tensor:
        inputs = self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=32)
        outputs = self.model.model(**{k: v.to(self.model.device) for k, v in inputs.items()}, output_hidden_states=True)
        z_emb = outputs.hidden_states[-1]  # Shape: [batch, seq_len, hidden_dim]
        return z_emb
