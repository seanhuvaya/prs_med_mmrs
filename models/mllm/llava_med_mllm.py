import os
from typing import List, Union, Dict, Any, Optional

import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

DEFAULT_MODEL_NAME = "chaoyinshe/llava-med-v1.5-mistral-7b-hf"

class LLavaMedMLLM(nn.Module):
    """
    Multimodal LLM (MLLM) + Reasoning Head using LLava-Med.
    """
    def __init__(
        self, 
        model_name: str = DEFAULT_MODEL_NAME, 
        device: Optional[str] = None,
        use_8bit: bool = False,
        use_4bit: bool = False,
        dtype: Optional[torch.dtype] = None,
        freeze_llm: bool = True,
        max_new_tokens: int = 0, # we just need logits/hidden states, not generating
    ):
        super().__init__()

        self.device = device or ("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        if dtype is None:
            dtype = torch.float16 if "cuda" in self.device else torch.float32

        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

        load_model_kwargs: Dict[str, Any] = dict(
            trust_remote_code=True,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            output_hidden_states=True,
        )

        if "cuda" in self.device:
            load_model_kwargs.update(dict(device_map="auto"))
            if use_8bit:
                load_model_kwargs.update(dict(load_in_8bit=True))
            elif use_4bit:
                load_model_kwargs.update(dict(load_in_4bit=True))

        # Use LlavaForConditionalGeneration instead of AutoModelForCausalLM
        self.model = LlavaForConditionalGeneration.from_pretrained(model_name, **load_model_kwargs)
        
        # Move to device if not using device_map="auto"
        if "device_map" not in load_model_kwargs:
            self.model.to(self.device)

        if freeze_llm:
            for param in self.model.parameters():
                param.requires_grad = False

        self.max_new_tokens = max_new_tokens
        
        # Hidden size (LLava-Med Mistral=4096)
        self.hidden_size = getattr(self.model.config, "hidden_size", 4096)

        # A small projector from 4096 -> 256 for seg conditioning
        self.to_seg_channels = nn.Sequential(
            nn.Linear(self.hidden_size, 1024),
            nn.GELU(),
            nn.Linear(1024, 256),
        )
        
        # Move projection head to the same device as the model
        self.to_seg_channels.to(self.device)
        
        for param in self.to_seg_channels.parameters():
            param.requires_grad = True

    def _ensure_images(self, images: List[Union[Image.Image, torch.Tensor]]) -> List[Image.Image]:
        output = []
        for img in images:
            if isinstance(img, Image.Image):
                output.append(img.convert("RGB"))
            elif isinstance(img, torch.Tensor):
                t = img.detach().cpu()
                if t.dim() == 3 and t.shape[0] in (1, 3):
                    if t.max() > 1.0:
                        t = t / 255.0
                    t = (t.clamp(0, 1) * 255).byte()
                    if t.shape[0] == 1:
                        t = t.repeat(3, 1, 1)
                    output.append(Image.fromarray(t.permute(1, 2, 0).numpy()))
                else:
                    raise ValueError(f"Tensor image must be CHW with C in {1, 3}: {t.shape}")
            else:
                raise ValueError(f"Images must be PIL.Image or 3xHxW Tensors")
        return output

    def forward(
        self,
        images: List[Union[Image.Image, torch.Tensor]],
        questions: List[str],
        return_projected: bool = True,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        """
        Returns:
            z_emb: (B, L, 4096)
            z_txt: (B, L, vocab)
            pred_ids: (B, L) argmax over logits
            z_emb_proj: (B, L, 256) if return_projected
        """
        assert len(images) == len(questions), "Batch size mismatch between images and questions"

        images = self._ensure_images(images)

        # Format prompts for LLaVA-Med
        formatted_prompts = []
        for question in questions:
            formatted_prompts.append(f"USER: <image>\n{question}\nASSISTANT:")
        
        inputs = self.processor(
            images=images, 
            text=formatted_prompts, 
            return_tensors="pt", 
            padding=True
        ).to(self.device)

        with torch.set_grad_enabled(self.training):
            outputs = self.model(**inputs, output_hidden_states=True)

        # Get hidden states - LLaVA returns tuple of hidden states for each layer
        if hasattr(outputs, 'hidden_states'):
            z_emb = outputs.hidden_states[-1]  # Last layer hidden states
        else:
            # Fallback: get encoder hidden states if available
            z_emb = outputs.encoder_hidden_states[-1] if hasattr(outputs, 'encoder_hidden_states') else None

        z_txt = outputs.logits

        pred_ids = torch.argmax(z_txt, dim=-1)

        out = {"z_emb": z_emb, "z_txt": z_txt, "pred_ids": pred_ids, "inputs": inputs}

        if return_projected and z_emb is not None:
            z_emb_proj = self.to_seg_channels(z_emb)
            out["z_emb_proj"] = z_emb_proj

        return out


if __name__ == "__main__":
    img = Image.new("RGB", (512, 512), color=(128, 128, 128))
    batch_images = [img, img]
    batch_questions = ["Where is the tumor located?", "Describe the position of the lesion."]

    mllm = LLavaMedMLLM(
        freeze_llm=True,
    )
    mllm.eval()

    with torch.no_grad():
        out = mllm(batch_images, batch_questions, return_projected=True)
    
    z_emb = out["z_emb"]
    z_emb_proj = out["z_emb_proj"]
    z_txt = out["z_txt"]
    pred_ids = out["pred_ids"]

    print(
        f"z_emb: {tuple(z_emb.shape) if z_emb is not None else 'None'}, "
        f"z_emb_proj: {tuple(z_emb_proj.shape) if 'z_emb_proj' in out else 'None'}, "
        f"z_txt: {tuple(z_txt.shape)}, "
        f"pred_ids: {tuple(pred_ids.shape)}"
    )