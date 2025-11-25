import os
from typing import List, Union, Dict, Any, Optional

import torch
import torch.nn as nn
from PIL import Image
from utils import logging as _logging
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
        # Paper-alignment controls
        paper_preset: bool = False,
        prompt_template: Optional[str] = None,
        hidden_state_layer: Union[int, str] = "last",
        visual_pooling: str = "none",  # one of: 'none' | 'mean' | 'cls'
    ):
        super().__init__()

        self._logger = _logging.get_logger(__name__)
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        if dtype is None:
            dtype = torch.float16 if "cuda" in self.device else torch.float32

        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

        # Configure prompt template
        # Default LLaVA-style chat template used by many LLaVA-Med examples
        default_template = "USER: <image>\n{question}\nASSISTANT:"
        # Defaults that can be overridden by the "paper_preset"
        if isinstance(hidden_state_layer, str) and hidden_state_layer != "last":
            raise ValueError("hidden_state_layer must be int or 'last'")
        if visual_pooling not in {"none", "mean", "cls"}:
            raise ValueError("visual_pooling must be one of {'none','mean','cls'}")

        # Apply preset overrides if requested (paper 2505.11872 compliance)
        if paper_preset:
            # Prompt format used in most LLaVA-Med instruction templates
            # Keep explicit <image> sentinel and chat roles
            self.prompt_template = prompt_template or "USER: <image>\n{question}\nASSISTANT:"
            # Enforce paper-style choices: last hidden state, CLS pooling
            hidden_state_layer = "last" if isinstance(hidden_state_layer, str) else hidden_state_layer
            visual_pooling = "cls"
        else:
            self.prompt_template = prompt_template or default_template

        self.hidden_state_layer = hidden_state_layer
        self.visual_pooling = visual_pooling

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

        # Projectors (paper-aligned: MLP with LayerNorm + GELU + Dropout)
        proj_hidden = 1024
        proj_out = 256
        dropout_p = 0.1 if paper_preset else 0.0

        self.to_seg_channels = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, proj_hidden),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(proj_hidden, proj_out),
        )
        # A separate projector for pooled embeddings (B, H)
        self.to_seg_channels_pooled = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, proj_hidden),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(proj_hidden, proj_out),
        )
        
        # Move projection head to the same device as the model
        self.to_seg_channels.to(self.device)
        self.to_seg_channels_pooled.to(self.device)
        
        for param in self.to_seg_channels.parameters():
            param.requires_grad = True
        for param in self.to_seg_channels_pooled.parameters():
            param.requires_grad = True

    @staticmethod
    def _ensure_images(images: List[Union[Image.Image, torch.Tensor]]) -> List[Image.Image]:
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

        # Format prompts for LLaVA-Med (paper-aligned template if enabled)
        formatted_prompts = []
        for question in questions:
            formatted_prompts.append(self.prompt_template.format(question=question))

        # Optional: log prompt format in paper preset for verifiability (first item only)
        if len(formatted_prompts) > 0 and "<image>" not in formatted_prompts[0]:
            self._logger.warning("Prompt does not contain <image> token; check paper/template compliance.")
        if formatted_prompts and "USER:" not in formatted_prompts[0]:
            self._logger.debug("Using non-chat style prompt template.")
        
        inputs = self.processor(
            images=images, 
            text=formatted_prompts, 
            return_tensors="pt", 
            padding=True
        ).to(self.device)
        
        # FIX: Convert inputs to match model dtype
        model_dtype = next(self.model.parameters()).dtype
        inputs = {k: v.to(dtype=model_dtype) if v.dtype.is_floating_point else v 
                for k, v in inputs.items()}

        with torch.set_grad_enabled(self.training):
            outputs = self.model(**inputs, output_hidden_states=True)

        # Get hidden states - LLaVA returns tuple of hidden states for each layer
        z_emb = None
        if hasattr(outputs, 'hidden_states'):
            hs = outputs.hidden_states
            if isinstance(self.hidden_state_layer, int):
                # allow negative indexing
                z_emb = hs[self.hidden_state_layer]
            else:
                # 'last'
                z_emb = hs[-1]
        else:
            # Fallback: get encoder hidden states if available
            z_emb = outputs.encoder_hidden_states[-1] if hasattr(outputs, 'encoder_hidden_states') else None

        z_txt = outputs.logits

        pred_ids = torch.argmax(z_txt, dim=-1)

        out: Dict[str, Any] = {"z_emb": z_emb, "z_txt": z_txt, "pred_ids": pred_ids, "inputs": inputs}

        # Visual pooling if requested (works on token dimension: (B, L, H))
        if z_emb is not None and self.visual_pooling != "none":
            if self.visual_pooling == "mean":
                z_pooled = z_emb.mean(dim=1)
            elif self.visual_pooling == "cls":
                # Use the first token as CLS-equivalent
                z_pooled = z_emb[:, 0, :]
            else:
                raise ValueError(f"Unknown visual_pooling: {self.visual_pooling}")
            out["z_emb_pooled"] = z_pooled

        if return_projected and z_emb is not None:
            # FIX: Ensure z_emb matches the projection layer dtype
            if z_emb.dtype != next(self.to_seg_channels.parameters()).dtype:
                z_emb = z_emb.to(next(self.to_seg_channels.parameters()).dtype)
            z_emb_proj = self.to_seg_channels(z_emb)
            out["z_emb_proj"] = z_emb_proj

            # Also project pooled if available
            if "z_emb_pooled" in out:
                z_pooled = out["z_emb_pooled"]
                if z_pooled.dtype != next(self.to_seg_channels.parameters()).dtype:
                    z_pooled = z_pooled.to(next(self.to_seg_channels.parameters()).dtype)
                out["z_emb_proj_pooled"] = self.to_seg_channels_pooled(z_pooled)

        return out

    # ---- Checkpoint helpers to ensure LoRA + projectors are saved together ----
    def save_pretrained(self, save_directory: str) -> None:
        """
        Save the underlying Transformer (or PEFT adapter if present) and the
        additional projector heads defined in this wrapper so finetuning can be
        resumed correctly.

        This method intentionally goes beyond calling model.save_pretrained to
        also persist the projector modules (to_seg_channels, to_seg_channels_pooled)
        which are not part of the base Hugging Face model.
        """
        os.makedirs(save_directory, exist_ok=True)

        # 1) Save processor for consistent preprocessing
        try:
            self.processor.save_pretrained(save_directory)
        except Exception:
            pass

        # 2) Save the base/peft model
        try:
            # If it's a PEFT model, this stores only adapter weights (desired)
            self.model.save_pretrained(save_directory)
        except Exception as e:
            raise RuntimeError(f"Failed to save model/adapter to {save_directory}: {e}")

        # 3) Save projector heads
        projector_path = os.path.join(save_directory, "projectors.pt")
        torch.save({
            "to_seg_channels": self.to_seg_channels.state_dict(),
            "to_seg_channels_pooled": self.to_seg_channels_pooled.state_dict(),
        }, projector_path)

        # 4) Save basic wrapper config for reproducibility
        try:
            import json
            cfg = {
                "paper_preset": getattr(self, "visual_pooling", None) == "cls",  # heuristic
                "prompt_template": self.prompt_template,
                "hidden_state_layer": self.hidden_state_layer,
                "visual_pooling": self.visual_pooling,
                "hidden_size": self.hidden_size,
            }
            with open(os.path.join(save_directory, "prs_med_mllm_config.json"), "w") as f:
                json.dump(cfg, f)
        except Exception:
            pass

    @classmethod
    def from_pretrained(
        cls,
        load_directory: str,
        device: Optional[str] = None,
        freeze_llm: bool = True,
        **kwargs,
    ) -> "LLavaMedMLLM":
        """
        Load the base LLaVA-Med model and then restore projector heads from the
        provided directory. This complements save_pretrained.
        """
        # Instantiate fresh with defaults (will download base weights if needed)
        model = cls(device=device, freeze_llm=freeze_llm, **kwargs)

        # Try to load processor if available (not critical at inference)
        try:
            processor = AutoProcessor.from_pretrained(load_directory, trust_remote_code=True)
            model.processor = processor
        except Exception:
            pass

        # Attempt to load PEFT adapter or full model weights
        try:
            # Prefer PEFT adapter load if directory contains adapter_config.json
            adapter_config_path = os.path.join(load_directory, "adapter_config.json")
            if os.path.exists(adapter_config_path):
                from peft import PeftModel
                model.model = PeftModel.from_pretrained(model.model, load_directory)
                model.model.to(model.device)
        except Exception:
            # Fallback: ignore and keep the base model
            pass

        # Restore projector heads
        projector_path = os.path.join(load_directory, "projectors.pt")
        if os.path.exists(projector_path):
            state = torch.load(projector_path, map_location=model.device)
            model.to_seg_channels.load_state_dict(state.get("to_seg_channels", {}))
            model.to_seg_channels_pooled.load_state_dict(state.get("to_seg_channels_pooled", {}))

        # Restore wrapper config if present
        cfg_path = os.path.join(load_directory, "prs_med_mllm_config.json")
        if os.path.exists(cfg_path):
            try:
                import json
                with open(cfg_path, "r") as f:
                    cfg = json.load(f)
                model.prompt_template = cfg.get("prompt_template", model.prompt_template)
                model.hidden_state_layer = cfg.get("hidden_state_layer", model.hidden_state_layer)
                model.visual_pooling = cfg.get("visual_pooling", model.visual_pooling)
            except Exception:
                pass

        return model


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