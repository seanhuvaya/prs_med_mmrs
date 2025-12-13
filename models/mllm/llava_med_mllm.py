import os
from typing import List, Union, Dict, Any, Optional

import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

DEFAULT_MODEL_NAME = "chaoyinshe/llava-med-v1.5-mistral-7b-hf"


class LLavaMedMLLM(nn.Module):
    """
    Multimodal LLM (MLLM) wrapper around LLaVA-Med.

    Supports:
      - Extracting hidden states z_emb and logits z_txt for fusion & text loss
      - Using either question-only prompts (inference) or question+answer
        full text (training text branch for CE loss, Eq. (7) in PRS-Med).
    """
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: Optional[str] = None,
        use_8bit: bool = False,
        use_4bit: bool = False,
        dtype: Optional[torch.dtype] = None,
        freeze_llm: bool = True,
        max_new_tokens: int = 0,
        training_texts: bool = True,
    ):
        super().__init__()

        # Device & dtype
        self.device = device or (
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available()
            else "cpu"
        )
        if dtype is None:
            dtype = torch.float16 if "cuda" in self.device else torch.float32

        # Processor & model
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

        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_name, **load_model_kwargs
        )

        # If no device_map, move model
        if "device_map" not in load_model_kwargs:
            self.model.to(self.device)

        # Optionally freeze base LLM (LoRA will re-enable grads on adapters)
        if freeze_llm:
            for p in self.model.parameters():
                p.requires_grad = False

        self.max_new_tokens = max_new_tokens

        # Hidden size (LLaVA-Med Mistral = 4096 typically)
        self.hidden_size = getattr(self.model.config, "hidden_size", 4096)

        # Projector: 4096 -> 256 for seg conditioning (Eq. (3) context)
        self.to_seg_channels = nn.Sequential(
            nn.Linear(self.hidden_size, 1024),
            nn.GELU(),
            nn.Linear(1024, 256),
        ).to(self.device)

        for p in self.to_seg_channels.parameters():
            p.requires_grad = True

        self.training_texts = training_texts

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _ensure_images(
        self,
        images: List[Union[Image.Image, torch.Tensor]]
    ) -> List[Image.Image]:
        output = []
        for img in images:
            if isinstance(img, Image.Image):
                output.append(img.convert("RGB"))
            elif isinstance(img, torch.Tensor):
                t = img.detach().cpu()
                # Expect CHW
                if t.dim() == 3 and t.shape[0] in (1, 3):
                    if t.max() > 1.0:
                        t = t / 255.0
                    t = (t.clamp(0, 1) * 255).byte()
                    if t.shape[0] == 1:
                        t = t.repeat(3, 1, 1)
                    output.append(Image.fromarray(t.permute(1, 2, 0).numpy()))
                else:
                    raise ValueError(f"Tensor image must be CHW with C in {{1, 3}}: got {t.shape}")
            else:
                raise ValueError("Images must be PIL.Image.Image or 3xHxW torch.Tensor")
        return output

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #
    def forward(
        self,
        images: List[Union[Image.Image, torch.Tensor]],
        questions: List[str],
        training_texts: Optional[bool] = True,
        answers: Optional[List[str]] = None,
        return_projected: bool = True
    ):
        """
        Args:
            images:        list[PIL.Image or 3xHxW tensor]
            questions:     list[str]
            answers:       list[str] (only used when training_texts=True)
            training_texts: if True, use full question+answer text
                           for CE loss (Eq. 7). If False, question-only.

        Returns:
            dict with:
              z_emb       (B, L, hidden_size)
              z_txt       (B, L, vocab_size)
              pred_ids    (B, L)
              z_emb_proj  (B, L, 256) if return_projected
              inputs      processor inputs
        """
        assert len(images) == len(questions), "Batch size mismatch between images and questions"

        images = self._ensure_images(images)
        # training_texts flag:
        # - If an explicit flag is passed, use it.
        # - Otherwise, fall back to the instance default (self.training_texts).
        effective_training_texts = (
            self.training_texts if training_texts is None else training_texts
        )

        # ---------- Build text inputs ---------- #
        if effective_training_texts:
            # Question + answer text (X_txt for Eq. (1), Eq. (7))
            assert answers is not None, "answers must be provided when training_texts=True"
            assert len(answers) == len(questions), "questions and answers must match in batch size"

            texts_full = [
                f"USER: <image>\n{q}\nASSISTANT: {a}"
                for q, a in zip(questions, answers)
            ]
            inputs = self.processor(
                images=images,
                text=texts_full,
                return_tensors="pt",
                padding=True,
            )
        else:
            # Question-only prompts (inference / feature mode)
            texts_q = [
                f"USER: <image>\n{q}\nASSISTANT:"
                for q in questions
            ]
            inputs = self.processor(
                images=images,
                text=texts_q,
                return_tensors="pt",
                padding=True,
            )

        # Move to device & fix dtypes
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        model_dtype = next(self.model.parameters()).dtype
        for k, v in inputs.items():
            if torch.is_floating_point(v):
                inputs[k] = v.to(dtype=model_dtype)

        # Forward pass through LLaVA-Med
        outputs = self.model(
            **inputs,
            output_hidden_states=True,
        )

        # Hidden states and logits
        if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            z_emb = outputs.hidden_states[-1]  # (B, L, hidden_size)
        else:
            z_emb = None

        z_txt = outputs.logits  # (B, L, vocab_size)
        pred_ids = torch.argmax(z_txt, dim=-1)

        out = {
            "z_emb": z_emb,
            "z_txt": z_txt,
            "pred_ids": pred_ids,
            "inputs": inputs,
        }

        # Projection for seg conditioning
        if return_projected and z_emb is not None:
            proj_dtype = next(self.to_seg_channels.parameters()).dtype
            if z_emb.dtype != proj_dtype:
                z_emb = z_emb.to(proj_dtype)
            z_emb_proj = self.to_seg_channels(z_emb)
            out["z_emb_proj"] = z_emb_proj

        return out

    # ------------------------------------------------------------------ #
    # Inference helper (actual text generation)
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def generate_answers(
        self,
        images: List[Union[Image.Image, torch.Tensor]],
        questions: List[str],
        max_new_tokens: int = 48,
        temperature: float = 0.2,
        top_p: float = 0.8,
        top_k: int = 50,
        repetition_penalty: float = 1.2,
        no_repeat_ngram_size: int = 4,
    ) -> List[str]:
        """
        Generate free-form answers for question-only prompts.

        Args:
            images: list of images (PIL or tensors)
            questions: list of questions (same length as images)
            max_new_tokens: length of generated answer
            temperature: >0 enables sampling; 0 = greedy
        """
        images = self._ensure_images(images)

        tokenizer = self.processor.tokenizer
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        answers: List[str] = []

        # Process each sample individually to respect chat template + image pairing
        for img, q in zip(images, questions):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": q},
                    ],
                }
            ]

            prompt = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )

            inputs = self.processor(
                text=[prompt],
                images=[img],
                return_tensors="pt",
                padding=True,
            )

            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            model_dtype = next(self.model.parameters()).dtype
            for k, v in inputs.items():
                if torch.is_floating_point(v):
                    inputs[k] = v.to(dtype=model_dtype)

            generation = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-5) if temperature > 0 else None,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                length_penalty=1.0,
                early_stopping=True,
            )

            # Remove the prompt portion
            prompt_len = inputs["input_ids"].shape[1]
            gen_only = generation[:, prompt_len:]
            text = self.processor.batch_decode(gen_only, skip_special_tokens=True)[0]
            answers.append(text.strip())

        return answers


if __name__ == "__main__":
    # Quick sanity test
    img = Image.new("RGB", (512, 512), color=(128, 128, 128))
    batch_images = [img, img]
    batch_questions = ["Where is the tumor located?", "Describe the position of the lesion."]
    batch_answers = ["In the left lobe.", "In the upper right lung."]

    mllm = LLavaMedMLLM(freeze_llm=True)
    mllm.eval()

    with torch.no_grad():
        out = mllm(
            batch_images,
            batch_questions,
            answers=batch_answers,
            training_texts=True,
            return_projected=True,
        )

    z_emb = out["z_emb"]
    z_emb_proj = out.get("z_emb_proj", None)
    z_txt = out["z_txt"]
    pred_ids = out["pred_ids"]

    print(
        f"z_emb: {tuple(z_emb.shape) if z_emb is not None else 'None'}, "
        f"z_emb_proj: {tuple(z_emb_proj.shape) if z_emb_proj is not None else 'None'}, "
        f"z_txt: {tuple(z_txt.shape)}, "
        f"pred_ids: {tuple(pred_ids.shape)}"
    )
