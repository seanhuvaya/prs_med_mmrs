from typing import List, Union, Dict, Any, Optional

import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

DEFAULT_MODEL_NAME = "chaoyinshe/llava-med-v1.5-mistral-7b-hf"


class LLavaMedMLLM(nn.Module):
    """
    Multimodal LLM (MLLM) + projection head using LLaVA-Med.

    - Can be used purely as an encoder to produce hidden states for segmentation.
    - Can optionally compute a language modeling (QA) loss when answers are provided.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: Optional[str] = None,
        use_8bit: bool = False,
        use_4bit: bool = False,
        dtype: Optional[torch.dtype] = None,
        freeze_llm: bool = True,
        max_new_tokens: int = 64,  # used for generate_answers
    ):
        super().__init__()

        # ---- Device & dtype -------------------------------------------------
        self.device = device or (
            "mps"
            if torch.backends.mps.is_available()
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
        if dtype is None:
            dtype = torch.float16 if "cuda" in self.device else torch.float32

        # ---- Processor / tokenizer ------------------------------------------
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        # ---- Model loading ---------------------------------------------------
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
            model_name,
            **load_model_kwargs,
        )

        # If no device_map was used, move to single device
        if "device_map" not in load_model_kwargs:
            self.model.to(self.device)

        # Optionally freeze all base LLM parameters (for LoRA-only training, etc.)
        if freeze_llm:
            for param in self.model.parameters():
                param.requires_grad = False

        self.max_new_tokens = max_new_tokens

        # Hidden size (LLaVA-Med Mistral = 4096)
        self.hidden_size = getattr(self.model.config, "hidden_size", 4096)

        # Projection head from 4096 -> 256 for segmentation conditioning
        self.to_seg_channels = nn.Sequential(
            nn.Linear(self.hidden_size, 1024),
            nn.GELU(),
            nn.Linear(1024, 256),
        ).to(self.device)

        # By default, we want this head trainable
        for p in self.to_seg_channels.parameters():
            p.requires_grad = True

    # --------------------------------------------------------------------- #
    # Image utilities
    # --------------------------------------------------------------------- #
    def _ensure_images(
        self,
        images: List[Union[Image.Image, torch.Tensor]],
    ) -> List[Image.Image]:
        """
        Normalize images to a list of RGB PIL.Image objects.
        """
        output: List[Image.Image] = []
        for img in images:
            if isinstance(img, Image.Image):
                output.append(img.convert("RGB"))
            elif isinstance(img, torch.Tensor):
                t = img.detach().cpu()
                if t.dim() == 3 and t.shape[0] in (1, 3):
                    # CHW -> uint8 RGB
                    if t.max() > 1.0:
                        t = t / 255.0
                    t = (t.clamp(0, 1) * 255).byte()
                    if t.shape[0] == 1:
                        t = t.repeat(3, 1, 1)
                    output.append(Image.fromarray(t.permute(1, 2, 0).numpy()))
                else:
                    raise ValueError(
                        f"Tensor image must be CHW with C in {{1, 3}}: got {t.shape}"
                    )
            else:
                raise ValueError("Images must be PIL.Image or 3xHxW Tensors")
        return output

    # --------------------------------------------------------------------- #
    # Forward: embeddings + optional LM loss
    # --------------------------------------------------------------------- #
    def forward(
        self,
        images: List[Union[Image.Image, torch.Tensor]],
        questions: List[str],
        answers: Optional[List[str]] = None,
        *,
        return_projected: bool = True,
        compute_lm_loss: bool = False,
    ) -> Dict[str, Any]:
        """
        Forward pass.

        Args:
            images: List of images (PIL or 3xHxW tensors).
            questions: List of question strings.
            answers: Optional list of ground-truth answer strings.
            return_projected: If True, also returns z_emb_proj (for segmentation).
            compute_lm_loss: If True and answers is not None, compute LM loss
                             only on the answer tokens (question tokens masked
                             with -100 in labels).

        Returns:
            Dictionary with keys:
                - z_emb:        (B, L, hidden_size) last-layer hidden states
                - z_emb_proj:   (B, L, 256) projected features (if return_projected)
                - z_txt:        (B, L, vocab_size) logits
                - pred_ids:     (B, L) argmax over logits (teacher-forced)
                - lm_loss:      scalar LM loss (or None if not computed)
        """
        assert len(images) == len(
            questions
        ), "Batch size mismatch between images and questions"
        if compute_lm_loss and answers is not None:
            assert len(answers) == len(
                questions
            ), "Batch size mismatch between questions and answers"

        images = self._ensure_images(images)

        # Template used for LLaVA-Med
        prefixes = [
            f"USER: <image>\n{q}\nASSISTANT:" for q in questions
        ]

        # ================================================================== #
        # Case 1: QA training (use question + answer, compute LM loss)
        # ================================================================== #
        if compute_lm_loss and answers is not None:
            # Full texts: question + answer
            full_texts = [
                f"USER: <image>\n{q}\nASSISTANT: {a}"
                for q, a in zip(questions, answers)
            ]

            # Tokenize full sequences (for forward pass)
            batch_full = self.processor(
                images=images,
                text=full_texts,
                return_tensors="pt",
                padding=True,
            )

            # Tokenize prefixes (question + "ASSISTANT:") to know prefix length
            batch_prefix = self.processor(
                images=images,
                text=prefixes,
                return_tensors="pt",
                padding=True,
            )

            # Move to device
            batch_full = {k: v.to(self.device) for k, v in batch_full.items()}
            batch_prefix = {k: v.to(self.device) for k, v in batch_prefix.items()}

            input_ids = batch_full["input_ids"]          # (B, L_full)
            attention_mask = batch_full["attention_mask"]
            pixel_values = batch_full["pixel_values"]

            # Prepare labels: same shape as input_ids, but -100 for question tokens
            labels = input_ids.clone()
            labels.fill_(-100)

            with torch.no_grad():
                # prefix_mask: (B, L_full) with 1 where token belongs to prefix
                prefix_mask = batch_prefix["attention_mask"]

                # prefix length per sample (number of non-pad tokens)
                prefix_lengths = prefix_mask.sum(dim=1)  # (B,)

            for i in range(input_ids.size(0)):
                prefix_len = int(prefix_lengths[i].item())
                # Everything after prefix_len is answer; keep those labels
                labels[i, prefix_len:] = input_ids[i, prefix_len:]

            # Final input dict to the model
            inputs: Dict[str, Any] = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "labels": labels,
            }

        # ================================================================== #
        # Case 2: encoder-style forward (no LM loss, question only)
        # ================================================================== #
        else:
            batch = self.processor(
                images=images,
                text=prefixes,
                return_tensors="pt",
                padding=True,
            )
            inputs = {k: v.to(self.device) for k, v in batch.items()}

        # Ensure floating tensors match model dtype
        model_dtype = next(self.model.parameters()).dtype
        for k, v in inputs.items():
            if torch.is_floating_point(v):
                inputs[k] = v.to(dtype=model_dtype)

        # Forward pass
        with torch.set_grad_enabled(self.training):
            outputs = self.model(**inputs, output_hidden_states=True)

        # Last layer hidden states
        z_emb = outputs.hidden_states[-1]  # (B, L, hidden_size)
        z_txt = outputs.logits             # (B, L, vocab_size)
        pred_ids = torch.argmax(z_txt, dim=-1)  # (B, L)

        lm_loss = outputs.loss if (compute_lm_loss and answers is not None) else None

        out: Dict[str, Any] = {
            "z_emb": z_emb,
            "z_txt": z_txt,
            "pred_ids": pred_ids,
            "lm_loss": lm_loss,
        }

        # Project to seg-conditioning channels (B, L, 256)
        if return_projected and z_emb is not None:
            if z_emb.dtype != next(self.to_seg_channels.parameters()).dtype:
                z_emb = z_emb.to(next(self.to_seg_channels.parameters()).dtype)
            z_emb_proj = self.to_seg_channels(z_emb)
            out["z_emb_proj"] = z_emb_proj

        return out

    # --------------------------------------------------------------------- #
    # Generation helper (for evaluation / CSV predictions)
    # --------------------------------------------------------------------- #
    @torch.no_grad()
    def generate_answers(
        self,
        images: List[Union[Image.Image, torch.Tensor]],
        questions: List[str],
        *,
        max_new_tokens: Optional[int] = None,
        do_sample: bool = False,
        **generate_kwargs: Any,
    ) -> List[str]:
        """
        Autoregressively generate answers given (image, question).

        Returns a list of decoded strings. You probably want to post-process
        them to extract only the assistant's answer part.
        """
        self.eval()
        assert len(images) == len(questions)

        images = self._ensure_images(images)
        prefixes = [
            f"USER: <image>\n{q}\nASSISTANT:" for q in questions
        ]

        batch = self.processor(
            images=images,
            text=prefixes,
            return_tensors="pt",
            padding=True,
        )

        batch = {k: v.to(self.device) for k, v in batch.items()}
        model_dtype = next(self.model.parameters()).dtype
        for k, v in batch.items():
            if torch.is_floating_point(v):
                batch[k] = v.to(dtype=model_dtype)

        if max_new_tokens is None or max_new_tokens <= 0:
            max_new_tokens = self.max_new_tokens or 64

        gen_ids = self.model.generate(
            **batch,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            **generate_kwargs,
        )

        texts = self.processor.batch_decode(gen_ids, skip_special_tokens=True)

        return texts


if __name__ == "__main__":
    # Simple smoke test
    img = Image.new("RGB", (512, 512), color=(128, 128, 128))
    batch_images = [img, img]
    batch_questions = [
        "Where is the tumor located?",
        "Describe the position of the lesion.",
    ]
    batch_answers = [
        "The tumor is located in the upper left region.",
        "The lesion is in the central area of the lung.",
    ]

    mllm = LLavaMedMLLM(
        freeze_llm=True,
    )
    mllm.eval()

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

    # Test generation
    gen = mllm.generate_answers(batch_images, batch_questions, max_new_tokens=32)
    print("Generated:", gen)
