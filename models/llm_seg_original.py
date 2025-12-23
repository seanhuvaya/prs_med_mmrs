import os
import math
import torch
import torch.nn as nn
import logging
from peft import LoraConfig, TaskType, get_peft_model, PeftModel

from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from tinysam import sam_model_registry

from .decoder.mask_decoder_original import PromptedMaskDecoder
from .vision_backbone.sam_med2d_encoder import SAMMed2DVisionBackbone


def custom_lora_init(module):
    if hasattr(module, "lora_A"):
        nn.init.kaiming_uniform_(module.lora_A.weight, a=math.sqrt(5))
    if hasattr(module, "lora_B"):
        nn.init.zeros_(module.lora_B.weight)


class ImageEncoder(nn.Module):
    def __init__(self, model_type, checkpoint_path, encoder_type="tinysam", device="cuda:0", image_size=1024):
        super(ImageEncoder, self).__init__()
        self.encoder_type = encoder_type.lower() if encoder_type else "tinysam"
        self.device = device
        
        if self.encoder_type in ("sam_med2d", "sammed2d"):
            # Use SAM-Med2D encoder
            # Infer model type from checkpoint path
            sam_med2d_model_type = None  # Let SAMMed2DVisionBackbone auto-detect
            logging.info(f"Initializing SAM-Med2D encoder with checkpoint: {checkpoint_path}")
            self.image_encoder = SAMMed2DVisionBackbone(
                checkpoint_path=checkpoint_path,
                image_size=image_size,
                device=device,
                model_type=sam_med2d_model_type  # None = auto-detect from path
            )
        else:
            # Use TinySAM encoder (default)
            logging.info(f"Initializing TinySAM encoder with type {model_type} and checkpoint: {checkpoint_path}")
            self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
            self.image_encoder = self.sam.image_encoder

    def forward(self, inputs):
        return self.image_encoder(inputs)


class LLMSeg(nn.Module):
    def __init__(
            self, 
            model_path, 
            model_base=None, 
            load_8bit=False, 
            load_4bit=False, 
            device="cuda:0",
            cls_num_out=6
        ):

        super(LLMSeg, self).__init__()
        disable_torch_init()
        self.device = device        
        lora_config = LoraConfig(
            r=16,
            lora_alpha=16,
            lora_dropout=0.1,
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        )

        model_name = model_path.split('/')[-1]  # Extract model name from HF ID
        logging.info(f"Loading model from Hugging Face: {model_path}")
        
        self.tokenizer, self.base_model, self.image_processor, self.context_len = load_pretrained_model(
            model_path,
            model_base,
            model_name,
            load_8bit,
            load_4bit,
            device=self.device
        )
        self.base_model.eval()

        self.model = get_peft_model(self.base_model, lora_config)
        if self.training:
            self.model.to(dtype=torch.bfloat16)

        self.mask_decoder = PromptedMaskDecoder()

        # Image encoder will be set via set_image_encoder method
        self.image_encoder = None
        
        self.cls = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, cls_num_out)
        )
        torch.nn.init.xavier_uniform_(self.cls[2].weight)
        torch.nn.init.ones_(self.cls[2].bias)

    def set_image_encoder(self, model_type, checkpoint_path, encoder_type="tinysam", image_size=1024):
        """Set the image encoder after initialization
        
        Args:
            model_type: Model type string (e.g., "vit_t" for TinySAM, ignored for SAM-Med2D)
            checkpoint_path: Path to the checkpoint file
            encoder_type: "tinysam" or "sam_med2d" (default: "tinysam")
            image_size: Input image size (default: 1024)
        """
        self.image_encoder = ImageEncoder(
            model_type=model_type,
            checkpoint_path=checkpoint_path,
            encoder_type=encoder_type,
            device=self.device,
            image_size=image_size
        )
        self.image_encoder.train()

    def get_model_utils(self):
        return self.tokenizer, self.image_processor, self.context_len, self.base_model.config
    
    def save_model(self, save_path):
        os.makedirs(save_path, exist_ok=True)
        self.model.save_pretrained(os.path.join(save_path, "lora_adapter"))
        self.tokenizer.save_pretrained(os.path.join(save_path, "tokenizer"))
        torch.save(self.image_encoder.state_dict(), os.path.join(save_path, "image_encoder.pth"))
        torch.save(self.mask_decoder.state_dict(), os.path.join(save_path, "mask_decoder.pth"))
        torch.save(self.cls.state_dict(), os.path.join(save_path, "cls.pth"))

    def load_model(self, load_path):
        logging.info("Loading model from:", load_path)
        self.tokenizer = self.tokenizer.from_pretrained(os.path.join(load_path, "tokenizer"))
        self.mask_decoder.load_state_dict(torch.load(os.path.join(load_path, "mask_decoder.pth")))
        if self.image_encoder is not None:
            self.image_encoder.load_state_dict(torch.load(os.path.join(load_path, "image_encoder.pth")))
        self.model = PeftModel.from_pretrained(self.model, os.path.join(load_path, "lora_adapter"))
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
        self.mask_decoder.to(self.device)
        self.mask_decoder.eval()
        self.model = self.model.merge_and_unload()
        if self.image_encoder is not None:
            self.image_encoder.eval()
        self.model.eval()
        return self.tokenizer
    
    def generate(
        self,
        input_ids,
        image_tensor_for_vlm,
        image_tensor_for_image_enc,
        input_ids_for_seg=None,
        temperature=0.1,
        max_new_tokens=512,
        top_p=0.95
    ):
        self.image_encoder.eval()
        self.model.eval()
        self.mask_decoder.eval()
        
        # Ensure model is in float16 for inference (consistent with autocast)
        # Check current dtype and convert if needed - this converts ALL submodules including mm_projector
        model_dtype = next(self.model.parameters()).dtype
        if model_dtype == torch.bfloat16 or model_dtype == torch.float32:
            # Convert entire model (including all submodules) to float16 for inference
            self.model = self.model.to(dtype=torch.float16)
        
        # Also ensure base_model is in float16 if it exists and is different
        if hasattr(self, 'base_model') and self.base_model is not None:
            base_dtype = next(self.base_model.parameters()).dtype
            if base_dtype == torch.bfloat16 or base_dtype == torch.float32:
                self.base_model = self.base_model.to(dtype=torch.float16)
        
        with torch.no_grad():
            # Ensure image_tensor_for_vlm is in float16 to match model
            # Get dtype from model (should be float16 after conversion above)
            target_dtype = next(self.model.parameters()).dtype
            if image_tensor_for_vlm.dtype != target_dtype:
                image_tensor_for_vlm = image_tensor_for_vlm.to(dtype=target_dtype)
            
            output_ids = self.model.generate(
                inputs = input_ids,
                images = image_tensor_for_vlm,
                do_sample=True if temperature > 0 else False,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                top_p=top_p
            )

            image_embedding = self.image_encoder(image_tensor_for_image_enc)
            
            # After load_model(), self.model has merged LoRA weights
            # Use self.model if it has extract_last_hidden_state, otherwise use base_model
            # This ensures we use the merged weights (important for correct inference)
            if hasattr(self.model, 'extract_last_hidden_state'):
                model_for_embedding = self.model
            else:
                # Fallback to base_model if merged model doesn't have the method
                model_for_embedding = self.base_model
            
            # Ensure image_tensor_for_vlm matches model dtype (double-check)
            if image_tensor_for_vlm.dtype != next(model_for_embedding.parameters()).dtype:
                image_tensor_for_vlm = image_tensor_for_vlm.to(dtype=next(model_for_embedding.parameters()).dtype)
            
            prompt_embedding = model_for_embedding.extract_last_hidden_state(
                input_ids = input_ids_for_seg if input_ids_for_seg is not None else input_ids,
                images = image_tensor_for_vlm,
                do_sample=False,
                temperature=0,
                max_new_tokens=max_new_tokens,
                top_p=top_p
            )["hidden_states"][-1]
            
            # Validate embeddings before passing to mask_decoder
            if torch.isnan(prompt_embedding).any() or torch.isinf(prompt_embedding).any():
                raise RuntimeError(
                    f"NaN/Inf detected in prompt_embedding. "
                    f"Shape: {prompt_embedding.shape}, "
                    f"Model used: {'self.model' if hasattr(self.model, 'extract_last_hidden_state') else 'self.base_model'}"
                )
            if torch.isnan(image_embedding).any() or torch.isinf(image_embedding).any():
                raise RuntimeError(f"NaN/Inf detected in image_embedding. Shape: {image_embedding.shape}")
            
            final_mask = self.mask_decoder(
                image_embedding, prompt_embedding
            )
        return final_mask, output_ids

    def forward(self,
        input_ids,
        image_tensor_for_vlm,
        image_tensor_for_image_enc,
        attention_mask = None,
        answers=None,
        temperature=0.0001,
        max_new_tokens=512,
        top_p=0.95
    ):
        if self.training:
            self.model.to(dtype=torch.bfloat16)
        else:
            self.model.to(dtype=torch.float16)

        prompt_embedding = self.model(
            input_ids = answers,
            attention_mask=attention_mask,
            images=image_tensor_for_vlm,
            use_cache = False,
            labels=answers,
            return_dict=True,
            output_hidden_states=True,
        )["hidden_states"][-1]

        image_embedding = self.image_encoder(image_tensor_for_image_enc)
        output_cls = self.cls(image_embedding)
        final_mask = self.mask_decoder(
            image_embedding, prompt_embedding
        )
        if self.training:
            logit_loss = self.model(
                input_ids = answers,
                attention_mask=attention_mask,
                images=image_tensor_for_vlm,
                use_cache = False,
                labels=answers
            ).loss
            return final_mask, output_cls, logit_loss
        else:
            output = self.model(
                input_ids = answers,
                attention_mask=attention_mask,
                images=image_tensor_for_vlm,
                use_cache = False,
                labels=answers
            ).logits
            return final_mask, output


def build_llm_seg(
        model_path, 
        model_base=None, 
        load_8bit=False, 
        load_4bit=False, 
        device="cuda:0",
        sam_model_type="vit_t",
        sam_checkpoint_path=None,
        encoder_type="tinysam",
        image_size=1024,
        cls_num_out=6
):
    llm_seg = LLMSeg(
        model_path=model_path,
        model_base=model_base,
        load_8bit=load_8bit,
        load_4bit=load_4bit,
        device=device,
        cls_num_out=cls_num_out
    )
    
    if sam_checkpoint_path is not None:
        llm_seg.set_image_encoder(
            sam_model_type, 
            sam_checkpoint_path,
            encoder_type=encoder_type,
            image_size=image_size
        )

    tokenizer, image_processor, context_len, config = llm_seg.get_model_utils()
    return llm_seg, tokenizer, image_processor, config

