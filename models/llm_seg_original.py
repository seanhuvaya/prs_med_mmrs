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


def custom_lora_init(module):
    if hasattr(module, "lora_A"):
        nn.init.kaiming_uniform_(module.lora_A.weight, a=math.sqrt(5))
    if hasattr(module, "lora_B"):
        nn.init.zeros_(module.lora_B.weight)


class ImageEncoder(nn.Module):
    def __init__(self, model_type, checkpoint_path):
        super(ImageEncoder, self).__init__()
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

    def set_image_encoder(self, model_type, checkpoint_path):
        """Set the image encoder after initialization"""
        self.image_encoder = ImageEncoder(
            model_type=model_type,
            checkpoint_path=checkpoint_path
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
        with torch.no_grad():
            output_ids = self.model.generate(
                inputs = input_ids,
                images = image_tensor_for_vlm,
                do_sample=True if temperature > 0 else False,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                top_p=top_p
            )

            image_embedding = self.image_encoder(image_tensor_for_image_enc)
            
            prompt_embedding = self.base_model.extract_last_hidden_state(
                input_ids = input_ids_for_seg if input_ids_for_seg is not None else input_ids,
                images = image_tensor_for_vlm,
                do_sample=False,
                temperature=0,
                max_new_tokens=max_new_tokens,
                top_p=top_p
            )["hidden_states"][-1]
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
        llm_seg.set_image_encoder(sam_model_type, sam_checkpoint_path)

    tokenizer, image_processor, context_len, config = llm_seg.get_model_utils()
    return llm_seg, tokenizer, image_processor, config

