import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from data_utils.utils import load_annotation, load_image, binary_loader
from llava.mm_utils import tokenizer_image_token
from llava.mm_utils import process_images
from torchvision import transforms
import os

IGNORE_INDEX = 0
MAX_PROMPT_LENGTH = 512


class PromptSegmentDataset(Dataset):
    def __init__(
            self,
            data_path,
            annotation_path,
            data_config,
            image_processor,
            tokenizer,
            trainsize=512,
            mode="train"
    ):
        self.data_path = data_path
        self.annotation_path = annotation_path
        self.tokenizer = tokenizer

        self.train_df, self.eval_df = load_annotation(annotation_path)
        mode = (mode or "train").lower()
        if mode in ("train", "training"):
            self.annotation_df = self.train_df
        else:
            # use eval data for both 'val' and 'test'
            self.annotation_df = self.eval_df if not self.eval_df.empty else self.train_df

        self.trainsize = trainsize

        self.IMAGE_TOKEN_INDEX = -200
        self.image_processor = image_processor
        self.data_config = data_config

        self.mask_transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
        ])

        self.image_sam_transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.annotation_df)

    def process_image(self, image_path):
        image_pil = load_image(image_path)
        image_tensor = process_images([image_pil], self.image_processor, self.data_config)
        return image_tensor.squeeze(0).to(torch.float16)

    def process_sam_image(self, image_path):
        image_pil = load_image(image_path)
        image_sam_tensor = self.image_sam_transform(image_pil)
        return image_sam_tensor.to(torch.float32)

    def process_mask(self, mask_path):
        mask_image = binary_loader(mask_path)
        mask_tensor = self.mask_transform(mask_image)
        return mask_tensor

    def __getitem__(self, idx):
        row = self.annotation_df.iloc[idx]

        # --- Build absolute paths from your JSONL fields ---
        img_rel  = str(row["image_path"]).replace("\\", "/")
        msk_rel  = str(row.get("mask_path", "")).replace("\\", "/")  # some sets may not have masks
        image_path = os.path.join(self.data_path, img_rel)
        mask_path  = os.path.join(self.data_path, msk_rel) if msk_rel else None

        # ISIC jpg quirk
        if "ISIC" in image_path and image_path.endswith(".png"):
            jpg_candidate = image_path[:-4] + ".jpg"
            if os.path.exists(jpg_candidate):
                image_path = jpg_candidate

        # Label by dataset family (optional)
        p = image_path.lower()
        if "isic" in p:
            label = 4
        elif "breast" in p:
            label = 1
        elif "brain" in p:
            label = 0
        elif "lung_ct" in p:
            label = 2
        elif "lung_xray" in p:
            label = 3
        else:
            label = 5

        question = row["question"]
        prompt   = row.get("position", "")
        answer   = row["answer"]

        # Tensors
        image_tensor     = self.process_image(image_path)
        image_sam_tensor = self.process_sam_image(image_path)
        mask_tensor      = self.process_mask(mask_path) if mask_path and os.path.exists(mask_path) else torch.zeros(1, 1024, 1024)

        # Tokens
        # prompt fed to model as the "User" message
        prompt_for_vlm = "<image>\n" + question
        input_ids = tokenizer_image_token(prompt_for_vlm, self.tokenizer, self.IMAGE_TOKEN_INDEX, return_tensors='pt')
        # answer used as target
        answer_prompt = "<image>\n" + f"### User: {question} \n" + "### Assistant: \n" + answer
        answers_ids = tokenizer_image_token(answer_prompt, self.tokenizer, self.IMAGE_TOKEN_INDEX, return_tensors='pt')

        return {
            "input_ids": input_ids,             # (1, L)
            "image_tensor": image_tensor,       # (3, H, W) fp16
            "mask_tensor": mask_tensor,         # (1, H, W)
            "answers_ids": answers_ids,         # (1, L)
            "image_sam": image_sam_tensor,      # (3, H, W) fp32
            "label": label
        }


def collate_fn(batch):
    padded_input_ids = nn.utils.rnn.pad_sequence(
        [item['input_ids'].squeeze(0) for item in batch],
        batch_first=True,
        padding_value=IGNORE_INDEX
    )
    input_ids = padded_input_ids[:, :MAX_PROMPT_LENGTH]
    input_ids = input_ids.to(torch.int64)

    padded_answers_ids = nn.utils.rnn.pad_sequence(
        [item['answers_ids'] for item in batch],
        batch_first=True,
        padding_value=IGNORE_INDEX
    )

    answers_ids = padded_answers_ids[:, :MAX_PROMPT_LENGTH]
    answers_ids = answers_ids.to(torch.int64)

    attention_masks = torch.ones_like(answers_ids)
    attention_masks[answers_ids == IGNORE_INDEX] = 0
    attention_masks = attention_masks.to(torch.long)

    image_tensor = [item['image_tensor'] for item in batch]
    image_sam_tensor = [item['image_sam'] for item in batch]
    mask_tensor = [item['mask_tensor'] for item in batch]

    image_tensor = torch.stack(image_tensor, dim=0)
    mask_tensor = torch.stack(mask_tensor, dim=0)
    image_sam_tensor = torch.stack(image_sam_tensor, dim=0)
    return {
        'input_ids': input_ids,
        'image_tensor': image_tensor,
        'mask_tensor': mask_tensor,
        'answers_ids': answers_ids,
        'image_sam': image_sam_tensor,
        "attention_masks": attention_masks,
        "label": torch.tensor([item['label'] for item in batch])
    }

def _dl_runtime_opts(device: torch.device):
    is_cuda = (device.type == "cuda")
    return {
        "pin_memory": is_cuda,                # only benefits CUDA H2D copies
        "pin_memory_device": "cuda" if is_cuda else None,
        "num_workers": 0 if device.type == "mps" else 8,   # macOS often happier with 0–2
        "persistent_workers": False if device.type == "mps" else True,
        "prefetch_factor": None if device.type == "mps" else 2,
    }


def create_dataloader(
        data_path,
        annotation_path,
        data_config,
        image_processor,
        tokenizer,
        batch_size=2,
        mode="train"
):
    dataset = PromptSegmentDataset(
        data_path=data_path,
        annotation_path=annotation_path,
        data_config=data_config,
        image_processor=image_processor,
        tokenizer=tokenizer,
        mode=mode
    )

    device = getattr(data_config, "device", None)

    if device is None:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    opts = _dl_runtime_opts(device)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == "train"),
        collate_fn=collate_fn,
        num_workers=opts["num_workers"],
        pin_memory=opts["pin_memory"],
        persistent_workers=opts["persistent_workers"],
        **({} if opts["prefetch_factor"] is None else {"prefetch_factor": opts["prefetch_factor"]})
    )
    return {mode: loader}

