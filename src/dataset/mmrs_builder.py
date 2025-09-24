import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

TRAIN_TEMPLATES = pd.read_csv('data/train.csv')
TEST_TEMPLATES = pd.read_csv('data/test.csv')


def extract_positions_from_mask(mask: np.ndarray, center_thresh: float = 0.1) -> list[Any]:
    height, width = mask.shape
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    positions = []
    center_x, center_y = width // 2, height // 2

    for contour in contours:
        if cv2.contourArea(contour) < 50:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        cx, cy = center_y + x, center_y + y

        if cx < center_x and cy < center_y:
            quadrant = "top left"
        elif cx >= center_x and cy < center_y:
            quadrant = "top right"
        elif cx < center_x and cy >= center_y:
            quadrant = "bottom left"
        else:
            quadrant = "bottom right"

        distance = np.sqrt((cx - center_x) ** 2 + (cy - center_y) ** 2)
        if distance < center_thresh:
            pos = f"{quadrant}, near the center"
        else:
            pos = f"{quadrant}"

        positions.append(pos)

    return positions


def combine_positions(positions: list[Any]) -> str:
    if not positions:
        return "no tumour detected"

    if len(positions) == 1:
        return f"the {positions[0]}"

    return "tumours located in the " + ", and another in the ".join(positions)


def generate_qa_pairs(img_type: str, pos: str):
    qa_row = TRAIN_TEMPLATES.sample(n=1).iloc[0]
    question = qa_row['Question']
    answer = qa_row['Answer']
    return question.format(image_type=img_type), answer.format(position_description=pos, image_type=img_type)


def build_mmrs_dataset(image_dir: str, mask_dir: str, img_type: str = "MRI", output_file: str = "data/mmrs/mmrs.csv"):
    dataset = []
    image_dir, mask_dir = Path(image_dir), Path(mask_dir)

    for img_file in image_dir.glob("*.jpg"):
        mask_file = mask_dir / img_file.name
        if not mask_file.exists():
            continue

        img = cv2.imread(str(img_file))
        mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
        _, mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)

        positions = extract_positions_from_mask(mask=mask)

        if not positions:
            continue

        pos_text = combine_positions(positions=positions)
        question, answer = generate_qa_pairs(img_type=img_type, pos=pos_text)

        dataset.append({
            "image": str(img_file),
            "mask": str(mask_file),
            "question": question,
            "answer": answer,
        })

        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        df = pd.DataFrame(dataset)
        df.to_csv(output_file, index=False)
