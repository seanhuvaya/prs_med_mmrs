"""
Generate annotation CSV files from split datasets using paper templates.

This script:
1. Computes position from masks (center, quadrant, etc.)
2. Uses train templates (50 templates) for train and val splits
3. Uses test templates (5 templates) for test split
4. Generates CSV files with question-answer pairs
"""

import os
import random
from pathlib import Path
from typing import List, Tuple, Dict
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy import ndimage

from utils import logging
logger = logging.get_logger(__name__)


# Train templates (50 templates from paper Appendix A.2)
TRAIN_TEMPLATES = [
    "Q: Can you identify the location of the tumour in this {image_type} medical image? A: The tumour is located in the {position_description} region of the {image_type} image.",
    "Q: Please describe the tumour's position in this medical image of types {image_type}. A: In this {image_type} medical image, the tumour appears in the {position_description}.",
    "Q: What is the anatomical location of the tumour in this {image_type} medical image? A: The anatomical position of the tumour in this {image_type} image is {position_description}.",
    "Q: Based on this {image_type} medical image, can you provide the location of the tumour in this image? A: From the {image_type} image, the tumour is seen in the {position_description} area.",
    "Q: Where is the tumour located in this {image_type} medical image? A: The tumour is situated in the {position_description} part of the {image_type} image.",
    "Q: In this {image_type} image, where can the tumour be found? A: The tumour can be found in the {position_description}.",
    "Q: What part of the {image_type} image shows the tumour? A: The {position_description} part.",
    "Q: In this {image_type} image, what is the tumour's anatomical position? A: The anatomical position of the tumour is {position_description}.",
    "Q: Identify the segment of this {image_type} that has a tumour. A: The segment is {position_description}.",
    "Q: Where is the abnormal mass located in this {image_type} scan? A: The abnormal mass appears in the {position_description}.",
    "Q: Can you detect the tumour's placement in the {image_type} image? A: The placement is in the {position_description} zone.",
    "Q: Is the tumour visible in this {image_type}, and where is it found? A: Yes, it is located in the {position_description} portion.",
    "Q: Which anatomical zone in the {image_type} image shows a tumour? A: It is visible in the {position_description} region.",
    "Q: Where does the tumour appear in this {image_type} scan? A: It appears in the {position_description} region of the scan.",
    "Q: Indicate the region where the tumour is located in this {image_type}. A: The region is the {position_description}.",
    "Q: In this scan of {image_type}, where do you see the tumour? A: The tumour is seen in the {position_description} area.",
    "Q: What area in the {image_type} image reveals the tumour? A: The area is {position_description}.",
    "Q: According to this {image_type} image, where is the tumour found? A: It is found in the {position_description}.",
    "Q: What is the approximate tumour position in this {image_type}? A: Approximately, it lies in the {position_description}.",
    "Q: Give the precise tumour location in this {image_type} image. A: It is precisely located in the {position_description}.",
    "Q: Can the tumour be located in the upper or lower part of the {image_type}? A: It is found in the {position_description} section.",
    "Q: Which side of the {image_type} contains the tumour? A: The tumour is on the {position_description} side.",
    "Q: In this {image_type} scan, which quadrant has the tumour? A: The {position_description} quadrant contains the tumour.",
    "Q: What part of the {image_type} is affected by the tumour? A: The {position_description} part is affected.",
    "Q: Where is the main tumour mass observed in this {image_type}? A: It is observed in the {position_description} region.",
    "Q: Describe the tumour's spatial location in this {image_type} scan. A: The spatial location corresponds to the {position_description}.",
    "Q: Where is the suspicious mass situated in this {image_type}? A: It is situated at the {position_description}.",
    "Q: Which image region shows the most tumour density in this {image_type}? A: The region with most density is the {position_description}.",
    "Q: Can you tell which section of the image highlights the tumour? A: The highlighted tumour appears in the {position_description} section.",
    "Q: In this {image_type} medical scan, where can the tumour be localized? A: It can be localized in the {position_description} area.",
    "Q: Where is the focal point of the tumour in this {image_type}? A: The focal point is at the {position_description}.",
    "Q: Which directional area of the {image_type} shows the tumour? A: The tumour shows up in the {position_description} direction.",
    "Q: Can you indicate the approximate region where the tumour lies? A: It lies approximately in the {position_description}.",
    "Q: Where would you mark the tumour in this {image_type}? A: I would mark the tumour in the {position_description}.",
    "Q: In this view of the {image_type}, what part contains the tumour? A: The tumour is in the {position_description} view.",
    "Q: What's the visible tumour location in the {image_type} image? A: Visibly, it is in the {position_description}.",
    "Q: According to the image, where does the tumour appear? A: It appears in the {position_description} area.",
    "Q: From the given {image_type}, where can we see the tumour? A: It is seen in the {position_description} region.",
    "Q: What is the rough location of the tumour in the image? A: Roughly, the tumour is at the {position_description}.",
    "Q: Could you highlight the tumour's location in this {image_type} image? A: The tumour is highlighted in the {position_description} region.",
    "Q: Show me where the tumour is in this {image_type} scan. A: The tumour is in the {position_description}.",
    "Q: Locate the tumour position in this {image_type} medical image. A: The position is in the {position_description}.",
    "Q: What region contains the tumour in this {image_type}? A: The {position_description} region contains it.",
    "Q: Point out the tumour location in this {image_type} image. A: It is located in the {position_description}.",
    "Q: In this {image_type}, identify where the tumour lies. A: The tumour lies in the {position_description}.",
    "Q: Determine the tumour's position in this {image_type} scan. A: The position is the {position_description}.",
    "Q: Find the tumour location in this {image_type} medical image. A: The location is the {position_description}.",
    "Q: Specify where the tumour is in this {image_type}. A: The tumour is in the {position_description}.",
    "Q: Reveal the tumour position in this {image_type} image. A: The position is the {position_description}.",
]

# Test templates (5 templates from paper Appendix A.2)
TEST_TEMPLATES = [
    "Q: Can you identify the location of the tumour in this {image_type} medical image? A: The tumour is located in the {position_description} region of the {image_type} image.",
    "Q: Please describe the tumour's position in this medical image of types {image_type}. A: In this {image_type} medical image, the tumour appears in the {position_description}.",
    "Q: What is the anatomical location of the tumour in this {image_type} medical image? A: The anatomical position of the tumour in this {image_type} image is {position_description}.",
    "Q: Based on this {image_type} medical image, can you provide the location of the tumour in this image? A: From the {image_type} image, the tumour is seen in the {position_description} area.",
    "Q: Where is the tumour located in this {image_type} medical image? A: The tumour is situated in the {position_description} part of the {image_type} image.",
]


def compute_position_from_mask(mask_path: Path, center_threshold: float = 20.0) -> str:
    """
    Compute position description from mask image following paper methodology.
    
    Paper method:
    1. For single tumor: select region with largest mask area
    2. For multiple tumors: detect both regions and combine descriptions
    3. Compute Euclidean distance from image center to mask centroid
    4. If distance < threshold, add "near the center" qualifier
    5. Combine into descriptive sentences
    
    Args:
        mask_path: Path to mask image
        center_threshold: Distance threshold for "near the center" (default: 20.0)
    
    Returns:
        Position description string (e.g., "top left region and near the center")
    """
    mask = Image.open(mask_path).convert("L")
    mask_array = np.array(mask)
    height, width = mask_array.shape
    
    # Get mask region (where pixel values > 0)
    mask_binary = mask_array > 0
    
    if mask_binary.sum() == 0:
        return "center"  # Default if no mask found
    
    # Find connected components to detect multiple tumors
    labeled_mask, num_features = ndimage.label(mask_binary)
    
    # Get properties of each connected component
    tumor_regions = []
    for label_id in range(1, num_features + 1):
        region_mask = labeled_mask == label_id
        area = region_mask.sum()
        
        # Compute centroid
        y_coords, x_coords = np.where(region_mask)
        centroid_y = y_coords.mean()
        centroid_x = x_coords.mean()
        
        tumor_regions.append({
            'area': area,
            'centroid': (centroid_y, centroid_x),
            'mask': region_mask
        })
    
    # Sort by area (largest first) - paper says "select the region that consists the largest mask area"
    tumor_regions.sort(key=lambda x: x['area'], reverse=True)
    
    # Image center
    image_center_y = height / 2
    image_center_x = width / 2
    
    position_descriptions = []
    
    # Process tumors (up to 2 for paper methodology)
    for i, tumor in enumerate(tumor_regions[:2]):  # Paper mentions handling up to 2 tumors
        centroid_y, centroid_x = tumor['centroid']
        
        # Compute Euclidean distance from image center
        distance_to_center = np.sqrt(
            (centroid_y - image_center_y)**2 + (centroid_x - image_center_x)**2
        )
        
        # Determine quadrant/region
        # Divide image into 4 quadrants
        if centroid_y < height / 2:
            vertical = "top"
        else:
            vertical = "bottom"
        
        if centroid_x < width / 2:
            horizontal = "left"
        else:
            horizontal = "right"
        
        region_name = f"{vertical} {horizontal}"
        
        # Check if near center (paper: "distance threshold for being considered 'near the center' at 20 unit")
        is_near_center = distance_to_center < center_threshold
        
        # Build position description
        # Paper format: "top left region and near the center" or "bottom left quadrant, near the center"
        if i == 0:
            # First tumor uses "region"
            if is_near_center:
                position_desc = f"{region_name} region and near the center"
            else:
                position_desc = f"{region_name} region"
        else:
            # Second tumor uses "quadrant" (paper example)
            if is_near_center:
                position_desc = f"{region_name} quadrant, near the center"
            else:
                position_desc = f"{region_name} quadrant"
        
        position_descriptions.append(position_desc)
    
    # Combine descriptions following paper format
    if len(position_descriptions) == 1:
        return position_descriptions[0]
    elif len(position_descriptions) == 2:
        # Paper format: "bottom left quadrant, near the center, and another tumour is located in the bottom right quadrant, near the center"
        # Note: first description already includes "region and near the center", second includes "quadrant, near the center"
        return f"{position_descriptions[0]}, and another tumour is located in the {position_descriptions[1]}"
    else:
        # If more than 2 tumors, combine first two (largest areas)
        return f"{position_descriptions[0]}, and another tumour is located in the {position_descriptions[1]}"


def generate_qa_pair(template: str, position_description: str, image_type: str = "medical") -> Tuple[str, str]:
    """
    Generate question-answer pair from template.
    
    Args:
        template: Template string with {position_description} and {image_type} placeholders
        position_description: Position description (e.g., "upper-left")
        image_type: Type of medical image (default: "medical")
    
    Returns:
        Tuple of (question, answer)
    """
    qa_text = template.format(
        position_description=position_description,
        image_type=image_type
    )
    
    # Split into question and answer
    if " A: " in qa_text:
        question, answer = qa_text.split(" A: ", 1)
        question = question.replace("Q: ", "").strip()
        answer = answer.strip()
    else:
        # Fallback if format is different
        question = qa_text
        answer = position_description
    
    return question, answer


def generate_annotations_for_split(
    dataset_root: Path,
    split: str,
    dataset_name: str,
    templates: List[str],
    image_type: str = "medical"
) -> pd.DataFrame:
    """
    Generate annotations for a specific split.
    
    Args:
        dataset_root: Root directory of the dataset
        split: Split name (train, val, or test)
        dataset_name: Name of the dataset
        templates: List of templates to use
        image_type: Type of medical image
    
    Returns:
        DataFrame with annotations
    """
    images_dir = dataset_root / f"{split}_images"
    masks_dir = dataset_root / f"{split}_masks"
    
    if not images_dir.exists():
        logger.warning(f"{split}_images directory not found: {images_dir}")
        return pd.DataFrame()
    
    # Get all image files
    image_files = sorted(list(images_dir.glob("*.png")))
    
    if len(image_files) == 0:
        logger.warning(f"No images found in {images_dir}")
        return pd.DataFrame()
    
    rows = []
    
    for img_file in tqdm(image_files, desc=f"Processing {split} images"):
        # Find corresponding mask
        mask_file = masks_dir / img_file.name
        if not mask_file.exists():
            logger.warning(f"Mask not found for {img_file.name}, skipping")
            continue
        
        # Compute position from mask
        position_description = compute_position_from_mask(mask_file)
        
        # Randomly select a template
        template = random.choice(templates)
        
        # Generate Q&A pair
        question, answer = generate_qa_pair(template, position_description, image_type)
        
        # Create relative path for image (prefixed with dataset name)
        image_path = f"{dataset_name}/{split}_images/{img_file.name}"
        
        rows.append({
            "image_path": image_path,
            "image_name": img_file.name,
            "question": question,
            "answer": answer,
            "position": position_description,
            "split": split,
        })
    
    df = pd.DataFrame(rows)
    return df


def generate_annotations(
    dataset_root: str,
    dataset_name: str,
    image_type: str = "medical",
    seed: int = 42
):
    """
    Generate annotation CSV files for all splits.
    
    Args:
        dataset_root: Root directory of the dataset
        dataset_name: Name of the dataset
        image_type: Type of medical image (e.g., "CT", "MRI", "X-ray")
        seed: Random seed for template selection
    """
    random.seed(seed)
    
    dataset_path = Path(dataset_root)
    
    if not dataset_path.exists():
        raise ValueError(f"Dataset root not found: {dataset_root}")
    
    # Create annotations directory
    annotations_dir = dataset_path.parent / "annotations"
    annotations_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate annotations for each split
    all_dfs = []
    
    # Train and val use train templates
    for split in ["train", "val"]:
        logger.info(f"Generating annotations for {split} split...")
        df = generate_annotations_for_split(
            dataset_path, split, dataset_name, TRAIN_TEMPLATES, image_type
        )
        if len(df) > 0:
            all_dfs.append(df)
            logger.info(f"  Generated {len(df)} annotations for {split}")
    
    # Test uses test templates
    logger.info("Generating annotations for test split...")
    test_df = generate_annotations_for_split(
        dataset_path, "test", dataset_name, TEST_TEMPLATES, image_type
    )
    if len(test_df) > 0:
        all_dfs.append(test_df)
        logger.info(f"  Generated {len(test_df)} annotations for test")
    
    # Combine all splits
    if len(all_dfs) == 0:
        logger.warning("No annotations generated!")
        return
    
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # Save to CSV
    csv_path = annotations_dir / f"{dataset_name}.csv"
    combined_df.to_csv(csv_path, index=False)
    logger.info(f"✓ Saved annotations to {csv_path}")
    logger.info(f"  Total annotations: {len(combined_df)}")
    logger.info(f"  Train: {len(combined_df[combined_df['split'] == 'train'])}")
    logger.info(f"  Val: {len(combined_df[combined_df['split'] == 'val'])}")
    logger.info(f"  Test: {len(combined_df[combined_df['split'] == 'test'])}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate annotation CSV files from split datasets"
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        required=True,
        help="Root directory of the dataset (should contain train_images, train_masks, etc.)"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Name of the dataset (used for CSV filename)"
    )
    parser.add_argument(
        "--image_type",
        type=str,
        default="medical",
        help="Type of medical image (e.g., 'CT', 'MRI', 'X-ray', default: 'medical')"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for template selection (default: 42)"
    )
    
    args = parser.parse_args()
    generate_annotations(
        dataset_root=args.dataset_root,
        dataset_name=args.dataset_name,
        image_type=args.image_type,
        seed=args.seed
    )

