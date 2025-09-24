from src.dataset.mmrs_builder import build_mmrs_dataset

if __name__ == "__main__":
    build_mmrs_dataset(
        image_dir="data/raw/Kvasir-SEG/images",
        mask_dir="data/raw/Kvasir-SEG/masks",
        img_type="polyp"
    )