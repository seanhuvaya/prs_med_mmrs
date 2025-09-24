from src.dataset.mmrs_builder import build_mmrs_dataset

if __name__ == "__main__":
    build_mmrs_dataset(
        image_dir="data/raw/BUSI/images",
        mask_dir="data/raw/BUSI/masks",
        source="BUSI",
        img_type="ultrasound"
    )