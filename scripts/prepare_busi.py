import cv2
import numpy as np

from pathlib import Path


def prepare_busi(input_dir: str = "data/raw/BUSI",
                 output_img_dir: str = "data/raw/BUSI/images",
                 output_masks_dir: str = "data/raw/BUSI/masks") -> None:
    input_dir = Path(input_dir)
    output_img_dir = Path(output_img_dir)
    output_masks_dir = Path(output_masks_dir)

    output_img_dir.mkdir(parents=True, exist_ok=True)
    output_masks_dir.mkdir(parents=True, exist_ok=True)

    for sub_dir in sorted(p for p in input_dir.iterdir() if p.is_dir() and p.name not in ("images", "masks")):
        for img_path in sub_dir.glob("*.png"):
            if "_mask" in img_path.stem:
                continue

            base_stem = img_path.stem.strip()  # e.g., "normal (92)"

            # Be explicit and case-insensitive about suffix
            candidates = list(sub_dir.iterdir())
            mask_paths = sorted(
                p for p in candidates
                if p.is_file()
                and p.suffix.lower() == ".png"
                and p.stem.startswith(base_stem + "_mask")
            )

            img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"[WARN] Skipping unreadable image: {img_path}")
                continue

            # Save image
            cv2.imwrite(str(output_img_dir / f"{base_stem}.png"), img)

            if not mask_paths:
                print(f"[WARN] No mask for {img_path.name}. Looked for prefix '{base_stem}_mask'")
                continue

            combined_mask = None
            for mpath in mask_paths:
                m = cv2.imread(str(mpath), cv2.IMREAD_GRAYSCALE)
                if m is None:
                    print(f"[WARN] Skipping unreadable mask: {mpath}")
                    continue
                combined_mask = m if combined_mask is None else (
                    np.logical_or(combined_mask, m).astype(np.uint8)
                )

            if combined_mask is None:
                print(f"[WARN] All masks unreadable for {img_path.name}")
                continue

            cv2.imwrite(str(output_masks_dir / f"{base_stem}.png"), combined_mask * 255)


if __name__ == "__main__":
    prepare_busi()
