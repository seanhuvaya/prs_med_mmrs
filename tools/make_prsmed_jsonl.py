import json, pandas as pd, pathlib as p

CSV = "data/breast_tumors_ct_scan.csv"
OUT_DIR = p.Path("data/annotations")
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV)

def guess_paths(row):
    mask_path = p.Path(row["image_path"])
    img_path = p.Path(str(mask_path).replace("train_masks", "train_images").replace("val_masks", "val_images").replace("test_masks", "test_images"))
    return str(img_path), str(mask_path)

records = []
for _, row in df.iterrows():
    img_path, mask_path = guess_paths(row)
    records.append({
        "image_path": img_path,
        "mask_path": mask_path,
        "image_id": row.get("image_name", p.Path(img_path).name),
        "question": row["question"],
        "answer": row["answer"],
        "position": row["position"],
        "split": row["split"],
    })

for split in ["train", "val", "test"]:
    with open(OUT_DIR / f"{split}.jsonl", "w") as f:
        for record in records:
            if record["split"] == split:
                f.write(json.dumps(record) + "\n")


print("Wrote: ", len(records), " records:", list(OUT_DIR.glob("*.jsonl")), end="")


