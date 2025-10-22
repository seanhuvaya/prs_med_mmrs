# import pandas as pd
#
#
# csv_path = "data/breast_tumors_ct_scan.csv"
#
# df = pd.read_csv(csv_path)
#
# if "split" not in df.columns:
#     raise ValueError("No split")
#
# train_df = df[df["split"].str.lower() == "train"]
# test_df = df[df["split"].str.lower() == "test"]
# val_df = df[df["split"].str.lower() == "val"]
#
# train_df.to_csv("data/breast_tumors_ct_scan_train.csv", index=False)
# test_df.to_csv("data/breast_tumors_ct_scan_test.csv", index=False)
# val_df.to_csv("data/breast_tumors_ct_scan_val.csv", index=False)
#
# print(f"✅ Saved {len(train_df)} training samples, {len(val_df)} validation samples and {len(test_df)} testing samples.")


from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="llava-hf/llava-v1.6-mistral-7b",
    local_dir="weights/llava-v1.6-mistral-7b",
    local_dir_use_symlinks=False
)

