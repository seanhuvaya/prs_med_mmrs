from src.dataset.download import fetch_dataset

if __name__ == "__main__":
    datasets = ["BUSI","Kvasir-SEG"]
    for dataset in datasets:
        fetch_dataset(dataset)