import os
import pandas as pd

PROCESSED_DIR = os.path.join("data", "processed")

FULL_OUTPUT_FILE = os.path.join(PROCESSED_DIR, "merged_real_v1_full_train.csv")
SAMPLE_OUTPUT_FILE = os.path.join(PROCESSED_DIR, "merged_real_v1_sample_train.csv")

DATASET_FILES = [
    "hc3_train.csv",
    "hape_train.csv",
]

REQUIRED_COLUMNS = ["text", "label", "source", "subdomain"]

MAX_SAMPLES_PER_LABEL = 5000
RANDOM_STATE = 42


def load_dataset(file_path):
    df = pd.read_csv(file_path)

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"{file_path} missing columns {missing_cols}")

    df = df[REQUIRED_COLUMNS].copy()

    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.lower()
    df["source"] = df["source"].astype(str).str.lower()
    df["subdomain"] = df["subdomain"].astype(str).str.lower()

    df = df[df["text"] != ""]
    df = df[df["label"].isin(["human", "ai"])]

    return df


def main():
    datasets = []

    for file in DATASET_FILES:
        path = os.path.join(PROCESSED_DIR, file)
        df = load_dataset(path)
        datasets.append(df)

    merged_df = pd.concat(datasets, ignore_index=True)

    # save full dataset
    merged_df.to_csv(FULL_OUTPUT_FILE, index=False)

    # create balanced sample
    samples = []
    for label in ["human", "ai"]:
        label_df = merged_df[merged_df["label"] == label]
        n = min(MAX_SAMPLES_PER_LABEL, len(label_df))
        samples.append(label_df.sample(n=n, random_state=RANDOM_STATE))

    sample_df = pd.concat(samples).sample(frac=1, random_state=RANDOM_STATE)

    sample_df.to_csv(SAMPLE_OUTPUT_FILE, index=False)

    print("Merged dataset:", len(merged_df))
    print("Sample dataset:", len(sample_df))


if __name__ == "__main__":
    main()