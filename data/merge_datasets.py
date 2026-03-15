import os
import pandas as pd

from datasets_registry import DATASETS

PROCESSED_DIR = os.path.join("data", "processed")

FULL_OUTPUT_FILE = os.path.join(PROCESSED_DIR, "merged_real_v1_full_train.csv")
SAMPLE_OUTPUT_FILE = os.path.join(PROCESSED_DIR, "merged_real_v1_sample_train.csv")

REQUIRED_COLUMNS = ["text", "label", "source", "subdomain"]

MAX_SAMPLES_PER_LABEL = 5000
RANDOM_STATE = 42


def load_and_validate_csv(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)

    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"{file_path} is missing columns: {missing_cols}")

    df = df[REQUIRED_COLUMNS].copy()

    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["source"] = df["source"].astype(str).str.strip().str.lower()
    df["subdomain"] = df["subdomain"].astype(str).str.strip().str.lower()

    df = df[df["text"] != ""]
    df = df[df["label"].isin(["human", "ai"])]

    return df


def build_sample_dataset(merged_df: pd.DataFrame) -> pd.DataFrame:
    sampled_parts = []

    for label in ["human", "ai"]:
        label_df = merged_df[merged_df["label"] == label]
        n = min(MAX_SAMPLES_PER_LABEL, len(label_df))

        sampled_label_df = label_df.sample(n=n, random_state=RANDOM_STATE)
        sampled_parts.append(sampled_label_df)

    sampled_df = pd.concat(sampled_parts, ignore_index=True)
    sampled_df = sampled_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    return sampled_df


def print_summary(df: pd.DataFrame, title: str) -> None:
    print(f"\n=== {title} ===")
    print("Total samples:", len(df))

    print("\nLabel distribution:")
    print(df["label"].value_counts())

    print("\nSource distribution:")
    print(df["source"].value_counts())

    print("\nSubdomain distribution (top 10):")
    print(df["subdomain"].value_counts().head(10))


def main():
    all_dfs = []

    print("Starting dataset merge...")

    for dataset in DATASETS:
        dataset_name = dataset["name"]
        dataset_path = dataset["path"]

        print(f"\nLoading dataset: {dataset_name}")
        print(f"Path: {dataset_path}")

        df = load_and_validate_csv(dataset_path)

        print(f"Loaded {len(df)} samples from {dataset_name}")
        all_dfs.append(df)

    merged_df = pd.concat(all_dfs, ignore_index=True)

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    merged_df.to_csv(FULL_OUTPUT_FILE, index=False, encoding="utf-8")
    print_summary(merged_df, "FULL MERGED DATASET")
    print(f"\nSaved full dataset to: {FULL_OUTPUT_FILE}")

    sample_df = build_sample_dataset(merged_df)
    sample_df.to_csv(SAMPLE_OUTPUT_FILE, index=False, encoding="utf-8")
    print_summary(sample_df, "SAMPLED MERGED DATASET")
    print(f"\nSaved sampled dataset to: {SAMPLE_OUTPUT_FILE}")


if __name__ == "__main__":
    main()