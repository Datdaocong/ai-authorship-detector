import os
import pandas as pd

PROCESSED_DIR = os.path.join("data", "processed")
OUTPUT_FILE = os.path.join(PROCESSED_DIR, "merged_real_v1_train.csv")

DATASET_FILES = [
    "hc3_train.csv",
]

REQUIRED_COLUMNS = ["text", "label", "source", "subdomain"]


def load_and_validate_csv(file_path):
    df = pd.read_csv(file_path)

    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns {missing_cols} in {file_path}")

    df = df[REQUIRED_COLUMNS].copy()

    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["source"] = df["source"].astype(str).str.strip().str.lower()
    df["subdomain"] = df["subdomain"].astype(str).str.strip().str.lower()

    df = df[df["text"] != ""]
    df = df[df["label"].isin(["human", "ai"])]

    return df


def main():
    all_dfs = []

    for filename in DATASET_FILES:
        file_path = os.path.join(PROCESSED_DIR, filename)

        print(f"Loading: {file_path}")
        df = load_and_validate_csv(file_path)

        print(f"Samples: {len(df)}")
        print(df['label'].value_counts())
        print("-" * 40)

        all_dfs.append(df)

    merged_df = pd.concat(all_dfs, ignore_index=True)

    print("\nMerged dataset summary:")
    print("Total samples:", len(merged_df))
    print("\nLabel distribution:")
    print(merged_df["label"].value_counts())
    print("\nSource distribution:")
    print(merged_df["source"].value_counts())

    merged_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    print(f"\nSaved merged dataset to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()