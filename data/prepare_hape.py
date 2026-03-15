from datasets import load_dataset
import pandas as pd
import os

SOURCE_NAME = "hape"

HUMAN_FILE = "hf://datasets/browndw/human-ai-parallel-corpus/text_data/hape-text_human-chunk-2.parquet"

AI_FILES = [
    ("hf://datasets/browndw/human-ai-parallel-corpus/text_data/hape-text_gpt-4o-2024-08-06.parquet", "gpt4o"),
    ("hf://datasets/browndw/human-ai-parallel-corpus/text_data/hape-text_gpt-4o-mini-2024-07-18.parquet", "gpt4o_mini"),
    ("hf://datasets/browndw/human-ai-parallel-corpus/text_data/hape-text_llama-3-70B.parquet", "llama3_70b"),
    ("hf://datasets/browndw/human-ai-parallel-corpus/text_data/hape-text_llama-3-70B-Instruct.parquet", "llama3_70b_instruct"),
    ("hf://datasets/browndw/human-ai-parallel-corpus/text_data/hape-text_llama-3-8B.parquet", "llama3_8b"),
    ("hf://datasets/browndw/human-ai-parallel-corpus/text_data/hape-text_llama-3-8B-Instruct.parquet", "llama3_8b_instruct"),
]

POSSIBLE_TEXT_COLUMNS = [
    "text",
    "content",
    "generated_text",
    "completion",
    "response",
]

POSSIBLE_DOMAIN_COLUMNS = [
    "text_type",
    "genre",
    "domain",
    "source_type",
    "register",
]

OUTPUT_PATH = "data/processed/hape_train.csv"


def find_text_column(df: pd.DataFrame) -> str:
    for col in POSSIBLE_TEXT_COLUMNS:
        if col in df.columns:
            return col
    raise ValueError(f"Could not find text column. Available columns: {df.columns.tolist()}")


def find_domain_column(df: pd.DataFrame):
    for col in POSSIBLE_DOMAIN_COLUMNS:
        if col in df.columns:
            return col
    return None


def load_parquet_as_df(path: str) -> pd.DataFrame:
    ds = load_dataset("parquet", data_files=path, split="train")
    return ds.to_pandas()


def normalize_frame(df: pd.DataFrame, label: str, subdomain_fallback: str) -> pd.DataFrame:
    text_col = find_text_column(df)
    domain_col = find_domain_column(df)

    out = pd.DataFrame()
    out["text"] = df[text_col].astype(str).str.strip()
    out["label"] = label
    out["source"] = SOURCE_NAME

    if domain_col is not None:
        out["subdomain"] = df[domain_col].astype(str).str.strip().str.lower()
    else:
        out["subdomain"] = subdomain_fallback

    out = out[out["text"] != ""].copy()
    return out


def main():
    print("Loading HAP-E human data...")
    human_df_raw = load_parquet_as_df(HUMAN_FILE)
    human_df = normalize_frame(human_df_raw, "human", "general")

    all_frames = [human_df]

    print("Loading HAP-E AI data...")
    for ai_path, model_tag in AI_FILES:
        ai_df_raw = load_parquet_as_df(ai_path)
        ai_df = normalize_frame(ai_df_raw, "ai", model_tag)

        # nếu file có subdomain thật thì giữ subdomain đó;
        # nếu không có thì fallback theo model_tag
        ai_df["subdomain"] = ai_df["subdomain"].replace("", model_tag)

        all_frames.append(ai_df)
        print(f"Loaded AI file: {model_tag} -> {len(ai_df)} samples")

    merged = pd.concat(all_frames, ignore_index=True)

    os.makedirs("data/processed", exist_ok=True)
    merged.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    print("\nSaved dataset:", OUTPUT_PATH)
    print("Total samples:", len(merged))
    print("\nLabel distribution:")
    print(merged["label"].value_counts())
    print("\nSource distribution:")
    print(merged["source"].value_counts())
    print("\nSubdomain distribution (top 10):")
    print(merged["subdomain"].value_counts().head(10))
    print("\nColumns:", merged.columns.tolist())
    print("\nPreview:")
    print(merged.head())


if __name__ == "__main__":
    main()