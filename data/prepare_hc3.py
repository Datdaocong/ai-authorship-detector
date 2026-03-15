from datasets import load_dataset
import pandas as pd
import os

print("Loading HC3 dataset from JSONL...")

ds = load_dataset(
    "json",
    data_files="hf://datasets/Hello-SimpleAI/HC3/all.jsonl",
    split="train",
)

rows = []

for item in ds:
    for ans in item["human_answers"]:
        if ans and ans.strip():
            rows.append((ans.strip(), "human"))

    for ans in item["chatgpt_answers"]:
        if ans and ans.strip():
            rows.append((ans.strip(), "ai"))

df = pd.DataFrame(rows, columns=["text", "label"])

os.makedirs("data/processed", exist_ok=True)
output_path = "data/processed/hc3_train.csv"
df.to_csv(output_path, index=False, encoding="utf-8")

print("Saved dataset:", output_path)
print("Total samples:", len(df))
print(df["label"].value_counts())