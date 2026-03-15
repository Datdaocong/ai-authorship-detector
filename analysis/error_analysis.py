import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix

DATA_PATH = "data/processed/merged_real_v1_sample_train.csv"

MODEL_PATH = "model/merged_real_v1_sample_char_detector.pkl"
VECTORIZER_PATH = "model/merged_real_v1_sample_char_vectorizer.pkl"

print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

texts = df["text"]
labels = df["label"]

print("Loading model...")
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

print("Vectorizing text...")
X = vectorizer.transform(texts)

print("Predicting...")
preds = model.predict(X)

df["prediction"] = preds

# Misclassified samples
errors = df[df["label"] != df["prediction"]]

print(f"\nTotal errors: {len(errors)}")

# Human predicted as AI
human_to_ai = errors[
    (errors["label"] == "human") & (errors["prediction"] == "ai")
]

# AI predicted as Human
ai_to_human = errors[
    (errors["label"] == "ai") & (errors["prediction"] == "human")
]

print(f"Human → AI errors: {len(human_to_ai)}")
print(f"AI → Human errors: {len(ai_to_human)}")

# Save errors
human_to_ai.to_csv("analysis/human_predicted_ai.csv", index=False)
ai_to_human.to_csv("analysis/ai_predicted_human.csv", index=False)

print("\nSaved error samples to analysis/")