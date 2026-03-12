import os
import sys
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.preprocess import clean_text


# Load dataset
data_path = os.path.join(os.path.dirname(__file__), "..", "data", "dataset.csv")
df = pd.read_csv(data_path)

print(f"Loaded dataset with {len(df)} samples.")

# Clean text
df["clean_text"] = df["text"].apply(clean_text)

# Features and labels
X = df["clean_text"]
y = df["label"]

# Convert text to numerical vectors
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

print(f"TF-IDF matrix shape: {X_vectorized.shape}")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# Show most important words for each class
feature_names = vectorizer.get_feature_names_out()
coefficients = model.coef_[0]
classes = model.classes_

positive_class = classes[1]
negative_class = classes[0]

top_positive_indices = coefficients.argsort()[-10:][::-1]
top_negative_indices = coefficients.argsort()[:10]

print(f"\nTop words predicting '{positive_class}':")
for idx in top_positive_indices:
    print(f"{feature_names[idx]}: {coefficients[idx]:.4f}")

print(f"\nTop words predicting '{negative_class}':")
for idx in top_negative_indices:
    print(f"{feature_names[idx]}: {coefficients[idx]:.4f}")


# Save model and vectorizer
model_dir = os.path.join(os.path.dirname(__file__), "..", "model")
os.makedirs(model_dir, exist_ok=True)

joblib.dump(model, os.path.join(model_dir, "detector.pkl"))
joblib.dump(vectorizer, os.path.join(model_dir, "vectorizer.pkl"))

print("\nModel and vectorizer saved successfully.")