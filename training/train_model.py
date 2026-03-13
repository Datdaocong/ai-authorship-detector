import os
import sys
import joblib
import pandas as pd

from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.preprocess import clean_text


# Load dataset
data_path = os.path.join(os.path.dirname(__file__), "..", "data", "dataset_better.csv")
df = pd.read_csv(data_path)

print(f"Loaded dataset with {len(df)} samples.")

# Clean text
df["clean_text"] = df["text"].apply(clean_text)

# Features and labels
X = df["clean_text"]
y = df["label"]

# Convert text to numerical vectors
vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2
)
X_vectorized = vectorizer.fit_transform(X)

cv_model = LinearSVC()
cv_scores = cross_val_score(cv_model, X_vectorized, y, cv=5)

print("\nCross-validation scores:", cv_scores)
print(f"Mean CV accuracy: {cv_scores.mean():.4f}")

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
model = LinearSVC()
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

accuracy = accuracy_score(y_test, y_pred)


# Logging training results
log_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(log_dir, exist_ok=True)

log_path = os.path.join(log_dir, "training_log.txt")

with open(log_path, "a") as f:
    f.write(f"Date: {datetime.now()}\n")
    f.write(f"Dataset size: {len(df)}\n")
    f.write(f"TFIDF features: {X_vectorized.shape[1]}\n")
    f.write(f"Train samples: {X_train.shape[0]}\n")
    f.write(f"Test samples: {X_test.shape[0]}\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write("-" * 40 + "\n")


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