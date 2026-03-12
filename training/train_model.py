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

# Clean text
df["clean_text"] = df["text"].apply(clean_text)

# Features and labels
X = df["clean_text"]
y = df["label"]

# Convert text to numerical vectors
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save model and vectorizer
model_dir = os.path.join(os.path.dirname(__file__), "..", "model")
os.makedirs(model_dir, exist_ok=True)

joblib.dump(model, os.path.join(model_dir, "detector.pkl"))
joblib.dump(vectorizer, os.path.join(model_dir, "vectorizer.pkl"))

print("\nModel and vectorizer saved successfully.")