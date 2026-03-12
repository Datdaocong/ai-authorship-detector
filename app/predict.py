import os
import sys
import joblib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.preprocess import clean_text


MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "detector.pkl")
VECTORIZER_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "vectorizer.pkl")

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)


def predict_text(text):
    cleaned_text = clean_text(text)
    vectorized_text = vectorizer.transform([cleaned_text])

    prediction = model.predict(vectorized_text)[0]
    probabilities = model.predict_proba(vectorized_text)[0]

    class_labels = model.classes_
    confidence_scores = {
        class_labels[i]: float(probabilities[i]) for i in range(len(class_labels))
    }

    return prediction, confidence_scores