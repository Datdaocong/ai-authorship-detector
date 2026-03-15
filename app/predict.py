import os
import sys
import joblib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.preprocess import clean_text

MODEL_NAME = "merged_real_v1_sample_char"

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "model",
    f"{MODEL_NAME}_detector.pkl"
)

VECTORIZER_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "model",
    f"{MODEL_NAME}_vectorizer.pkl"
)

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)


def predict_text(text: str) -> dict:
    cleaned_text = clean_text(text)

    if not cleaned_text.strip():
        return {
            "prediction": None,
            "decision_score": None,
            "cleaned_text": cleaned_text,
            "model_name": MODEL_NAME,
            "message": "Empty input after preprocessing."
        }

    X = vectorizer.transform([cleaned_text])

    prediction = model.predict(X)[0]

    # LinearSVC does not support predict_proba()
    # decision_function gives the signed distance to the decision boundary
    decision_score = float(model.decision_function(X)[0])

    return {
        "prediction": prediction,
        "decision_score": decision_score,
        "cleaned_text": cleaned_text,
        "model_name": MODEL_NAME,
        "message": "Success"
    }