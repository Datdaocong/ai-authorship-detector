import streamlit as st

from predict import predict_text
from research_summary import get_best_experiment, get_top_experiments

st.set_page_config(
    page_title="AI Authorship Detector",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 AI Authorship Detector")
st.caption(
    "A lightweight research-style demo for distinguishing human-written text from AI-generated text."
)

st.markdown("### About this model")
st.write(
    "This demo uses a LinearSVC classifier trained on merged real datasets "
    "with character-level TF-IDF features."
)

# =========================
# Research summary section
# =========================
best_experiment = get_best_experiment()
top_experiments = get_top_experiments()

st.markdown("### Research Summary")

if best_experiment is not None:
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Best Feature Mode", str(best_experiment["feature_mode"]))
        st.metric("CV Accuracy", f"{float(best_experiment['cv_accuracy']):.4f}")
        st.metric("Test Accuracy", f"{float(best_experiment['test_accuracy']):.4f}")

    with col2:
        st.metric("AI Precision", f"{float(best_experiment['ai_precision']):.4f}")
        st.metric("AI Recall", f"{float(best_experiment['ai_recall']):.4f}")
        st.metric("AI F1", f"{float(best_experiment['ai_f1']):.4f}")

    st.caption(
        f"Best experiment: dataset={best_experiment['dataset_name']}, "
        f"model={best_experiment['model_type']}, "
        f"features={best_experiment['feature_mode']}"
    )
else:
    st.info(
        "No experiment summary found yet. Train the model first to generate logs/experiments.csv."
    )

if top_experiments is not None:
    st.markdown("### Top Experiments")
    st.dataframe(top_experiments, use_container_width=True)

st.divider()

# =========================
# Sample texts
# =========================
sample_human = (
    "I was tired after class, so I stopped by a small coffee shop near campus "
    "and spent an hour reviewing my notes before going home."
)

sample_ai = (
    "Artificial intelligence systems can improve operational efficiency by "
    "automating repetitive tasks and supporting data-driven decision making."
)

st.markdown("### Try a sample")

col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    if st.button("Use Human-like Sample", use_container_width=True):
        st.session_state["input_text"] = sample_human

with col2:
    if st.button("Use AI-like Sample", use_container_width=True):
        st.session_state["input_text"] = sample_ai

with col3:
    if st.button("Clear Text", use_container_width=True):
        st.session_state["input_text"] = ""

default_text = st.session_state.get("input_text", "")

# =========================
# Input section
# =========================
st.markdown("### Input Text")

user_text = st.text_area(
    "Paste or type text to analyze",
    value=default_text,
    height=220,
    placeholder="Enter a paragraph here..."
)

analyze = st.button("Analyze Text", use_container_width=True)

# =========================
# Prediction section
# =========================
if analyze:
    result = predict_text(user_text)

    if result["prediction"] is None:
        st.warning("Please enter valid text before analyzing.")
    else:
        prediction = result["prediction"]
        decision_score = result["decision_score"]
        cleaned_text = result["cleaned_text"]
        model_name = result["model_name"]

        st.divider()
        st.markdown("## Result")

        if prediction == "human":
            st.success("Likely **HUMAN-WRITTEN**")
        else:
            st.error("Likely **AI-GENERATED**")

        st.markdown("## Decision Score")
        st.write(
            "The decision score shows how strongly the model leans toward its prediction. "
            "Values farther from 0 indicate stronger confidence."
        )
        st.code(f"{decision_score:.4f}")

        score_strength = min(abs(decision_score) / 3.0, 1.0)
        st.progress(score_strength)

        st.markdown("## Model Info")
        st.write(f"**Model:** {model_name}")
        st.write("**Classifier:** LinearSVC")
        st.write("**Features:** Character-level TF-IDF")
        st.write("**Task:** Human vs AI authorship detection")

        st.markdown("## Cleaned Text Used by Model")
        st.code(cleaned_text)

        st.info(
            "This detector is a research-style baseline. "
            "Its predictions depend on the training data and learned writing patterns, "
            "so results should be interpreted cautiously."
        )