import streamlit as st
from predict import predict_text

st.set_page_config(
    page_title="AI Authorship Detector",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 AI Authorship Detector")
st.caption("A lightweight detector for distinguishing human-written text from AI-generated text.")

st.markdown("### About this model")
st.write(
    "This demo uses a LinearSVC classifier trained on merged real datasets "
    "with character-level TF-IDF features."
)

sample_human = (
    "I was tired after class, so I stopped by a small coffee shop near campus "
    "and spent an hour reviewing my notes before going home."
)

sample_ai = (
    "Artificial intelligence systems can improve operational efficiency by "
    "automating repetitive tasks and supporting data-driven decision making."
)

st.markdown("### Try a sample")
col1, col2 = st.columns(2)

with col1:
    if st.button("Use Human-like Sample", use_container_width=True):
        st.session_state["input_text"] = sample_human

with col2:
    if st.button("Use AI-like Sample", use_container_width=True):
        st.session_state["input_text"] = sample_ai

default_text = st.session_state.get("input_text", "")

st.markdown("### Input text")
user_text = st.text_area(
    "Paste or type text to analyze",
    value=default_text,
    height=220,
    placeholder="Enter a paragraph here..."
)

analyze = st.button("Analyze Text", use_container_width=True)

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

        st.markdown("## Cleaned Text Used by Model")
        st.code(cleaned_text)

        st.info(
            "This detector is a research-style baseline. "
            "Its predictions depend on the training data and learned writing patterns, "
            "so results should be interpreted cautiously."
        )