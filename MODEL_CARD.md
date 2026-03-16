# Model Card

AI Authorship Detection Model

---

# Model Overview

This model predicts whether a text is:

* **Human-written**
* **AI-generated**

The model is trained using classical NLP techniques with TF-IDF features and a Linear Support Vector Machine.

---

# Model Architecture

Feature Extraction:

TF-IDF Vectorizer

Classifier:

LinearSVC

Pipeline:

Text → TF-IDF → LinearSVC → Prediction

---

# Training Data

The model is trained on merged datasets:

| Dataset | Description                |
| ------- | -------------------------- |
| HC3     | Human vs ChatGPT responses |
| HAPE    | Human-AI Parallel Corpus   |

Datasets are normalized into a unified schema before training.

---

# Evaluation Metrics

Metrics used:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

Best observed performance:

Accuracy ≈ **0.91**

---

# Intended Use

This model is intended for:

* research experiments
* educational purposes
* demonstration of ML pipelines

---

# Limitations

The model may fail when:

* text is very short
* writing style is neutral
* AI output is heavily edited by humans

It should **not be used as a definitive AI detector**.

---

# Ethical Considerations

AI authorship detection is a challenging and evolving task.

False positives may occur, and predictions should not be used as definitive proof of AI usage.

---

# Future Improvements

Possible improvements include:

* transformer-based models
* larger training datasets
* multilingual support
* improved evaluation benchmarks
