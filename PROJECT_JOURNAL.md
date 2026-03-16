# AI Authorship Detector — Project Journal

This document records the development process, design decisions, and improvements made during the creation of the **AI Authorship Detector** project.

The goal of this journal is to track the evolution of the system from initial idea to final working demo.

---

# 1. Project Goal

The objective of this project is to build a machine learning system that can classify whether a piece of text is:

* **Human-written**
* **AI-generated**

The project focuses on building a **clean NLP pipeline** rather than relying on large deep learning models.

Key goals:

* Build a reproducible ML pipeline
* Experiment with feature engineering
* Train and evaluate a classifier
* Provide a web demo application

---

# 2. Initial System Design

The initial architecture of the project was designed as a classical NLP pipeline:

Dataset
→ Preprocessing
→ Feature extraction (TF-IDF)
→ Linear classifier (LinearSVC)
→ Evaluation
→ Demo application

This approach was chosen because:

* It is lightweight
* Easy to interpret
* Suitable for experimentation

---

# 3. Dataset Collection

Two datasets were selected for training:

### HC3 Dataset

A dataset containing human and ChatGPT responses.

Used for:

* initial experimentation
* baseline model training

### HAPE Dataset

Human-AI Parallel Corpus containing paired human and AI generated text.

Used to improve:

* dataset diversity
* model generalization

---

# 4. Data Processing Pipeline

To standardize different datasets, a common schema was introduced.

Required columns:

```
text
label
source
subdomain
```

Processing scripts were created:

* `prepare_hc3.py`
* `prepare_hape.py`

These scripts:

* clean text
* normalize labels
* standardize metadata

---

# 5. Dataset Merging

After preprocessing, datasets are merged using:

```
merge_datasets.py
```

Key features of the merge process:

* dataset validation
* column normalization
* label filtering
* balanced sampling
* generation of both full and sample datasets

The sample dataset is used for quick experiments.

---

# 6. Feature Engineering Experiments

Three feature configurations were tested:

### Word n-grams

Pros:

* captures word patterns

Cons:

* sensitive to vocabulary shifts

---

### Character n-grams

Pros:

* captures stylistic patterns
* robust to vocabulary variation

Cons:

* larger feature space

---

### Hybrid features

Combination of word and character features.

---

# 7. Model Selection

The main classifier used in this project is:

```
LinearSVC
```

Reasons for this choice:

* strong performance on high-dimensional text features
* efficient training
* widely used in text classification

---

# 8. Model Evaluation

Evaluation metrics used:

* accuracy
* precision
* recall
* F1 score
* confusion matrix

Example result:

Accuracy ≈ **0.91**

The model achieved balanced performance between AI and human classes.

---

# 9. Error Analysis

To better understand model behavior, misclassified examples were extracted.

Two analysis files were generated:

* AI predicted as human
* Human predicted as AI

This helped identify:

* ambiguous writing styles
* short responses
* neutral language patterns

---

# 10. Experiment Tracking

Training runs are logged into:

```
logs/experiments.csv
```

Each experiment records:

* dataset name
* feature configuration
* training size
* accuracy
* timestamp

This allows easy comparison between experiments.

---

# 11. Demo Application

A Streamlit application was developed to demonstrate the model.

Features:

* interactive text input
* AI vs human prediction
* confidence score
* research summary page

Run locally:

```
streamlit run app/app.py
```

The application was later deployed online using **Streamlit Cloud**.

---

# 12. Repository Cleanup

Before finalizing the project, the repository was cleaned:

Removed:

* experimental models
* intermediate datasets
* temporary files

Kept:

* final model
* training pipeline
* dataset processing scripts

This improved readability and maintainability.

---

# 13. Final Project Structure

The final repository structure:

```
ai-authorship-detector
│
├── app
├── analysis
├── data
├── logs
├── model
├── training
├── utils
├── requirements.txt
└── README.md
```

---

# 14. Lessons Learned

Key lessons from this project:

* Dataset quality strongly affects model performance
* Character features are powerful for stylistic analysis
* Clean pipelines are critical for reproducibility
* Error analysis provides valuable insights

---

# 15. Future Improvements

Possible extensions:

* transformer-based models (BERT)
* larger datasets
* multilingual support
* improved UI for the demo application

---

# Project Status

Project completed as a portfolio machine learning project demonstrating:

* NLP pipeline design
* dataset engineering
* model training and evaluation
* ML experiment tracking
* interactive deployment
