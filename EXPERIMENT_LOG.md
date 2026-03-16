# Experiment Log

AI Authorship Detector

This document tracks the main experiments conducted during the development of the AI Authorship Detector.

---

# Experiment 1 — HC3 Baseline

Dataset: HC3
Features: TF-IDF word n-grams
Model: LinearSVC

Training samples: ~68k
Testing samples: ~17k

Results:

Accuracy: ~0.96

Notes:

The accuracy was very high, which suggested the dataset might be too easy for the task.
Additional real-world data was needed to make the task more challenging.

---

# Experiment 2 — Real Dataset Merge

Datasets used:

* HC3
* HAPE

The datasets were merged to increase diversity and realism.

A preprocessing pipeline was created:

* dataset normalization
* label cleaning
* dataset validation

---

# Experiment 3 — Word-level Features

Dataset: merged_real_v1_sample

Features:

TF-IDF word n-grams

Training samples: 8000
Testing samples: 2000

Results:

Accuracy ≈ 0.877

Observations:

Word features capture vocabulary patterns but struggle with stylistic differences.

---

# Experiment 4 — Character-level Features

Dataset: merged_real_v1_sample

Features:

TF-IDF character n-grams

Training samples: 8000
Testing samples: 2000

Results:

Accuracy ≈ 0.909

Observations:

Character features performed significantly better.

This suggests that stylistic patterns are important for AI authorship detection.

---

# Experiment 5 — Hybrid Features

Dataset: merged_real_v1_sample

Features:

Combination of:

* word n-grams
* char n-grams

Results:

Accuracy ≈ 0.907

Observations:

Hybrid features performed well but did not outperform pure character features.

---

# Best Model

Feature type:

Character TF-IDF

Classifier:

LinearSVC

Accuracy:

~0.91

This model was selected for the final demo application.

---

# Key Insights

Important findings from the experiments:

1. Dataset diversity strongly affects model performance
2. Character-level features capture stylistic patterns effectively
3. Simple linear models can perform very well on text classification tasks
4. Error analysis is important to understand model limitations

---

# Next Steps

Future experiments could include:

* transformer models (BERT)
* larger datasets
* multilingual evaluation
