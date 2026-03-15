import os
import pandas as pd

EXPERIMENTS_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "logs",
    "experiments.csv"
)


def load_experiments():
    if not os.path.exists(EXPERIMENTS_PATH):
        return None

    df = pd.read_csv(EXPERIMENTS_PATH)

    if df.empty:
        return None

    return df


def get_best_experiment():
    df = load_experiments()

    if df is None:
        return None

    df = df.sort_values(
        by=["test_accuracy", "ai_f1"],
        ascending=False
    ).reset_index(drop=True)

    return df.iloc[0].to_dict()


def get_top_experiments(limit=5):
    df = load_experiments()

    if df is None:
        return None

    df = df.sort_values(
        by=["test_accuracy", "ai_f1"],
        ascending=False
    ).head(limit)

    return df