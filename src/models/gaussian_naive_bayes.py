"""Gaussian Naive Bayes baseline - fast probabilistic contrast to the
ensembles. See notebooks/05_baseline_models.ipynb section 3 for the CV
results this was verified against, and why it's expected to underperform
(the independence assumption breaks on correlated features like
score_mean vs. Score_Source_1/2/3).
"""

from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline

from src.preprocessing.feature_engineering import build_pipeline


def build_gaussian_naive_bayes_pipeline() -> Pipeline:
    """The pipeline notebook, backend and tests all build the model from -
    so every caller constructs the exact same Gaussian Naive Bayes model."""
    return build_pipeline(GaussianNB())
