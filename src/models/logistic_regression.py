"""Logistic Regression baseline for automobile-loan default prediction.

The estimator is wrapped in the project's shared preprocessing and feature-
engineering pipeline.  Callers therefore pass raw application features to the
returned pipeline and can persist that single fitted object for inference.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import RANDOM_STATE
from src.preprocessing.feature_engineering import build_pipeline


def build_logistic_regression_pipeline(
    C: float = 1.0,
    class_weight="balanced",
    max_iter: int = 1000,
    random_state: int = RANDOM_STATE,
    solver: str = "lbfgs",
) -> Pipeline:
    """Build the reproducible end-to-end Logistic Regression baseline.

    ``class_weight='balanced'`` compensates for the roughly 8% default rate,
    while the shared pipeline scales continuous variables before they reach
    the linear classifier. ``C=1.0`` and L2 regularisation (the sklearn
    default) intentionally provide an untuned baseline for later comparison.
    """
    model = LogisticRegression(
        C=C,
        class_weight=class_weight,
        max_iter=max_iter,
        random_state=random_state,
        solver=solver,
    )
    return build_pipeline(model)
