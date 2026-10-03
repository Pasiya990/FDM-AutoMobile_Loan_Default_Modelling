"""XGBoost model for automobile loan default prediction.

This module provides the baseline XGBoost classifier used in Stage 6
(Model Development). Hyperparameter optimization will be performed
separately in Stage 7.
"""

from xgboost import XGBClassifier

from src.config import RANDOM_STATE


def build_xgboost() -> XGBClassifier:
    """Build the baseline XGBoost classifier.

    Returns:
        XGBClassifier: Configured baseline XGBoost model.
    """

    return XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        n_jobs=-1,
    )