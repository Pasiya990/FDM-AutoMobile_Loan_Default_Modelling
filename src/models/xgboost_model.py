"""XGBoost model for automobile loan default prediction.

Baseline XGBoost classifier used in Stage 6. It now weights the default class
(scale_pos_weight), the boosting equivalent of the Random Forest's
class_weight="balanced" - without it, XGBoost's probabilities sit below 0.5
for almost every defaulter and its recall at 0.5 was ~0.02, which made the
earlier RF-vs-XGBoost comparison unfair. Leak-free tuning against the Random
Forest: see src/models/fair_comparison.py.
"""

import numpy as np
from xgboost import XGBClassifier

from src.config import RANDOM_STATE

# Weight for the default class: the training split's ratio of non-defaulters
# to defaulters (78,815 / 6,941 = 11.35 after near-duplicate removal). Kept as
# a fixed default so results stay reproducible; pass imbalance_ratio(y_train)
# to compute it from other data.
SCALE_POS_WEIGHT = 11.35


def imbalance_ratio(y) -> float:
    """negatives / positives - XGBoost's equivalent of class_weight='balanced'."""
    y = np.asarray(y)
    return float((y == 0).sum() / (y == 1).sum())


def build_xgboost(scale_pos_weight: float = SCALE_POS_WEIGHT, **overrides) -> XGBClassifier:
    """Build the baseline XGBoost classifier.

    Args:
        scale_pos_weight: weight on the default class (1.0 = unweighted, the
            original baseline).
        **overrides: any other XGBClassifier parameter (e.g. tuned values).
    """
    params = dict(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        n_jobs=-1,
    )
    params.update(overrides)
    return XGBClassifier(**params)
