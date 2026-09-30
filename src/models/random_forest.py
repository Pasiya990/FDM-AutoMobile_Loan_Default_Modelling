"""Random Forest baseline - bagging ensemble. See notebooks/05_baseline_models.ipynb
section 2 for the CV results and feature importances this was verified against.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from src.config import RANDOM_STATE
from src.preprocessing.feature_engineering import build_pipeline


def build_random_forest_pipeline(random_state: int = RANDOM_STATE) -> Pipeline:
    """The pipeline notebook, backend and tests all build the model from -
    so every caller constructs the exact same Random Forest."""
    return build_pipeline(
        RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=random_state, n_jobs=-1)
    )
