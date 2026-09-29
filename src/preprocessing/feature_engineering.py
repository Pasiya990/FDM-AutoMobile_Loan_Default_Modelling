"""Feature engineering, as sklearn-compatible transformers: rare-category
grouping, derived features, RF-importance feature selection, scaling. Runs
after src/preprocessing/preprocessing.py's steps - see
notebooks/04_feature_engineering.ipynb, which this module mirrors.

Also provides build_pipeline(), assembling every preprocessing and
feature-engineering step plus a given model into one fitted sklearn.Pipeline
- the single object the backend will load and call .predict_proba() on for
one new application, with no train_df required at inference time.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (
    COLS_RESERVED_FOR_FEATURE_ENGINEERING,
    LOW_IMPORTANCE_THRESHOLD,
    RANDOM_STATE,
    RARE_CATEGORY_MIN_FREQ,
)
from src.preprocessing.preprocessing import CategoricalOneHotEncoder, build_preprocessing_steps


class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    """Categories below min_freq of training rows get grouped into 'Other'.
    Learned on train only, reapplied to any later data.
    """

    def __init__(self, cols=None, min_freq: float = RARE_CATEGORY_MIN_FREQ):
        self.cols = cols
        self.min_freq = min_freq

    def fit(self, X, y=None):
        cols = self.cols if self.cols is not None else COLS_RESERVED_FOR_FEATURE_ENGINEERING
        min_count = len(X) * self.min_freq
        self.rare_categories_ = {}
        for col in cols:
            value_counts = X[col].value_counts()
            self.rare_categories_[col] = value_counts[value_counts < min_count].index.tolist()
        return self

    def transform(self, X):
        X = X.copy()
        for col, rare_categories in self.rare_categories_.items():
            X[col] = X[col].replace(rare_categories, "Other")
        return X


class DerivedFeatureAdder(BaseEstimator, TransformerMixin):
    """Age_Years, Credit_Loan_Ratio, has_children, contact_count. Then drops
    Age_Days and Loan_Annuity, now redundant with the derived features.
    Stateless.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["Age_Years"] = X["Age_Days"] / 365
        X["Credit_Loan_Ratio"] = X["Credit_Amount"] / X["Loan_Annuity"]
        X["has_children"] = (X["Child_Count"] > 0).astype(int)
        X["contact_count"] = X["Homephone_Tag"] + X["Workphone_Working"]
        return X.drop(columns=["Age_Days", "Loan_Annuity"])


class ImportanceFeatureSelector(BaseEstimator, TransformerMixin):
    """Fits a quick RandomForest on train, remembers which features have
    importance below `threshold`, and drops them on every future transform -
    including a single new row.
    """

    def __init__(self, threshold: float = LOW_IMPORTANCE_THRESHOLD, random_state: int = RANDOM_STATE):
        self.threshold = threshold
        self.random_state = random_state

    def fit(self, X, y):
        rf = RandomForestClassifier(
            n_estimators=200, max_depth=8, class_weight="balanced", random_state=self.random_state, n_jobs=-1
        )
        rf.fit(X, y)
        self.importances_ = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
        self.features_to_drop_ = self.importances_[self.importances_ < self.threshold].index.tolist()
        return self

    def transform(self, X):
        return X.drop(columns=self.features_to_drop_)


class SelectiveStandardScaler(BaseEstimator, TransformerMixin):
    """StandardScaler restricted to continuous numeric columns (>2 unique
    values in train) - binary/flag/dummy columns are left unscaled. Learns
    which columns to scale, and their mean/std, from train only.
    """

    def fit(self, X, y=None):
        self.cols_to_scale_ = [c for c in X.select_dtypes(include=[np.number]).columns if X[c].nunique() > 2]
        self.scaler_ = StandardScaler()
        self.scaler_.fit(X[self.cols_to_scale_])
        return self

    def transform(self, X):
        X = X.copy()
        X[self.cols_to_scale_] = self.scaler_.transform(X[self.cols_to_scale_])
        return X


def build_feature_engineering_steps() -> list[tuple[str, BaseEstimator]]:
    """The feature-engineering steps from notebook 04, as (name, transformer)
    tuples ready to feed into an sklearn.Pipeline."""
    return [
        ("rare_category_grouper", RareCategoryGrouper()),
        ("grouped_categorical_encoder", CategoricalOneHotEncoder(cols=COLS_RESERVED_FOR_FEATURE_ENGINEERING)),
        ("derived_features", DerivedFeatureAdder()),
        ("feature_selector", ImportanceFeatureSelector()),
        ("scaler", SelectiveStandardScaler()),
    ]


def build_pipeline(model) -> Pipeline:
    """Chains every preprocessing + feature-engineering step (src/preprocessing/)
    and the given model into one sklearn.Pipeline. Fit once on (X_train,
    y_train), the fitted pipeline preprocesses and predicts for a single new
    raw application exactly as it did for training data.

    Usage:
        X_train = train_df.drop(columns=[config.TARGET])
        y_train = train_df[config.TARGET]
        pipe = build_pipeline(LogisticRegression())
        pipe.fit(X_train, y_train)
        pipe.predict_proba(X_test)
    """
    steps = build_preprocessing_steps() + build_feature_engineering_steps() + [("model", model)]
    return Pipeline(steps)
