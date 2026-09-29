"""Fit-on-train preprocessing, as sklearn-compatible transformers: score
summaries, missing-value imputation, income-ratio features, outlier
treatment, categorical encoding. Each stateful step learns its parameters
in fit() (from training data only) and reapplies them in transform(), so a
fitted pipeline preprocesses one new raw application exactly as it
preprocessed the training data - see notebooks/03_data_preprocessing.ipynb
sections 6-10, which this module mirrors.

X is assumed to never contain the target column (split off as y before
fitting), matching standard sklearn.Pipeline usage.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.config import (
    CAR_AGE_COL,
    COLS_RESERVED_FOR_FEATURE_ENGINEERING,
    MISSING_FLAG_THRESHOLD,
    SCORE_COLS,
)


class ScoreSummaryAdder(BaseEstimator, TransformerMixin):
    """score_mean/score_min/scores_available, computed before imputation so
    they reflect the genuine missingness pattern in Score_Source_1/2/3.
    Stateless - nothing is learned from train.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["score_mean"] = X[SCORE_COLS].mean(axis=1, skipna=True)
        X["score_min"] = X[SCORE_COLS].min(axis=1, skipna=True)
        X["scores_available"] = X[SCORE_COLS].notna().sum(axis=1)
        return X


class MissingValueImputer(BaseEstimator, TransformerMixin):
    """Numeric -> median (learned from train), with a '_was_missing' flag
    for columns whose train missing rate exceeds flag_threshold (car_age
    excluded, since has_car_age already serves that purpose, and it's
    filled with 0, not the median). Categorical -> 'Missing' category.
    """

    def __init__(self, flag_threshold: float = MISSING_FLAG_THRESHOLD):
        self.flag_threshold = flag_threshold

    def fit(self, X, y=None):
        numeric_cols_missing = [
            c for c in X.select_dtypes(include=[np.number]).columns if c != "has_car_age" and X[c].isna().sum() > 0
        ]
        self.categorical_cols_missing_ = [
            c for c in X.select_dtypes(include=["object", "string"]).columns if X[c].isna().sum() > 0
        ]
        self.flag_cols_ = [
            c for c in numeric_cols_missing if c != CAR_AGE_COL and X[c].isna().mean() > self.flag_threshold
        ]
        self.fill_values_ = {c: (0 if c == CAR_AGE_COL else X[c].median()) for c in numeric_cols_missing}
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.flag_cols_:
            X[col + "_was_missing"] = X[col].isna().astype(int)
        for col, fill_value in self.fill_values_.items():
            X[col] = X[col].fillna(fill_value)
        for col in self.categorical_cols_missing_:
            X[col] = X[col].fillna("Missing")
        return X


class IncomeRatioAdder(BaseEstimator, TransformerMixin):
    """credit_to_income/annuity_to_income/income_per_member, computed before
    the Client_Income log-transform so they use genuine dollar-scale income.
    Stateless.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["credit_to_income"] = X["Credit_Amount"] / X["Client_Income"]
        X["annuity_to_income"] = X["Loan_Annuity"] / X["Client_Income"]
        X["income_per_member"] = X["Client_Income"] / X["Client_Family_Members"].clip(lower=1)
        return X


class IncomeLogTransformer(BaseEstimator, TransformerMixin):
    """log1p(Client_Income) - corrects extreme right skew. Deterministic,
    nothing learned from train.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["Client_Income"] = np.log1p(X["Client_Income"])
        return X


class CategoricalOneHotEncoder(BaseEstimator, TransformerMixin):
    """One-hot encode categorical columns, with the dummy-column set learned
    from train, so a single new row always produces exactly the columns
    training produced - unseen categories, or categories simply absent from
    this particular row, both come out as 0.

    Pass `cols` to encode exactly those columns (used for the grouped
    Type_Organization/Client_Education/etc. columns after RareCategoryGrouper),
    or `exclude_cols` to encode every object/string column except those (used
    for the general categorical columns). Reused for both.
    """

    def __init__(self, cols=None, exclude_cols=None):
        self.cols = cols
        self.exclude_cols = exclude_cols

    def fit(self, X, y=None):
        if self.cols is not None:
            self.cols_to_encode_ = list(self.cols)
        else:
            exclude = self.exclude_cols or []
            self.cols_to_encode_ = [
                c for c in X.select_dtypes(include=["object", "string"]).columns if c not in exclude
            ]
        dummies = pd.get_dummies(X, columns=self.cols_to_encode_, drop_first=True)
        self.output_columns_ = dummies.columns.tolist()
        return self

    def transform(self, X):
        X = pd.get_dummies(X, columns=self.cols_to_encode_, drop_first=True)
        X = X.reindex(columns=self.output_columns_, fill_value=0)
        dummy_cols = [c for c in X.columns if X[c].dtype == bool]
        X[dummy_cols] = X[dummy_cols].astype(int)
        return X


def build_preprocessing_steps() -> list[tuple[str, BaseEstimator]]:
    """The fit-on-train steps from notebook 03, sections 6-10, as (name,
    transformer) tuples ready to feed into an sklearn.Pipeline."""
    return [
        ("score_summary", ScoreSummaryAdder()),
        ("imputer", MissingValueImputer()),
        ("income_ratios", IncomeRatioAdder()),
        ("income_log", IncomeLogTransformer()),
        ("categorical_encoder", CategoricalOneHotEncoder(exclude_cols=COLS_RESERVED_FOR_FEATURE_ENGINEERING)),
    ]
