"""build_pipeline(model, scale): chains LoanCleaner -> FeatureEngineer ->
RareCategoryGrouper -> impute -> log-transform -> drop redundant columns
-> encode -> scale -> (optional) importance-based selector -> model.

Usage:
    X_train = train_df.drop(columns=[config.TARGET])
    y_train = train_df[config.TARGET]
    pipe = build_pipeline(LogisticRegression())
    pipe.fit(X_train, y_train)
    pipe.predict(X_test)

    # preprocessing only, e.g. for a single raw application:
    pipe[:-1].transform(single_row_df)
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config
from .cleaning import LoanCleaner
from .features import FeatureEngineer, RareCategoryGrouper


class MissingValueImputer(BaseEstimator, TransformerMixin):
    """Numeric -> median (fit on train), with a "_was_missing" flag added
    when the missing rate exceeds `flag_threshold` (car_age is excluded,
    since has_car_age already serves that purpose). Categorical ->
    "Missing" category.
    """

    def __init__(self, flag_threshold=None):
        self.flag_threshold = (
            flag_threshold if flag_threshold is not None else config.MISSING_FLAG_THRESHOLD
        )

    def fit(self, X, y=None):
        self.numeric_fills_ = {}
        self.flag_cols_ = []
        numeric_cols = X.select_dtypes(include="number").columns
        for col in numeric_cols:
            if col == config.TARGET or X[col].isna().sum() == 0:
                continue
            if col != config.CAR_AGE_COL and X[col].isna().mean() > self.flag_threshold:
                self.flag_cols_.append(col)
            fill_value = 0 if col == config.CAR_AGE_COL else X[col].median()
            self.numeric_fills_[col] = fill_value

        self.categorical_cols_ = [
            c
            for c in X.select_dtypes(include=["object", "string"]).columns
            if X[c].isna().sum() > 0
        ]
        return self

    def transform(self, X):
        df = X.copy()
        for col in self.flag_cols_:
            if col in df.columns:
                df[f"{col}_was_missing"] = df[col].isna().astype(int)
        for col, fill_value in self.numeric_fills_.items():
            if col in df.columns:
                df[col] = df[col].fillna(fill_value)
        for col in self.categorical_cols_:
            if col in df.columns:
                df[col] = df[col].fillna("Missing")
        return df


class LogTransformer(BaseEstimator, TransformerMixin):
    """log1p on skewed financial columns (Client_Income). Must run after
    the income-ratio features in FeatureEngineer, not before.
    """

    def __init__(self, cols=None):
        self.cols = cols if cols is not None else config.LOG_TRANSFORM_COLS

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        for col in self.cols:
            if col in df.columns:
                df[col] = np.log1p(df[col])
        return df


class ColumnDropper(BaseEstimator, TransformerMixin):
    """Drops fixed, known-redundant columns (e.g. Age_Days once Age_Years
    exists, Loan_Annuity once Credit_Loan_Ratio exists).
    """

    def __init__(self, cols=None):
        self.cols = cols if cols is not None else config.REDUNDANT_COLS_TO_DROP

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        drop = [c for c in self.cols if c in X.columns]
        return X.drop(columns=drop)


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """One-hot encoding, fit on train (learns the dummy-column set),
    applied consistently to any new data - including a single row, where
    reindexing fills any dummy column absent from that row with 0.
    """

    def fit(self, X, y=None):
        self.cat_cols_ = X.select_dtypes(include=["object", "string"]).columns.tolist()
        dummies = pd.get_dummies(X[self.cat_cols_], columns=self.cat_cols_, drop_first=True)
        self.dummy_columns_ = dummies.columns.tolist()
        return self

    def transform(self, X):
        df = X.copy()
        cat_cols = [c for c in self.cat_cols_ if c in df.columns]
        dummies = pd.get_dummies(df[cat_cols], columns=cat_cols, drop_first=True)
        dummies = dummies.reindex(columns=self.dummy_columns_, fill_value=0)
        dummies = dummies.astype(int)
        df = df.drop(columns=cat_cols)
        return pd.concat([df, dummies], axis=1)


class Scaler(BaseEstimator, TransformerMixin):
    """StandardScaler on continuous numeric columns (more than 2 unique
    values, so binary/dummy flags are skipped), fit on train only.
    Set scale=False to skip entirely (e.g. for tree-based models).
    """

    def __init__(self, scale=True):
        self.scale = scale

    def fit(self, X, y=None):
        if not self.scale:
            return self
        self.cols_ = [
            c
            for c in X.select_dtypes(include="number").columns
            if c != config.TARGET and X[c].nunique() > 2
        ]
        self.scaler_ = StandardScaler()
        self.scaler_.fit(X[self.cols_])
        return self

    def transform(self, X):
        if not self.scale:
            return X
        df = X.copy()
        df[self.cols_] = self.scaler_.transform(df[self.cols_])
        return df


class ImportanceSelector(BaseEstimator, TransformerMixin):
    """Fits a quick Random Forest to rank features and keeps only those
    with importance >= threshold. Optional - set select_features=False
    in build_pipeline to skip.
    """

    def __init__(self, threshold=None, random_state=None):
        self.threshold = (
            threshold if threshold is not None else config.LOW_IMPORTANCE_THRESHOLD
        )
        self.random_state = (
            random_state if random_state is not None else config.RANDOM_STATE
        )

    def fit(self, X, y):
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            class_weight="balanced",
            random_state=self.random_state,
            n_jobs=-1,
        )
        rf.fit(X, y)
        importances = pd.Series(rf.feature_importances_, index=X.columns)
        self.keep_cols_ = importances[importances >= self.threshold].index.tolist()
        return self

    def transform(self, X):
        return X[[c for c in self.keep_cols_ if c in X.columns]]


def build_pipeline(model, scale=True, select_features=True):
    steps = [
        ("cleaner", LoanCleaner()),
        ("feature_engineer", FeatureEngineer()),
        ("rare_category_grouper", RareCategoryGrouper()),
        ("imputer", MissingValueImputer()),
        ("log_transform", LogTransformer()),
        ("redundant_dropper", ColumnDropper()),
        ("encoder", CategoricalEncoder()),
        ("scaler", Scaler(scale=scale)),
    ]
    if select_features:
        steps.append(("selector", ImportanceSelector()))
    steps.append(("model", model))
    return Pipeline(steps)
