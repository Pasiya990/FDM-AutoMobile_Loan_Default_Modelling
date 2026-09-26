"""LoanCleaner: value-level cleaning that must run identically on the
training data and on a single new application at prediction time. Row
removal (duplicates) is NOT here - it lives in prepare_data.py, because a
transformer must never change the row count.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from . import config


class LoanCleaner(BaseEstimator, TransformerMixin):
    """Fixed, non-data-dependent cleaning rules found in EDA:
    - parse numeric-looking text columns (the backend receives raw text,
      so this must happen here too, not just in prepare_data.py)
    - Employed_Days sentinel (365243) -> Is_Retired_Or_Unemployed flag + missing
    - disguised missing values (XNA, ##) -> missing
    - corrupted numeric values (>1 where the range should be 0-1) -> missing
    - Own_House_Age reinterpreted as car_age (tied to Car_Owned, not House_Own)
    - ID dropped (non-predictive identifier)

    No parameters are learned from data, so fit() is a no-op.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        for col in config.NUMERIC_TEXT_COLS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        if config.SENTINEL_COL in df.columns:
            is_sentinel = df[config.SENTINEL_COL] == config.SENTINEL_VALUE
            df[config.SENTINEL_FLAG_COL] = is_sentinel.astype(int)
            df.loc[is_sentinel, config.SENTINEL_COL] = np.nan

        for col, tokens in config.PLACEHOLDER_TOKENS.items():
            if col in df.columns:
                df.loc[df[col].isin(tokens), col] = np.nan

        for col in config.CORRUPTED_ABOVE_ONE_COLS:
            if col in df.columns:
                df.loc[df[col] > 1, col] = np.nan

        car_age_source = list(config.CAR_AGE_RENAME.keys())[0]
        if car_age_source in df.columns and config.CAR_OWNED_COL in df.columns:
            df[config.HAS_CAR_AGE_COL] = (df[config.CAR_OWNED_COL] == 1).astype(int)
            df = df.rename(columns=config.CAR_AGE_RENAME)

        drop_cols = [c for c in config.DROP_COLS if c in df.columns]
        df = df.drop(columns=drop_cols)

        return df
