"""Row-level cleaning: fix dtypes, remove duplicates, fix sentinel/placeholder
values, and split into train/test. These steps run once, before the split,
because they either change the row count (duplicates) or are fixed,
data-independent rules (not fit on train) - see notebooks/03_data_preprocessing.ipynb
sections 2-5, which this module mirrors.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    CAR_AGE_COL,
    CAR_OWNED_COL,
    CORRUPTED_ABOVE_ONE_COLS,
    HAS_CAR_AGE_COL,
    ID_COL,
    NUMERIC_TEXT_COLS,
    RANDOM_STATE,
    RAW_DATA_PATH,
    SENTINEL_COL,
    SENTINEL_FLAG_COL,
    SENTINEL_VALUE,
    TARGET,
)


def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def fix_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce the text-numeric columns to numeric, invalid tokens -> NaN."""
    df = df.copy()
    for col in NUMERIC_TEXT_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def remove_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate rows, ignoring ID (every row has a unique ID)."""
    cols_no_id = [c for c in df.columns if c != ID_COL]
    return df.drop_duplicates(subset=cols_no_id).reset_index(drop=True)


def apply_sentinel_and_placeholder_fixes(df: pd.DataFrame) -> pd.DataFrame:
    """Fixed, data-independent rules: sentinel -> flag + missing, disguised
    placeholders -> missing, corrupted values -> missing, Own_House_Age
    reinterpreted as car_age, ID dropped.
    """
    df = df.copy()

    df[SENTINEL_FLAG_COL] = (df[SENTINEL_COL] == SENTINEL_VALUE).astype(int)
    df.loc[df[SENTINEL_COL] == SENTINEL_VALUE, SENTINEL_COL] = np.nan

    # Type_Organization's "XNA" is kept as-is - it's a valid "no employer" category
    df.loc[df["Client_Gender"] == "XNA", "Client_Gender"] = np.nan
    df.loc[df["Accompany_Client"] == "##", "Accompany_Client"] = np.nan

    for col in CORRUPTED_ABOVE_ONE_COLS:
        df.loc[df[col] > 1, col] = np.nan

    # Own_House_Age is actually car age (filled only when Car_Owned == 1)
    df[HAS_CAR_AGE_COL] = (df[CAR_OWNED_COL] == 1).astype(int)
    df = df.rename(columns={"Own_House_Age": CAR_AGE_COL})

    df = df.drop(columns=[ID_COL])
    return df


def split_train_test(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = RANDOM_STATE
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified split on the target. Must run before any step that fits
    on the data (imputation, encoding, scaling, feature selection)."""
    return train_test_split(df, test_size=test_size, stratify=df[TARGET], random_state=random_state)


def clean_and_split(path: Path = RAW_DATA_PATH) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience wrapper chaining every row-level step, in order."""
    df = load_raw_data(path)
    df = fix_dtypes(df)
    df = remove_duplicate_rows(df)
    df = apply_sentinel_and_placeholder_fixes(df)
    return split_train_test(df)
