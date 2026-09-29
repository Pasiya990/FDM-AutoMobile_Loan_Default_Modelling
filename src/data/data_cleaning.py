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

RAW_DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "Train_Dataset.csv"

TARGET = "Default"
ID_COL = "ID"

# Columns found to be numeric but stored as text due to stray characters
# ('$', 'x', '&', '#VALUE!', '@', '#') - see notebooks/01_data_understanding.ipynb
NUMERIC_TEXT_COLS = [
    "Client_Income",
    "Credit_Amount",
    "Loan_Annuity",
    "Population_Region_Relative",
    "Age_Days",
    "Employed_Days",
    "Registration_Days",
    "ID_Days",
    "Score_Source_3",
]

# Employed_Days sentinel: 365243 marks retired/unemployed applicants
SENTINEL_COL = "Employed_Days"
SENTINEL_VALUE = 365243
SENTINEL_FLAG_COL = "Is_Retired_Or_Unemployed"

# Corrupted numeric values found via EDA (exact value 100, should be <=1)
CORRUPTED_ABOVE_ONE_COLS = ["Score_Source_2", "Population_Region_Relative"]

CAR_AGE_COL = "car_age"
HAS_CAR_AGE_COL = "has_car_age"
CAR_OWNED_COL = "Car_Owned"

RANDOM_STATE = 42


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
