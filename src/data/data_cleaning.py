"""
Data Cleaning Module for Automobile Loan Default Prediction.

Functions:
1. clean_corrupted_numerical_columns: Safely converts string-corrupted numbers to float.
2. handle_employed_sentinel: Replaces 365243 placeholder with NaN and creates a binary flag.
3. remove_identifiers: Drops non-predictive ID column.
4. clean_raw_data: Runs all cleaning steps in one call.
"""

import numpy as np
import pandas as pd


def clean_corrupted_numerical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert numerical columns with rogue symbols ('$', 'x', '#VALUE!', '&') to float.
    Uses pd.to_numeric(errors='coerce') to safely replace corrupted values with NaN.
    """
    # Make a copy so original data is not modified
    df = df.copy()

    # List of columns where numbers were accidentally saved with text symbols ($, x, &)
    corrupted_cols = [
        'Client_Income',
        'Credit_Amount',
        'Loan_Annuity',
        'Age_Days',
        'Employed_Days',
        'Score_Source_3',
        'Registration_Days',
        'ID_Days',
        'Population_Region_Relative',
    ]

    for col in corrupted_cols:
        if col in df.columns:
            # Convert text numbers to float; rogue symbols are safely turned into NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


def handle_employed_sentinel(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle the 365,243 placeholder in 'Employed_Days'.

    Why 365243 is a placeholder:
    - 365,243 days is roughly 1,000 years (365,243 / 365.25 ≈ 1,000).
    - It is a database placeholder used for applicants with no active job.
    - Most of these applicants (96.7%) are Retired, while a few are Unemployed.

    Why it cannot be left as raw days:
    - Leaving 365,243 as raw days severely distorts machine learning models
      by making it look like 1,000 years of work experience instead of retirement.

    Solution:
    1. Create a binary flag 'Is_Retired_Unemployed' (1 if 365243, 0 otherwise).
    2. Replace 365243 with NaN in 'Employed_Days'.
    """
    # Make a copy of the dataframe
    df = df.copy()

    if 'Employed_Days' in df.columns:
        # Ensure Employed_Days is a numeric column first
        df['Employed_Days'] = pd.to_numeric(df['Employed_Days'], errors='coerce')

        # Step 1: Create a 0/1 flag (1 = retired/unemployed, 0 = normally employed)
        df['Is_Retired_Unemployed'] = (df['Employed_Days'] == 365243).astype(int)

        # Step 2: Replace 365243 with NaN so it doesn't trick models into thinking 1000 years worked
        df['Employed_Days'] = df['Employed_Days'].replace(365243, np.nan)

    return df


def remove_identifiers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop the 'ID' column.
    ID is an arbitrary database key with no predictive value.
    Keeping it risks data leakage and model overfitting.
    """
    # Make a copy of the dataframe
    df = df.copy()

    # Drop ID because it's just an account number and has no predictive value
    if 'ID' in df.columns:
        df = df.drop(columns=['ID'])

    return df


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """Run all cleaning steps on raw data."""
    # 1. Fix columns with rogue symbols and convert to float
    df = clean_corrupted_numerical_columns(df)

    # 2. Fix the 365,243 placeholder and add Is_Retired_Unemployed flag
    df = handle_employed_sentinel(df)

    # 3. Drop the non-predictive ID column
    df = remove_identifiers(df)

    return df
