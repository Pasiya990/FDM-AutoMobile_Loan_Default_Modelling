"""
Feature Engineering Module for Automobile Loan Default Prediction.
Creates domain-specific features related to age, debt ratios, and credit scores.
"""

import pandas as pd


def create_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates 7 domain-specific features:
    - Age_Years: Human age in years (older borrowers usually have lower default risk).
    - Employed_Years: Work experience in years (longer job tenure reduces default risk).
    - Annuity_to_Income_Ratio: Payment burden (higher ratio = tighter budget = higher risk).
    - Credit_to_Income_Ratio: Loan leverage compared to income (higher ratio = higher risk).
    - Credit_to_Annuity_Ratio: Estimated loan term (longer loans increase exposure risk).
    - Mean_Bureau_Score: Average credit bureau rating (higher score = lower risk).
    - Bureau_Scores_Count: Number of credit bureau scores available (fewer scores = less history).
    """
    df = df.copy()

    # 1. Convert days to years
    if 'Age_Days' in df.columns:
        # Calculate customer age in years
        df['Age_Years'] = df['Age_Days'] / 365.25

    if 'Employed_Days' in df.columns:
        # Calculate work experience in years
        df['Employed_Years'] = df['Employed_Days'] / 365.25

    # 2. Financial debt ratios (+1 prevents division by zero)
    if 'Client_Income' in df.columns:
        if 'Loan_Annuity' in df.columns:
            # Monthly payment burden relative to income
            df['Annuity_to_Income_Ratio'] = df['Loan_Annuity'] / (df['Client_Income'] + 1)
        if 'Credit_Amount' in df.columns:
            # Total loan borrowed compared to income (leverage)
            df['Credit_to_Income_Ratio'] = df['Credit_Amount'] / (df['Client_Income'] + 1)

    if 'Credit_Amount' in df.columns and 'Loan_Annuity' in df.columns:
        # Estimated loan duration in installments
        df['Credit_to_Annuity_Ratio'] = df['Credit_Amount'] / (df['Loan_Annuity'] + 1)

    # 3. Credit bureau score features
    bureau_cols = [c for c in ['Score_Source_1', 'Score_Source_2', 'Score_Source_3'] if c in df.columns]
    if bureau_cols:
        # Average rating across available credit bureau scores
        df['Mean_Bureau_Score'] = df[bureau_cols].mean(axis=1)
        # Total number of available credit scores (0 to 3)
        df['Bureau_Scores_Count'] = df[bureau_cols].notnull().sum(axis=1)

    return df
