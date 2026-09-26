"""Feature engineering: derived features and rare-category grouping.

Ordering matters for correctness: FeatureEngineer must run right after
LoanCleaner and before any imputation or log-transform, since some of its
features need the genuine (pre-imputation, pre-transform) values to mean
anything - see 03_data_preprocessing.ipynb for the bugs this caught.
"""

from sklearn.base import BaseEstimator, TransformerMixin

from . import config


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Adds derived features, all deterministic formulas (no parameters
    learned from data, so fit() is a no-op):

    - score_mean / score_min / scores_available: need genuine missingness
      in Score_Source_1/2/3 - must run before imputation.
    - credit_to_income / annuity_to_income / income_per_member: need
      dollar-scale Client_Income - must run before its log-transform.
    - Age_Years, Credit_Loan_Ratio, has_children, contact_count: no
      ordering constraint.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        score_cols = [c for c in config.SCORE_COLS if c in df.columns]
        if score_cols:
            df["score_mean"] = df[score_cols].mean(axis=1, skipna=True)
            df["score_min"] = df[score_cols].min(axis=1, skipna=True)
            df["scores_available"] = df[score_cols].notna().sum(axis=1)

        if {"Credit_Amount", "Client_Income"}.issubset(df.columns):
            df["credit_to_income"] = df["Credit_Amount"] / df["Client_Income"]

        if {"Loan_Annuity", "Client_Income"}.issubset(df.columns):
            df["annuity_to_income"] = df["Loan_Annuity"] / df["Client_Income"]

        if {"Client_Income", "Client_Family_Members"}.issubset(df.columns):
            df["income_per_member"] = df["Client_Income"] / df[
                "Client_Family_Members"
            ].clip(lower=1)

        if "Age_Days" in df.columns:
            df["Age_Years"] = df["Age_Days"] / 365

        if {"Credit_Amount", "Loan_Annuity"}.issubset(df.columns):
            df["Credit_Loan_Ratio"] = df["Credit_Amount"] / df["Loan_Annuity"]

        if "Child_Count" in df.columns:
            df["has_children"] = (df["Child_Count"] > 0).astype(int)

        if {"Homephone_Tag", "Workphone_Working"}.issubset(df.columns):
            df["contact_count"] = df["Homephone_Tag"] + df["Workphone_Working"]

        return df


class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    """Learns, per column, which categories occur at least `threshold`
    fraction of the training rows. In transform(), any category not in
    that learned frequent set - including genuinely unseen categories at
    prediction time - is mapped to "Other". Missing values are left as
    missing (handled separately by imputation).
    """

    def __init__(self, cols=None, threshold=None):
        self.cols = cols if cols is not None else config.RARE_CATEGORY_COLS
        self.threshold = (
            threshold if threshold is not None else config.RARE_CATEGORY_THRESHOLD
        )

    def fit(self, X, y=None):
        self.frequent_categories_ = {}
        min_count = len(X) * self.threshold
        for col in self.cols:
            if col in X.columns:
                value_counts = X[col].value_counts()
                self.frequent_categories_[col] = set(
                    value_counts[value_counts >= min_count].index
                )
        return self

    def transform(self, X):
        df = X.copy()
        for col, frequent in self.frequent_categories_.items():
            if col in df.columns:
                keep_mask = df[col].isin(frequent) | df[col].isna()
                df[col] = df[col].where(keep_mask, other="Other")
        return df
