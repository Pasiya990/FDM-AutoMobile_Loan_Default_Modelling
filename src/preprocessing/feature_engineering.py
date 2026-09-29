"""Feature engineering: rare-category grouping, derived features, RF-importance
feature selection, scaling. Runs after src/preprocessing/preprocessing.py -
see notebooks/04_feature_engineering.ipynb, which this module mirrors.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from src.config import (
    COLS_RESERVED_FOR_FEATURE_ENGINEERING,
    LOW_IMPORTANCE_THRESHOLD,
    RANDOM_STATE,
    RARE_CATEGORY_MIN_FREQ,
    TARGET,
)


def group_rare_categories(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cols: list[str] = COLS_RESERVED_FOR_FEATURE_ENGINEERING,
    min_freq: float = RARE_CATEGORY_MIN_FREQ,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Categories below min_freq of training rows get grouped into 'Other'.
    Threshold fit on train only, applied to both."""
    train_df = train_df.copy()
    test_df = test_df.copy()

    min_count = len(train_df) * min_freq
    for col in cols:
        value_counts = train_df[col].value_counts()
        rare_categories = value_counts[value_counts < min_count].index.tolist()
        train_df[col] = train_df[col].replace(rare_categories, "Other")
        test_df[col] = test_df[col].replace(rare_categories, "Other")

    return train_df, test_df


def encode_grouped_categoricals(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cols: list[str] = COLS_RESERVED_FOR_FEATURE_ENGINEERING,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One-hot encode the (now grouped) reserved categorical columns."""
    train_df = train_df.copy()
    test_df = test_df.copy()

    train_df = pd.get_dummies(train_df, columns=cols, drop_first=True)
    test_df = pd.get_dummies(test_df, columns=cols, drop_first=True)

    train_df, test_df = train_df.align(test_df, join="left", axis=1, fill_value=0)

    dummy_cols = [c for c in train_df.columns if train_df[c].dtype == bool]
    train_df[dummy_cols] = train_df[dummy_cols].astype(int)
    test_df[dummy_cols] = test_df[dummy_cols].astype(int)

    return train_df, test_df


def derive_features(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Age_Years, Credit_Loan_Ratio, has_children, contact_count. Then drops
    Age_Days and Loan_Annuity, now redundant with the derived features."""
    train_df = train_df.copy()
    test_df = test_df.copy()

    for d in (train_df, test_df):
        d["Age_Years"] = d["Age_Days"] / 365
        d["Credit_Loan_Ratio"] = d["Credit_Amount"] / d["Loan_Annuity"]
        d["has_children"] = (d["Child_Count"] > 0).astype(int)
        d["contact_count"] = d["Homephone_Tag"] + d["Workphone_Working"]

    train_df = train_df.drop(columns=["Age_Days", "Loan_Annuity"])
    test_df = test_df.drop(columns=["Age_Days", "Loan_Annuity"])

    return train_df, test_df


def select_features_by_importance(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    threshold: float = LOW_IMPORTANCE_THRESHOLD,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Fit a quick RandomForest on train, drop features with importance
    below `threshold`. Returns the importances (full, pre-drop) too."""
    X_train = train_df.drop(columns=[TARGET])
    y_train = train_df[TARGET]

    rf = RandomForestClassifier(
        n_estimators=200, max_depth=8, class_weight="balanced", random_state=random_state, n_jobs=-1
    )
    rf.fit(X_train, y_train)

    importances = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    features_to_drop = importances[importances < threshold].index.tolist()

    train_df = train_df.drop(columns=features_to_drop)
    test_df = test_df.drop(columns=features_to_drop)

    return train_df, test_df, importances


def scale_numeric_features(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """StandardScaler on continuous numeric columns (>2 unique values),
    fit on train only. Binary/flag/dummy columns are left unscaled."""
    train_df = train_df.copy()
    test_df = test_df.copy()

    numeric_cols_to_scale = [
        c for c in train_df.select_dtypes(include=[np.number]).columns if c != TARGET and train_df[c].nunique() > 2
    ]

    scaler = StandardScaler()
    train_df[numeric_cols_to_scale] = scaler.fit_transform(train_df[numeric_cols_to_scale])
    test_df[numeric_cols_to_scale] = scaler.transform(test_df[numeric_cols_to_scale])

    return train_df, test_df


def engineer_features(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Convenience wrapper chaining every feature-engineering step, in order."""
    train_df, test_df = group_rare_categories(train_df, test_df)
    train_df, test_df = encode_grouped_categoricals(train_df, test_df)
    train_df, test_df = derive_features(train_df, test_df)
    train_df, test_df, importances = select_features_by_importance(train_df, test_df)
    train_df, test_df = scale_numeric_features(train_df, test_df)
    return train_df, test_df, importances
