"""Tests for near-duplicate applicant removal (src/data/near_duplicates.py)
and the class weighting that makes XGBoost comparable to the Random Forest.

The synthetic tests check the matching rules; the real-data tests check the
properties the leak-free split depends on.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import TARGET
from src.data.data_cleaning import clean_and_split, fix_dtypes, load_raw_data, remove_duplicate_rows
from src.data.near_duplicates import FINGERPRINT_KEYS, assign_applicant_groups, remove_near_duplicate_rows
from src.models.xgboost_model import SCALE_POS_WEIGHT, build_xgboost, imbalance_ratio

FINGERPRINT_COLS = sorted({c for key in FINGERPRINT_KEYS for c in key})


def make_rows(n):
    """n distinct applicants: every fingerprint column unique per row."""
    df = pd.DataFrame({c: np.arange(n, dtype=float) * 7 + i for i, c in enumerate(FINGERPRINT_COLS)})
    df["Client_Occupation"] = "Sales"
    df[TARGET] = 0
    return df


def test_copy_with_blanked_value_is_grouped():
    df = make_rows(3)
    copy = df.iloc[[0]].copy()
    copy["Client_Occupation"] = np.nan
    copy["ID_Days"] = np.nan  # breaks one key, others still match
    df = pd.concat([df, copy], ignore_index=True)

    groups = assign_applicant_groups(df)
    assert groups[0] == groups[3]
    assert groups.nunique() == 3


def test_shared_fingerprint_with_conflicting_value_is_not_grouped():
    # Same person's fingerprint but a different loan: not a copy, keep both.
    df = make_rows(2)
    other_loan = df.iloc[[0]].copy()
    other_loan["Client_Occupation"] = "Drivers"
    df = pd.concat([df, other_loan], ignore_index=True)

    assert assign_applicant_groups(df).nunique() == 3


def test_most_complete_copy_is_kept():
    df = make_rows(2)
    blanked = df.iloc[[1]].copy()
    blanked["Client_Occupation"] = np.nan
    df = pd.concat([blanked, df], ignore_index=True)  # blanked copy comes first

    kept = remove_near_duplicate_rows(df)
    assert len(kept) == 2
    assert kept["Client_Occupation"].notna().all()


@pytest.fixture(scope="module")
def raw_without_exact_duplicates():
    return remove_duplicate_rows(fix_dtypes(load_raw_data()))


def test_applicant_groups_are_label_consistent(raw_without_exact_duplicates):
    df = raw_without_exact_duplicates
    groups = assign_applicant_groups(df)
    assert groups.value_counts().max() > 1  # the data does contain copies
    assert (df[TARGET].groupby(groups).nunique() == 1).all()


def test_no_applicant_left_after_removal_has_a_copy(raw_without_exact_duplicates):
    deduped = remove_near_duplicate_rows(raw_without_exact_duplicates)
    assert assign_applicant_groups(deduped).is_unique


def test_split_is_a_partition_of_deduplicated_rows():
    # train and test are disjoint slices of the deduplicated rows, so no
    # applicant can have a copy on the other side
    train_df, test_df = clean_and_split()
    assert not set(train_df.index) & set(test_df.index)
    assert len(train_df) + len(test_df) == 107196


def test_build_xgboost_weights_default_class():
    assert build_xgboost().get_params()["scale_pos_weight"] == SCALE_POS_WEIGHT
    assert build_xgboost(scale_pos_weight=1.0).get_params()["scale_pos_weight"] == 1.0


def test_imbalance_ratio_is_negatives_over_positives():
    assert imbalance_ratio([0, 0, 0, 1]) == 3.0
