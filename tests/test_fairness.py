"""Tests for the fairness check: group labels, per-group rates and gaps."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.fairness import gaps, group_columns, group_metrics, without_sensitive


def _applicants():
    return pd.DataFrame(
        {
            "Age_Days": [25 * 365.25, 45 * 365.25, np.nan, 68 * 365.25],
            "Client_Gender": ["Female", "Male", np.nan, "Male"],
            "Client_Marital_Status": ["M", "S", "W", np.nan],
            "Client_Education": ["Secondary", "Graduation", "Secondary", "Graduation"],
            "Client_Income_Type": ["Service", "Service", "Retired", "Retired"],
        }
    )


def test_group_columns_use_readable_names_and_mark_blanks():
    groups = group_columns(_applicants())

    assert list(groups["age_band"]) == ["21-30", "41-50", "Not given", "61-69"]
    assert list(groups["gender"]) == ["Female", "Male", "Not given", "Male"]
    assert list(groups["marital_status"]) == ["Married", "Single", "Widowed", "Not given"]


def test_group_metrics_rates():
    groups = pd.DataFrame({"gender": ["A", "A", "A", "A", "B", "B", "B", "B"]})
    y = [1, 1, 0, 0, 1, 0, 0, 0]
    scores = [0.9, 0.1, 0.8, 0.2, 0.9, 0.9, 0.1, 0.1]

    table = group_metrics(y, scores, 0.5, groups, "m").set_index("group")

    assert table.loc["A", "recall"] == 0.5
    assert table.loc["A", "false_positive_rate"] == 0.5
    assert table.loc["B", "recall"] == 1.0
    assert table.loc["B", "false_positive_rate"] == pytest.approx(1 / 3)
    assert table.loc["B", "precision"] == 0.5
    assert not table["enough_data"].any()  # far too few applicants


def test_gaps_use_only_groups_with_enough_data_and_flag_large_ones():
    by_group = pd.DataFrame(
        {
            "model": ["m"] * 3,
            "attribute": ["gender"] * 3,
            "group": ["A", "B", "tiny"],
            "recall": [0.60, 0.45, 0.0],
            "false_positive_rate": [0.25, 0.22, 1.0],
            "precision": [0.18, 0.17, 0.0],
            "share_flagged": [0.27, 0.25, 1.0],
            "enough_data": [True, True, False],
        }
    )

    row = gaps(by_group).iloc[0]

    assert row["groups_compared"] == 2
    assert row["recall_gap"] == pytest.approx(0.15)
    assert row["gap_above_limit"]


def test_without_sensitive_drops_gender_and_marital_status_and_flattens_age():
    X = without_sensitive(_applicants())

    assert "Client_Gender" not in X and "Client_Marital_Status" not in X
    assert X["Age_Days"].nunique() == 1
