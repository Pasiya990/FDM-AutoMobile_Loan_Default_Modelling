"""Tests for the serving information in metadata.json (no model fitting, no test set)."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.model_metadata import build_input_schema, build_risk_bands, column_schema


def _frame():
    rng = np.random.default_rng(0)
    income = rng.lognormal(10, 0.5, 1000)
    income[:20] = np.nan
    return pd.DataFrame(
        {
            "Client_Income": income,
            "Child_Count": rng.integers(0, 5, 1000).astype(float),
            "Car_Owned": rng.integers(0, 2, 1000),
            "Client_Gender": rng.choice(["Female", "Male"], 1000),
            "Is_Retired_Or_Unemployed": rng.integers(0, 2, 1000),
        }
    )


def test_every_column_is_described_with_a_kind():
    X = _frame()
    schema = build_input_schema(X)

    assert list(schema) == list(X.columns)
    assert schema["Client_Gender"]["kind"] == "category"
    assert schema["Client_Gender"]["allowed"] == ["Female", "Male"]
    assert schema["Car_Owned"]["kind"] == "binary"
    assert schema["Child_Count"]["kind"] == "integer"
    assert schema["Client_Income"]["kind"] == "number"


def test_hard_limits_are_the_training_extremes_and_contain_the_typical_range():
    X = _frame()
    spec = column_schema(X["Client_Income"])

    assert spec["min"] == pytest.approx(X["Client_Income"].min(), rel=1e-4)
    assert spec["max"] == pytest.approx(X["Client_Income"].max(), rel=1e-4)
    assert spec["min"] <= spec["typical_min"] <= spec["typical_max"] <= spec["max"]
    assert X["Client_Income"].dropna().between(spec["min"], spec["max"]).all()


def test_missing_rate_and_derived_flag():
    schema = build_input_schema(_frame())

    assert schema["Client_Income"]["missing_rate"] == pytest.approx(0.02)
    assert schema["Is_Retired_Or_Unemployed"]["derived"] is True
    assert "derived" not in schema["Client_Income"]


def test_risk_bands_follow_the_operating_threshold():
    bands = build_risk_bands(0.52)

    assert bands["low_upper_bound"] == 0.25
    assert bands["high_lower_bound"] == 0.52
    assert bands["labels"] == ["Low", "Medium", "High"]


def test_band_outcomes_come_from_the_out_of_fold_predictions(tmp_path):
    path = tmp_path / "oof.csv"
    pd.DataFrame({"y_true": [0, 0, 0, 1, 0, 1], "xgboost_tuned": [0.1, 0.2, 0.3, 0.4, 0.6, 0.7]}).to_csv(path, index=False)

    bands = build_risk_bands(0.52, path)

    assert bands["outcomes"]["Low"] == {"share_of_applicants": pytest.approx(2 / 6, abs=1e-4), "default_rate": 0.0}
    assert bands["outcomes"]["Medium"]["default_rate"] == 0.5
    assert bands["outcomes"]["High"]["default_rate"] == 0.5
    assert "outcomes" not in build_risk_bands(0.52, tmp_path / "missing.csv")
