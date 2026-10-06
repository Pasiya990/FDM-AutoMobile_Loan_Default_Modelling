"""Tests for the selection-file script. They do not fit a model: they check the
parameter reader and that the selection has everything train_final.py reads."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.select_final import build_selection, load_xgboost_parameters


def test_load_xgboost_parameters_strips_prefix_and_types(tmp_path):
    path = tmp_path / "best.csv"
    pd.DataFrame(
        {"parameter": ["model__max_depth", "model__n_estimators", "model__learning_rate"], "value": [4.0, 400.0, 0.05]}
    ).to_csv(path, index=False)

    params = load_xgboost_parameters(path)

    assert params == {"max_depth": 4, "n_estimators": 400, "learning_rate": 0.05}
    assert isinstance(params["max_depth"], int) and isinstance(params["n_estimators"], int)


def test_selection_reaches_target_recall_and_has_the_keys_train_final_reads():
    rng = np.random.default_rng(0)
    y = pd.Series((rng.random(2000) < 0.1).astype(int))
    proba = np.clip(0.3 * y.to_numpy() + rng.random(2000) * 0.7, 0, 1)
    folds = pd.DataFrame({"roc_auc": [0.74, 0.75], "pr_auc": [0.22, 0.23]})

    selection = build_selection({"max_depth": 4}, y, proba, folds)

    for key in ("model", "family", "params", "selected_by", "operating_threshold", "target_recall", "oof_at_threshold", "cv_reference"):
        assert key in selection
    assert selection["family"] == "xgboost"
    assert selection["params"]["spw_factor"] == 1.0
    assert selection["oof_at_threshold"]["recall"] >= selection["target_recall"] - 1e-9
    assert selection["cv_reference"]["roc_auc_mean"] == pytest.approx(0.745)
