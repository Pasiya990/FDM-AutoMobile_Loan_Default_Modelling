"""Tests for the shared evaluation harness in src/evaluation/metrics.py.

Uses a small synthetic dataset and a plain LogisticRegression, so these
check the harness itself (folds, metrics, logging) quickly and without the
full preprocessing pipeline.
"""

import os
import sys

import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_is_fitted

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.evaluation.metrics import evaluate_model, log_result_to_csv, results_to_row

EXPECTED_KEYS = {
    "model_name", "cv_folds", "threshold", "roc_auc_mean", "roc_auc_std", "pr_auc_mean", "pr_auc_std",
    "recall_mean", "precision_mean", "f1_mean", "confusion_matrix", "fit_seconds",
}


@pytest.fixture(scope="module")
def data():
    X, y = make_classification(n_samples=600, n_features=6, weights=[0.85], random_state=0)
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(6)]), pd.Series(y)


def model():
    return LogisticRegression(max_iter=1000)


def test_returns_all_metrics(data):
    X, y = data
    results = evaluate_model(model(), X, y, model_name="lr", cv=5)
    assert set(results) == EXPECTED_KEYS
    assert results["model_name"] == "lr"
    assert results["cv_folds"] == 5
    assert results["threshold"] == 0.5
    assert 0.5 < results["roc_auc_mean"] <= 1.0


def test_every_row_is_validated_exactly_once(data):
    # The confusion matrix is summed over folds, so it must cover each row once.
    X, y = data
    cm = evaluate_model(model(), X, y)["confusion_matrix"]
    assert sum(sum(row) for row in cm) == len(y)
    assert cm[1][0] + cm[1][1] == int(y.sum())


def test_results_are_reproducible(data):
    X, y = data
    first = evaluate_model(model(), X, y)
    second = evaluate_model(model(), X, y)
    for key in EXPECTED_KEYS - {"fit_seconds"}:
        assert first[key] == second[key]


def test_threshold_changes_recall_but_not_ranking(data):
    X, y = data
    default = evaluate_model(model(), X, y, threshold=0.5)
    flag_all = evaluate_model(model(), X, y, threshold=0.0)
    assert flag_all["recall_mean"] == 1.0
    assert flag_all["roc_auc_mean"] == default["roc_auc_mean"]
    assert flag_all["pr_auc_mean"] == default["pr_auc_mean"]


def test_passed_pipeline_is_not_fitted(data):
    # Each fold works on a clone, so the caller's object stays unfitted.
    X, y = data
    estimator = model()
    evaluate_model(estimator, X, y)
    with pytest.raises(NotFittedError):
        check_is_fitted(estimator)


def test_results_to_row_flattens_confusion_matrix(data):
    X, y = data
    results = evaluate_model(model(), X, y, model_name="lr")
    row = results_to_row(results)
    (tn, fp), (fn, tp) = results["confusion_matrix"]
    assert len(row) == 1
    assert "confusion_matrix" not in row.columns
    assert row.loc[0, ["tn", "fp", "fn", "tp"]].tolist() == [tn, fp, fn, tp]


def test_log_creates_file(tmp_path, data):
    X, y = data
    path = tmp_path / "experiments.csv"
    log_result_to_csv(evaluate_model(model(), X, y, model_name="lr"), str(path))
    log = pd.read_csv(path)
    assert log["model_name"].tolist() == ["lr"]


def test_relogging_a_model_replaces_its_row(tmp_path, data):
    X, y = data
    path = str(tmp_path / "experiments.csv")
    log_result_to_csv(evaluate_model(model(), X, y, model_name="lr", threshold=0.5), path)
    log_result_to_csv(evaluate_model(model(), X, y, model_name="lr", threshold=0.3), path)
    log = pd.read_csv(path)
    assert log["model_name"].tolist() == ["lr"]
    assert log.loc[0, "threshold"] == 0.3


def test_logging_another_model_keeps_existing_rows(tmp_path, data):
    X, y = data
    path = str(tmp_path / "experiments.csv")
    log_result_to_csv(evaluate_model(model(), X, y, model_name="lr"), path)
    log_result_to_csv(evaluate_model(model(), X, y, model_name="other"), path)
    assert sorted(pd.read_csv(path)["model_name"]) == ["lr", "other"]
