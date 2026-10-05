"""Tests for the final-model script. They do not fit a model or touch the
held-out test set: they check the metric helper and the guard that stops the
test set being reused."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.train_final import METADATA_FILE, evaluate_at_threshold, train_final


def test_evaluate_at_threshold_counts_and_metrics():
    y = np.array([1, 1, 0, 0, 0, 0])
    proba = np.array([0.9, 0.3, 0.8, 0.2, 0.1, 0.05])

    metrics = evaluate_at_threshold(y, proba, threshold=0.5)

    assert metrics["confusion_matrix"] == {"tn": 3, "fp": 1, "fn": 1, "tp": 1}
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["share_flagged"] == pytest.approx(2 / 6)
    assert metrics["threshold"] == 0.5
    assert metrics["rows"] == 6


def test_lower_threshold_catches_more_defaulters():
    y = np.array([1, 1, 0, 0, 0, 0])
    proba = np.array([0.9, 0.3, 0.8, 0.2, 0.1, 0.05])

    assert evaluate_at_threshold(y, proba, 0.25)["recall"] > evaluate_at_threshold(y, proba, 0.5)["recall"]


def test_refuses_to_reuse_the_test_set(tmp_path):
    # An existing metadata file means the test set was already used; the guard
    # must raise before any data is loaded or any model is fitted.
    (tmp_path / METADATA_FILE).write_text("{}", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already been used"):
        train_final(output_dir=tmp_path)
