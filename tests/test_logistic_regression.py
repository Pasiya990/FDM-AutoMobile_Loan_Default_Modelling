"""Smoke tests for the isolated Logistic Regression baseline."""

import os
import sys

import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import RANDOM_STATE, TARGET
from src.data.data_cleaning import clean_and_split
from src.models.logistic_regression import build_logistic_regression_pipeline


@pytest.fixture(scope="module")
def sample_data():
    train_df, test_df = clean_and_split()
    train_sample = train_df.sample(n=2000, random_state=RANDOM_STATE)
    X_train = train_sample.drop(columns=[TARGET])
    y_train = train_sample[TARGET]
    X_test = test_df.drop(columns=[TARGET])
    return X_train, y_train, X_test


def test_builder_uses_expected_baseline_configuration():
    pipeline = build_logistic_regression_pipeline()

    assert isinstance(pipeline, Pipeline)
    assert isinstance(pipeline.named_steps["model"], LogisticRegression)
    assert pipeline.named_steps["model"].class_weight == "balanced"
    assert pipeline.named_steps["model"].max_iter == 1000
    assert pipeline.named_steps["model"].random_state == RANDOM_STATE


def test_pipeline_fits_and_predicts_single_raw_row(sample_data):
    X_train, y_train, X_test = sample_data
    pipeline = build_logistic_regression_pipeline()
    pipeline.fit(X_train, y_train)

    probability = pipeline.predict_proba(X_test.iloc[[0]])[:, 1][0]

    assert 0.0 <= probability <= 1.0
    assert list(pipeline.classes_) == [0, 1]
