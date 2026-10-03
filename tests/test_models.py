"""Smoke tests for src/models/*.py: each build_*_pipeline() must produce a
valid sklearn.Pipeline that fits and predicts on the real preprocessed data.
Full CV evaluation lives in notebooks/05_baseline_models.ipynb - these tests
only check the pipeline is wired correctly, on a small sample for speed.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import RANDOM_STATE, TARGET
from src.data.data_cleaning import clean_and_split


@pytest.fixture(scope="module")
def sample_data():
    train_df, _ = clean_and_split()
    sample = train_df.sample(n=2000, random_state=RANDOM_STATE)
    return sample.drop(columns=[TARGET]), sample[TARGET]


def test_random_forest_pipeline_fits_and_predicts(sample_data):
    from src.models.random_forest import build_random_forest_pipeline

    X, y = sample_data
    pipe = build_random_forest_pipeline()
    pipe.fit(X, y)

    assert "model" in pipe.named_steps
    proba = pipe.predict_proba(X.iloc[[0]])[:, 1][0]
    assert 0.0 <= proba <= 1.0


def test_gaussian_naive_bayes_pipeline_fits_and_predicts(sample_data):
    from src.models.gaussian_naive_bayes import build_gaussian_naive_bayes_pipeline

    X, y = sample_data
    pipe = build_gaussian_naive_bayes_pipeline()
    pipe.fit(X, y)

    assert "model" in pipe.named_steps
    proba = pipe.predict_proba(X.iloc[[0]])[:, 1][0]
    assert 0.0 <= proba <= 1.0
