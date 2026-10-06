"""Tests for the top-factor explanations on the real final model."""

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.adapter import build_model_row
from backend.examples import EXAMPLES_PATH
from backend.explain import contributions, factor_of, top_factors
from src.models.train_final import FINAL_DIR, MODEL_FILE

pytestmark = pytest.mark.skipif(not (FINAL_DIR / MODEL_FILE).exists(), reason="final model not built")


@pytest.fixture(scope="module")
def store():
    from backend.model_store import load_model_store

    return load_model_store()


@pytest.fixture(scope="module")
def example_rows(store):
    examples = json.loads(EXAMPLES_PATH.read_text(encoding="utf-8"))
    return [build_model_row(e["application"], store.metadata["input_columns"]) for e in examples]


def test_contributions_add_up_to_the_model_score(store, example_rows):
    for row in example_rows:
        raw = contributions(store.pipeline, row).sum()
        score = store.pipeline.predict_proba(row)[0, 1]
        assert 1 / (1 + np.exp(-raw)) == pytest.approx(score, abs=1e-5)


def test_every_model_feature_has_a_readable_factor(store, example_rows):
    features = contributions(store.pipeline, example_rows[0]).drop("base").index
    assert [f for f in features if factor_of(f) == "Other details"] == []


def test_one_hot_and_flag_columns_map_to_their_field():
    assert factor_of("Type_Organization_Self-employed") == "Employer type"
    assert factor_of("Client_Income_Type_Govt Job") == "Income type"
    assert factor_of("Score_Source_1_was_missing") == "External credit scores"
    assert factor_of("Client_Income") == "Income"


def test_top_factors_are_ordered_and_signed(store, example_rows):
    factors = top_factors(store.pipeline, example_rows[2])

    assert 1 <= len(factors) <= 4
    impacts = [abs(f["impact"]) for f in factors]
    assert impacts == sorted(impacts, reverse=True)
    for f in factors:
        assert f["effect"] == ("raises risk" if f["impact"] > 0 else "lowers risk")
