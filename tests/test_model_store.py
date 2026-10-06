"""Tests for loading the model artefact. They use a small stand-in model in a
temporary folder, plus one check on the real final model when it exists."""

import json
import os
import sys

import joblib
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.model_store import ModelNotAvailable, load_model_store, version_warnings
from src.models.train_final import FINAL_DIR, METADATA_FILE, MODEL_FILE

METADATA = {
    "model": "stand_in",
    "trained_on": "2026-10-06",
    "operating_threshold": 0.5,
    "input_schema": {},
    "risk_bands": {"low_upper_bound": 0.25, "high_lower_bound": 0.5, "labels": ["Low", "Medium", "High"]},
    "libraries": {},
}


def _write_artefact(folder, metadata=METADATA):
    model = LogisticRegression().fit(np.array([[0.0], [1.0]]), [0, 1])
    joblib.dump(model, folder / MODEL_FILE)
    (folder / METADATA_FILE).write_text(json.dumps(metadata), encoding="utf-8")


def test_loads_pipeline_and_metadata(tmp_path):
    _write_artefact(tmp_path)

    store = load_model_store(tmp_path)

    assert store.operating_threshold == 0.5
    assert store.model_version == "stand_in (2026-10-06)"
    assert store.pipeline.predict_proba([[1.0]]).shape == (1, 2)


def test_missing_model_file_raises_model_not_available(tmp_path):
    (tmp_path / METADATA_FILE).write_text(json.dumps(METADATA), encoding="utf-8")

    with pytest.raises(ModelNotAvailable, match=MODEL_FILE):
        load_model_store(tmp_path)


def test_corrupt_model_file_raises_model_not_available(tmp_path):
    _write_artefact(tmp_path)
    (tmp_path / MODEL_FILE).write_bytes(b"not a model")

    with pytest.raises(ModelNotAvailable, match="could not be loaded"):
        load_model_store(tmp_path)


def test_metadata_without_input_schema_is_rejected(tmp_path):
    _write_artefact(tmp_path, {k: v for k, v in METADATA.items() if k != "input_schema"})

    with pytest.raises(ModelNotAvailable, match="input_schema"):
        load_model_store(tmp_path)


def test_version_mismatch_is_reported():
    messages = version_warnings({"scikit-learn": "0.0.1", "unknown-library": "1.0"})

    assert len(messages) == 1
    assert messages[0].startswith("scikit-learn: trained with 0.0.1")


@pytest.mark.skipif(not (FINAL_DIR / MODEL_FILE).exists(), reason="final model not built on this machine")
def test_real_final_model_loads_with_matching_versions():
    store = load_model_store()

    assert store.metadata["model"] == "xgboost_tuned"
    assert store.version_warnings == []
