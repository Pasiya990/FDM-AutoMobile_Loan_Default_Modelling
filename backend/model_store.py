"""Loads the final model and its metadata once, when the API starts.

The fitted pipeline (models/final/final_pipeline.joblib) holds every
preprocessing step and the model, so the API applies exactly the preprocessing
used in training. metadata.json holds the operating threshold, the risk bands
and the input schema.

If either file is missing or unreadable, ModelNotAvailable is raised and the API
answers 503 instead of crashing. If an installed library version differs from
the one the model was trained with, a warning is logged: a pickled scikit-learn
or XGBoost model can behave differently, or fail to load, under another version.
"""

import json
import logging
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import joblib

from src.models.train_final import FINAL_DIR, METADATA_FILE, MODEL_FILE

logger = logging.getLogger(__name__)

# Libraries whose version affects how the saved pipeline loads and predicts
CHECKED_LIBRARIES = ("scikit-learn", "xgboost", "pandas", "numpy", "joblib")


class ModelNotAvailable(Exception):
    """The model or its metadata could not be loaded."""


@dataclass
class ModelStore:
    pipeline: object
    metadata: dict
    version_warnings: list = field(default_factory=list)

    @property
    def model_version(self) -> str:
        return f"{self.metadata['model']} ({self.metadata['trained_on']})"

    @property
    def operating_threshold(self) -> float:
        return float(self.metadata["operating_threshold"])


def version_warnings(trained_with: dict) -> list:
    """One message per checked library whose installed version differs from training."""
    messages = []
    for library in CHECKED_LIBRARIES:
        expected = trained_with.get(library)
        if expected is None:
            continue
        try:
            installed = version(library)
        except PackageNotFoundError:
            installed = "not installed"
        if installed != expected:
            messages.append(f"{library}: trained with {expected}, installed {installed}")
    return messages


def load_model_store(model_dir=FINAL_DIR) -> ModelStore:
    """Load the pipeline and metadata from `model_dir`, or raise ModelNotAvailable."""
    model_dir = Path(model_dir)
    model_path, metadata_path = model_dir / MODEL_FILE, model_dir / METADATA_FILE

    for path in (model_path, metadata_path):
        if not path.exists():
            raise ModelNotAvailable(f"{path.name} not found in {model_dir}. Run train_final.py first.")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        pipeline = joblib.load(model_path)
    except Exception as error:  # a corrupt file or an incompatible library version
        raise ModelNotAvailable(f"Model artefact could not be loaded: {error}") from error

    for key in ("operating_threshold", "input_schema", "risk_bands"):
        if key not in metadata:
            raise ModelNotAvailable(f"metadata.json has no '{key}'. Run python -m src.models.model_metadata.")

    warnings = version_warnings(metadata.get("libraries", {}))
    for message in warnings:
        logger.warning("Library version differs from training - %s", message)
    return ModelStore(pipeline=pipeline, metadata=metadata, version_warnings=warnings)
