"""Serving information saved next to the final model in metadata.json.

The backend validates a new application against `input_schema` and the form
builds its dropdowns and limits from it, so the allowed values are written once,
from the training data, and never typed in two places. `risk_bands` records the
cut-offs used to turn the risk score into Low / Medium / High.

Per input column the schema records:
    category : the values seen in training ("allowed")
    binary   : a 0/1 flag
    integer / number : hard limits "min" and "max" (the training extremes, so no
               real application the model has seen is rejected) and the typical
               range "typical_min" / "typical_max" (1st to 99th percentile), meant
               for hints and warnings, not for rejecting input
    missing_rate : share of training rows left blank. The pipeline imputes blanks
               itself, so a blank is never a model error.
    derived  : true when the value is computed from other inputs during cleaning
               and is not typed by the user.

This reads the training split only (the held-out test rows are discarded) and
fits no model, so it can be re-run on an existing metadata.json.

Usage (from the project root): python -m src.models.model_metadata
"""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from src.config import HAS_CAR_AGE_COL, LOW_RISK_UPPER_BOUND, SENTINEL_FLAG_COL, TARGET

TYPICAL_LOW_QUANTILE = 0.01
TYPICAL_HIGH_QUANTILE = 0.99
DERIVED_COLUMNS = (SENTINEL_FLAG_COL, HAS_CAR_AGE_COL)


def _number(value, rounding=round):
    """A plain Python number for JSON (integers stay integers).

    Pass math.floor / math.ceil style rounding for limits, so rounding never
    moves a limit inside a value that was seen in training."""
    value = float(value)
    if value.is_integer():
        return int(value)
    return rounding(value * 1e6) / 1e6


def column_schema(series: pd.Series) -> dict:
    """Allowed values or limits for one input column."""
    spec = {"missing_rate": round(float(series.isna().mean()), 4)}
    values = series.dropna()

    if not is_numeric_dtype(series):
        spec.update(kind="category", allowed=sorted(str(v) for v in values.unique()))
        return spec

    if set(values.unique()) <= {0, 1}:
        spec["kind"] = "binary"
        return spec

    spec["kind"] = "integer" if (values % 1 == 0).all() else "number"
    spec.update(
        min=_number(values.min(), math.floor),
        max=_number(values.max(), math.ceil),
        typical_min=_number(values.quantile(TYPICAL_LOW_QUANTILE)),
        typical_max=_number(values.quantile(TYPICAL_HIGH_QUANTILE)),
    )
    return spec


def build_input_schema(X: pd.DataFrame) -> dict:
    """Schema for every model input column, in column order."""
    schema = {}
    for column in X.columns:
        schema[column] = column_schema(X[column])
        if column in DERIVED_COLUMNS:
            schema[column]["derived"] = True
    return schema


def build_risk_bands(operating_threshold: float, oof_path=None) -> dict:
    """Band cut-offs: Low below `low_upper_bound`, High from the operating threshold.

    With the out-of-fold training predictions (oof_path), each band also records
    its share of applicants and how often they defaulted, so results can say
    "about 1 in 5 applicants in this band defaulted" from the data.
    """
    bands = {
        "low_upper_bound": LOW_RISK_UPPER_BOUND,
        "high_lower_bound": float(operating_threshold),
        "labels": ["Low", "Medium", "High"],
    }
    if oof_path is not None and Path(oof_path).exists():
        from src.models.risk_bands import band_summary

        oof = pd.read_csv(oof_path)
        summary = band_summary(oof["y_true"], oof["xgboost_tuned"], operating_threshold)
        bands["outcomes"] = {
            band: {"share_of_applicants": round(float(row["share_of_applicants"]), 4),
                   "default_rate": round(float(row["default_rate"]), 4)}
            for band, row in summary.iterrows()
        }
        bands["outcomes_source"] = "out-of-fold predictions on the training split"
    return bands


def serving_info(X: pd.DataFrame, operating_threshold: float, oof_path=None) -> dict:
    """The sections added to metadata.json for the backend and the form."""
    if oof_path is None:
        from src.models.select_final import OOF_PATH as oof_path
    return {"input_schema": build_input_schema(X), "risk_bands": build_risk_bands(operating_threshold, oof_path)}


def add_serving_info(metadata_path) -> dict:
    """Add the serving sections to an existing metadata.json without re-fitting the model."""
    from src.data.data_cleaning import clean_and_split

    metadata_path = Path(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    train_df, _ = clean_and_split()  # held-out test rows are discarded here
    X = train_df.drop(columns=[TARGET])
    if list(X.columns) != metadata["input_columns"]:
        raise ValueError("The training columns differ from metadata['input_columns']; rebuild the final model first.")

    metadata.update(serving_info(X, metadata["operating_threshold"]))
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


if __name__ == "__main__":
    from src.models.train_final import FINAL_DIR, METADATA_FILE

    updated = add_serving_info(FINAL_DIR / METADATA_FILE)
    print(json.dumps(updated["risk_bands"], indent=2))
    print(f"input_schema written for {len(updated['input_schema'])} columns")
