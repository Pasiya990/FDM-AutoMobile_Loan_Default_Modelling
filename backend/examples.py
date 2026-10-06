"""Example applicants for the demo, one per risk band, taken from real held-out
applications (only scored here, never used to choose anything).

Each example is a complete, valid application with few blank fields, so the
demo shows realistic inputs. Age and years employed are rounded to one decimal
and the band is taken from the score of the rounded application, so the saved
band is exactly what /predict returns. The application day and hour are kept,
so the score does not depend on when the demo is run.

Usage (from the project root): python -m backend.examples
"""

import json
from datetime import datetime
from pathlib import Path

from backend.adapter import build_model_row, to_request
from backend.model_store import load_model_store
from backend.validation import field_specs, validate_request
from src.config import TARGET
from src.models.risk_bands import risk_band

EXAMPLES_PATH = Path(__file__).resolve().parent / "examples.json"
CANDIDATES = 2000


def _rounded(request):
    for field in ("age_years", "years_employed"):
        if request.get(field) is not None:
            request[field] = round(request[field], 1)
    return request


def build_examples(store, X):
    """One example per band: the most typical score of each band among complete applications."""
    specs = field_specs(store.metadata["input_schema"])
    threshold = store.operating_threshold
    low_upper = store.metadata["risk_bands"]["low_upper_bound"]
    targets = {"Low": low_upper / 2, "Medium": (low_upper + threshold) / 2, "High": (threshold + 1) / 2}

    complete = X.dropna(subset=["Application_Process_Day", "Application_Process_Hour"])
    complete = complete.loc[complete.isna().sum(axis=1).sort_values(kind="stable").index[:CANDIDATES]]

    best = {}
    for _, row in complete.iterrows():
        request = _rounded(to_request(row))
        if validate_request(request, specs):
            continue
        model_row = build_model_row(request, store.metadata["input_columns"], submitted_at=datetime(2026, 1, 1))
        score = float(store.pipeline.predict_proba(model_row)[0, 1])
        band = risk_band(score, threshold, low_upper)
        distance = abs(score - targets[band])
        if band not in best or distance < best[band][0]:
            best[band] = (distance, score, request)

    return [
        {"label": f"{band}-risk applicant", "expected_band": band, "risk_score": round(best[band][1], 4),
         "application": {k: v for k, v in best[band][2].items() if v is not None}}
        for band in ("Low", "Medium", "High")
    ]


if __name__ == "__main__":
    from src.data.data_cleaning import clean_and_split

    _, test_df = clean_and_split()
    examples = build_examples(load_model_store(), test_df.drop(columns=[TARGET]))
    EXAMPLES_PATH.write_text(json.dumps(examples, indent=2) + "\n", encoding="utf-8")
    for example in examples:
        print(example["label"], example["risk_score"])
