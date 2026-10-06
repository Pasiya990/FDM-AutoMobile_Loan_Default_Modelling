"""API endpoints: GET /health, GET /schema, POST /predict.

/predict validates the application, converts it to the model's columns with the
adapter, and scores it with the saved pipeline, which applies exactly the
preprocessing used in training. Errors follow the plan's format: 422 with
per-field details for invalid input, 503 when the model is not loaded.
"""

from datetime import datetime

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from backend.adapter import build_model_row
from backend.messages import DISCLAIMER, PREDICTION_LABELS, SUGGESTED_ACTIONS
from backend.validation import REQUIRED_FIELDS, field_specs, validate_request
from src.models.risk_bands import risk_band

router = APIRouter()


def model_unavailable(request: Request) -> JSONResponse:
    message = getattr(request.app.state, "load_error", None) or "Model artefact could not be loaded."
    return JSONResponse(status_code=503, content={"error": "model_not_available", "message": message})


def validation_failed(details: list) -> JSONResponse:
    return JSONResponse(status_code=422, content={"error": "validation_failed", "details": details})


@router.get("/health")
def health(request: Request):
    store = request.app.state.store
    if store is None:
        return model_unavailable(request)
    return {
        "status": "ok",
        "model_version": store.model_version,
        "version_warnings": store.version_warnings,
    }


@router.get("/schema")
def schema(request: Request):
    """The form's fields with their types, limits and allowed values."""
    store = request.app.state.store
    if store is None:
        return model_unavailable(request)
    return {
        "fields": field_specs(store.metadata["input_schema"]),
        "required": list(REQUIRED_FIELDS),
        "risk_bands": store.metadata["risk_bands"],
        "model_version": store.model_version,
    }


@router.post("/predict")
async def predict(request: Request):
    store = request.app.state.store
    if store is None:
        return model_unavailable(request)

    try:
        application = await request.json()
    except ValueError:
        return validation_failed([{"field": None, "problem": "the body must be valid JSON", "received": None}])

    specs = field_specs(store.metadata["input_schema"])
    problems = validate_request(application, specs)
    if problems:
        return validation_failed(problems)

    row = build_model_row(application, store.metadata["input_columns"], submitted_at=datetime.now())
    score = float(store.pipeline.predict_proba(row)[0, 1])
    threshold = store.operating_threshold
    band = risk_band(score, threshold, store.metadata["risk_bands"]["low_upper_bound"])

    return {
        "prediction": PREDICTION_LABELS[score >= threshold],
        "flagged": score >= threshold,
        "risk_score": round(score, 4),
        "risk_band": band,
        "threshold": round(threshold, 4),
        "top_factors": [],
        "suggested_action": SUGGESTED_ACTIONS[band],
        "model_version": store.model_version,
        "disclaimer": DISCLAIMER,
    }
