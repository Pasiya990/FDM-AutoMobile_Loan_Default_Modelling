"""End-to-end tests for the prediction API, using the real final model.

The key test: /predict on real held-out applications returns exactly the score
the saved pipeline gives when called directly, so the service applies the same
preprocessing as training. The test rows are only scored here, never used to
choose anything.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient

from backend.adapter import to_request
from backend.app import create_app
from backend.messages import DISCLAIMER, SUGGESTED_ACTIONS
from backend.validation import field_specs, validate_request
from src.config import RAW_DATA_PATH, TARGET
from src.models.train_final import FINAL_DIR, MODEL_FILE

pytestmark = pytest.mark.skipif(
    not (FINAL_DIR / MODEL_FILE).exists() or not RAW_DATA_PATH.exists(),
    reason="final model or raw data not on this machine",
)

VALID = {
    "age_years": 35,
    "client_income": 20000,
    "credit_amount": 50000,
    "loan_annuity": 2500,
    "contract_type": "CL",
    "not_employed": False,
    "years_employed": 4,
}
RESPONSE_KEYS = {
    "prediction", "flagged", "risk_score", "risk_band", "threshold",
    "top_factors", "suggested_action", "model_version", "disclaimer",
}


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app()) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def scored_test_rows(client):
    """Real test-set applications that pass validation and have a recorded day and
    hour (otherwise the API fills them from the current time), with the score the
    pipeline gives when called directly."""
    from src.data.data_cleaning import clean_and_split

    store = client.app.state.store
    specs = field_specs(store.metadata["input_schema"])
    _, test_df = clean_and_split()
    X = test_df.drop(columns=[TARGET]).dropna(subset=["Application_Process_Day", "Application_Process_Hour"])

    rows = []
    for _, row in X.sample(400, random_state=0).iterrows():
        request = to_request(row)
        if not validate_request(request, specs):
            rows.append((request, float(store.pipeline.predict_proba(row.to_frame().T.astype(X.dtypes))[0, 1])))
    return rows


def _problems(response):
    return {(d["field"], d["problem"]) for d in response.json()["details"]}


# ---------------------------------------------------------------- health and schema
def test_health_reports_the_loaded_model(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_version"].startswith("xgboost_tuned")


def test_schema_lists_the_form_fields(client):
    body = client.get("/schema").json()

    assert set(body["required"]) <= set(body["fields"])
    assert body["fields"]["age_years"]["min"] == 21 and body["fields"]["age_years"]["max"] == 69
    assert body["fields"]["contract_type"]["allowed"] == ["CL", "RL"]
    assert body["risk_bands"]["labels"] == ["Low", "Medium", "High"]


# ---------------------------------------------------------------- training equals serving
def test_api_score_equals_the_pipeline_called_directly(client, scored_test_rows):
    assert len(scored_test_rows) >= 100
    for request, expected in scored_test_rows[:100]:
        response = client.post("/predict", json=request)
        assert response.status_code == 200, response.json()
        assert response.json()["risk_score"] == pytest.approx(expected, abs=5e-5)


def test_flag_band_and_threshold_agree(client, scored_test_rows):
    for request, _ in scored_test_rows[:50]:
        body = client.post("/predict", json=request).json()
        assert body["flagged"] == (body["risk_score"] >= body["threshold"])
        assert (body["risk_band"] == "High") == body["flagged"]
        assert body["suggested_action"] == SUGGESTED_ACTIONS[body["risk_band"]]


def test_a_high_risk_and_a_low_risk_applicant(client, scored_test_rows):
    highest = max(scored_test_rows, key=lambda pair: pair[1])[0]
    lowest = min(scored_test_rows, key=lambda pair: pair[1])[0]

    high, low = client.post("/predict", json=highest).json(), client.post("/predict", json=lowest).json()

    assert high["risk_band"] == "High" and high["prediction"] == "Likely to default"
    assert low["risk_band"] == "Low" and low["prediction"] == "Not likely to default" and not low["flagged"]


def test_response_has_every_field_of_the_contract(client):
    body = client.post("/predict", json=VALID).json()

    assert set(body) == RESPONSE_KEYS
    assert body["disclaimer"] == DISCLAIMER
    assert 0 <= body["risk_score"] <= 1


# ---------------------------------------------------------------- invalid input
def test_missing_required_field(client):
    response = client.post("/predict", json={k: v for k, v in VALID.items() if k != "client_income"})

    assert response.status_code == 422
    assert response.json()["error"] == "validation_failed"
    assert ("client_income", "is required") in _problems(response)


def test_wrong_type(client):
    response = client.post("/predict", json={**VALID, "credit_amount": "fifty thousand"})

    assert response.status_code == 422
    assert ("credit_amount", "must be a number") in _problems(response)


def test_negative_income(client):
    response = client.post("/predict", json={**VALID, "client_income": -100})

    assert response.status_code == 422
    assert any(field == "client_income" for field, _ in _problems(response))


@pytest.mark.parametrize("age, accepted", [(20.9, False), (21, True), (69, True), (70, False)])
def test_age_boundaries(client, age, accepted):
    response = client.post("/predict", json={**VALID, "age_years": age})

    assert (response.status_code == 200) == accepted


def test_unknown_category_and_unknown_field(client):
    response = client.post("/predict", json={**VALID, "education": "PhD", "nickname": "Sam"})

    assert response.status_code == 422
    assert ("education", "is not one of the allowed values") in _problems(response)
    assert ("nickname", "is not a known field") in _problems(response)


def test_optional_bureau_scores_may_be_blank(client):
    response = client.post("/predict", json={**VALID, "score_source_1": None, "score_source_2": None})

    assert response.status_code == 200


def test_body_that_is_not_json_or_not_an_object(client):
    not_json = client.post("/predict", content="age=35", headers={"content-type": "application/json"})
    a_list = client.post("/predict", json=[VALID])

    assert not_json.status_code == 422 and a_list.status_code == 422


# ---------------------------------------------------------------- model not loaded
def test_every_endpoint_answers_503_without_a_model(tmp_path):
    with TestClient(create_app(tmp_path)) as no_model:
        for response in (no_model.get("/health"), no_model.get("/schema"), no_model.post("/predict", json=VALID)):
            assert response.status_code == 503
            assert response.json()["error"] == "model_not_available"


# ---------------------------------------------------------------- demo examples
def test_demo_examples_get_their_stated_band(client):
    import json

    from backend.examples import EXAMPLES_PATH

    examples = json.loads(EXAMPLES_PATH.read_text(encoding="utf-8"))
    assert [e["expected_band"] for e in examples] == ["Low", "Medium", "High"]
    for example in examples:
        body = client.post("/predict", json=example["application"]).json()
        assert body["risk_band"] == example["expected_band"], "rebuild with: python -m backend.examples"
        assert body["risk_score"] == pytest.approx(example["risk_score"], abs=5e-5)
        assert 1 <= len(body["top_factors"]) <= 4
        assert {"factor", "effect", "impact"} <= set(body["top_factors"][0])
