"""Tests for request validation against the input schema saved with the model."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.adapter import to_request
from backend.validation import REQUIRED_FIELDS, field_specs, validate_request
from src.config import RAW_DATA_PATH, TARGET

# A cut-down schema in the layout of metadata["input_schema"]
SCHEMA = {
    "Client_Income": {"kind": "number", "min": 2565, "max": 1800009, "typical_min": 4500, "typical_max": 47250},
    "Credit_Amount": {"kind": "number", "min": 4500, "max": 405000, "typical_min": 7650, "typical_max": 186498},
    "Loan_Annuity": {"kind": "number", "min": 217.35, "max": 22500, "typical_min": 624, "typical_max": 6976},
    "Child_Count": {"kind": "integer", "min": 0, "max": 19, "typical_min": 0, "typical_max": 3},
    "Loan_Contract_Type": {"kind": "category", "allowed": ["CL", "RL"]},
    "Client_Gender": {"kind": "category", "allowed": ["Female", "Male"]},
    "car_age": {"kind": "integer", "min": 0, "max": 69, "typical_min": 0, "typical_max": 64},
    "Age_Days": {"kind": "integer", "min": 7676, "max": 25201, "typical_min": 8275, "typical_max": 24427},
    "Employed_Days": {"kind": "integer", "min": 2, "max": 17546, "typical_min": 111, "typical_max": 11305},
}
VALID = {
    "age_years": 35,
    "client_income": 20000,
    "credit_amount": 50000,
    "loan_annuity": 2500,
    "contract_type": "CL",
    "not_employed": False,
    "years_employed": 4,
}


@pytest.fixture
def specs(monkeypatch):
    # Only the fields of the cut-down schema; the real schema covers them all
    import backend.validation as validation

    monkeypatch.setattr(
        validation,
        "DIRECT_FIELDS",
        {
            "client_income": "Client_Income",
            "credit_amount": "Credit_Amount",
            "loan_annuity": "Loan_Annuity",
            "child_count": "Child_Count",
            "contract_type": "Loan_Contract_Type",
            "gender": "Client_Gender",
            "car_age": "car_age",
        },
    )
    return validation.field_specs(SCHEMA)


def _problems_for(specs, **changes):
    request = {**VALID, **changes}
    return {(p["field"], p["problem"]) for p in validate_request({k: v for k, v in request.items() if v != "DROP"}, specs)}


def test_a_complete_valid_application_passes(specs):
    assert validate_request(VALID, specs) == []


def test_age_limits_are_whole_years_21_to_69(specs):
    assert specs["age_years"]["min"] == 21 and specs["age_years"]["max"] == 69
    assert _problems_for(specs, age_years=21) == set()
    assert ("age_years", "must be between 21 and 69") in _problems_for(specs, age_years=17)


@pytest.mark.parametrize("field", REQUIRED_FIELDS)
def test_each_required_field_is_reported_when_missing(specs, field):
    assert (field, "is required") in _problems_for(specs, **{field: "DROP"})


def test_negative_income_and_wrong_types_are_rejected(specs):
    problems = _problems_for(specs, client_income=-5, credit_amount="lots", not_employed="no", child_count=1.5)

    assert ("client_income", "must be between 2565 and 1800009") in problems
    assert ("credit_amount", "must be a number") in problems
    assert ("not_employed", "must be true or false") in problems
    assert ("child_count", "must be a whole number") in problems


def test_unknown_category_and_unknown_field_are_rejected(specs):
    problems = _problems_for(specs, contract_type="Mortgage", favourite_colour="blue")

    assert ("contract_type", "is not one of the allowed values") in problems
    assert ("favourite_colour", "is not a known field") in problems


def test_employment_and_car_rules(specs):
    assert ("years_employed", "is required when the applicant is employed") in _problems_for(specs, years_employed="DROP")
    assert _problems_for(specs, not_employed=True, years_employed="DROP") == set()
    assert _problems_for(specs, years_employed=0) == set()  # just started a job
    assert ("car_age", "must be left blank when the applicant has no car") in _problems_for(specs, car_owned=False, car_age=5)


def test_optional_fields_may_be_blank_or_left_out(specs):
    assert _problems_for(specs, gender=None, car_age=None) == set()
    assert specs["gender"]["required"] is False


def test_a_hint_range_is_not_a_limit(specs):
    assert specs["client_income"]["typical_max"] == 47250
    assert _problems_for(specs, client_income=100000) == set()


@pytest.mark.skipif(not RAW_DATA_PATH.exists(), reason="raw data not on this machine")
def test_real_training_applications_pass_the_real_schema():
    from backend.model_store import load_model_store
    from src.data.data_cleaning import clean_and_split

    store = load_model_store()
    real_specs = field_specs(store.metadata["input_schema"])
    train_df, _ = clean_and_split()
    complete = train_df.drop(columns=[TARGET]).dropna(subset=["Client_Income", "Credit_Amount", "Loan_Annuity", "Age_Days", "Loan_Contract_Type"])

    for _, row in complete.sample(300, random_state=0).iterrows():
        request = to_request(row)
        if not request["not_employed"] and request["years_employed"] is None:
            continue  # employment left blank in the data; the form requires it
        assert validate_request(request, real_specs) == [], request
