"""Checks a loan application against the input schema saved with the model.

field_specs() turns metadata["input_schema"] (written in the dataset's columns)
into the form's fields (age in years, yes/no answers, ...). The same specs drive
validation here and the GET /schema endpoint, so the form and the API always
agree on what is allowed.

validate_request() returns a list of problems, one per field, in the plan's 422
format: {"field": ..., "problem": ..., "received": ...}. An empty list means the
application can be scored. Only the hard limits are enforced; the typical range
is passed to the form for hints, so an unusual but real value is never refused.
"""

import math

from backend.adapter import DAYS_PER_YEAR, DIRECT_FIELDS, FLAG_FIELDS, TAG_FIELDS
from src.config import SENTINEL_COL

REQUIRED_FIELDS = ("age_years", "client_income", "credit_amount", "loan_annuity", "contract_type", "not_employed")


def _in_years(column_spec, lowest=None):
    """Limits of a days column, in whole years (widened outward, so 21 and 69 are accepted)."""
    def years(days, rounding):
        return rounding(days / DAYS_PER_YEAR)

    return {
        "type": "number",
        "min": lowest if lowest is not None else years(column_spec["min"], math.floor),
        "max": years(column_spec["max"], math.ceil),
        "typical_min": round(column_spec["typical_min"] / DAYS_PER_YEAR, 1),
        "typical_max": round(column_spec["typical_max"] / DAYS_PER_YEAR, 1),
    }


def field_specs(input_schema: dict) -> dict:
    """Every form field with its type, whether it is required and what it allows."""
    specs = {}
    for field, column in DIRECT_FIELDS.items():
        column_spec = input_schema[column]
        if column_spec["kind"] == "category":
            spec = {"type": "category", "allowed": column_spec["allowed"]}
        else:
            spec = {"type": column_spec["kind"]}
            spec.update({k: column_spec[k] for k in ("min", "max", "typical_min", "typical_max") if k in column_spec})
        specs[field] = spec

    for field in list(FLAG_FIELDS) + list(TAG_FIELDS) + ["not_employed"]:
        specs[field] = {"type": "boolean"}

    specs["age_years"] = _in_years(input_schema["Age_Days"])
    # Someone who has just started a job has 0 years of employment
    specs["years_employed"] = _in_years(input_schema[SENTINEL_COL], lowest=0)

    for field, spec in specs.items():
        spec["required"] = field in REQUIRED_FIELDS
    return specs


def _type_problem(value, spec):
    """What is wrong with the value's type, or None."""
    kind = spec["type"]
    if kind == "boolean":
        return None if isinstance(value, bool) else "must be true or false"
    if kind == "category":
        if not isinstance(value, str):
            return "must be text"
        return None if value in spec["allowed"] else "is not one of the allowed values"
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return "must be a number"
    if kind == "integer" and value != int(value):
        return "must be a whole number"
    if value < spec["min"] or value > spec["max"]:
        return f"must be between {spec['min']} and {spec['max']}"
    return None


def validate_request(request: dict, specs: dict) -> list:
    """Problems with one application; an empty list means it can be scored."""
    if not isinstance(request, dict):
        return [{"field": None, "problem": "the application must be a JSON object", "received": request}]

    problems = []
    for field, value in request.items():
        if field not in specs:
            problems.append({"field": field, "problem": "is not a known field", "received": value})
        elif value is not None:
            problem = _type_problem(value, specs[field])
            if problem:
                problems.append({"field": field, "problem": problem, "received": value})

    for field in REQUIRED_FIELDS:
        if request.get(field) is None:
            problems.append({"field": field, "problem": "is required", "received": None})

    if request.get("not_employed") is False and request.get("years_employed") is None:
        problems.append(
            {"field": "years_employed", "problem": "is required when the applicant is employed", "received": None}
        )
    if request.get("car_owned") is False and request.get("car_age") is not None:
        problems.append(
            {"field": "car_age", "problem": "must be left blank when the applicant has no car", "received": request["car_age"]}
        )
    return problems
