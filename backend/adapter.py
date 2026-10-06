"""Turns a loan application, as entered in the form, into the one-row table the
model was trained on (the columns after src/data/data_cleaning.py).

The form uses friendly fields (age in years, "not employed", yes/no answers);
the model uses the dataset's columns (age in days, the 365243 sentinel, 0/1 and
"Yes"/"No" tags). Any field left out becomes a blank, which the pipeline imputes
exactly as in training. Checking that values are allowed is not done here, see
backend/validation.py.

to_request() does the reverse, for tests and example applicants: a training row
converted to form fields and back must give the same row.
"""

from datetime import datetime

import numpy as np
import pandas as pd

from src.config import (
    CAR_AGE_COL,
    CAR_OWNED_COL,
    HAS_CAR_AGE_COL,
    SENTINEL_COL,
    SENTINEL_FLAG_COL,
)

DAYS_PER_YEAR = 365.25  # matches the data: ages run from 7,676 days (21 years) to 25,201 days (69 years)

# Form field -> dataset column, copied as they are (numbers stay numbers, text stays text)
DIRECT_FIELDS = {
    "client_income": "Client_Income",
    "credit_amount": "Credit_Amount",
    "loan_annuity": "Loan_Annuity",
    "child_count": "Child_Count",
    "family_members": "Client_Family_Members",
    "accompany_client": "Accompany_Client",
    "income_type": "Client_Income_Type",
    "education": "Client_Education",
    "marital_status": "Client_Marital_Status",
    "gender": "Client_Gender",
    "contract_type": "Loan_Contract_Type",
    "housing_type": "Client_Housing_Type",
    "occupation": "Client_Occupation",
    "organization_type": "Type_Organization",
    "population_region_relative": "Population_Region_Relative",
    "city_rating": "Cleint_City_Rating",
    "days_since_registration_change": "Registration_Days",
    "days_since_id_change": "ID_Days",
    "days_since_phone_change": "Phone_Change",
    "car_age": CAR_AGE_COL,
    "score_source_1": "Score_Source_1",
    "score_source_2": "Score_Source_2",
    "score_source_3": "Score_Source_3",
    "social_circle_default": "Social_Circle_Default",
    "credit_bureau_enquiries": "Credit_Bureau",
    "application_day": "Application_Process_Day",
    "application_hour": "Application_Process_Hour",
}

# Yes/no answers stored as 0/1 in the dataset; these may be left blank
FLAG_FIELDS = {
    "car_owned": CAR_OWNED_COL,
    "bike_owned": "Bike_Owned",
    "active_loan": "Active_Loan",
    "house_owned": "House_Own",
}

# Yes/no answers that are never blank in the dataset: 0/1 columns and
# "Yes"/"No" columns. When left out, the most common training answer is used.
TAG_FIELDS = {
    "mobile_phone": ("Mobile_Tag", (1, 0), True),
    "homephone": ("Homephone_Tag", (1, 0), False),
    "workphone": ("Workphone_Working", (1, 0), False),
    "permanent_address_match": ("Client_Permanent_Match_Tag", ("Yes", "No"), True),
    "work_contact_match": ("Client_Contact_Work_Tag", ("Yes", "No"), True),
}


# Column types as in the cleaned training data: text, never-blank integers, the rest float
TEXT_COLUMNS = {
    "Accompany_Client", "Client_Income_Type", "Client_Education", "Client_Marital_Status", "Client_Gender",
    "Loan_Contract_Type", "Client_Housing_Type", "Client_Occupation", "Type_Organization",
    "Client_Permanent_Match_Tag", "Client_Contact_Work_Tag",
}
INTEGER_COLUMNS = {"Mobile_Tag", "Homephone_Tag", "Workphone_Working", SENTINEL_FLAG_COL, HAS_CAR_AGE_COL}


def _blank(value):
    return value is None or (isinstance(value, float) and np.isnan(value))


def application_day(submitted_at: datetime) -> int:
    """Dataset weekday code. 0 is taken to be Sunday: it is the quietest day in the
    data (about 4,400 applications against about 14,000 on weekdays), with 6 next."""
    return (submitted_at.weekday() + 1) % 7


def build_model_row(request: dict, input_columns, submitted_at: datetime | None = None) -> pd.DataFrame:
    """One application as a one-row DataFrame with the model's columns, in `input_columns` order."""
    submitted_at = submitted_at or datetime.now()
    row = {column: np.nan for column in input_columns}

    for field, column in DIRECT_FIELDS.items():
        value = request.get(field)
        if not _blank(value):
            row[column] = value

    for field, column in FLAG_FIELDS.items():
        value = request.get(field)
        if not _blank(value):
            row[column] = int(bool(value))

    for field, (column, (yes, no), default) in TAG_FIELDS.items():
        value = request.get(field)
        row[column] = yes if (default if _blank(value) else value) else no

    if not _blank(request.get("age_years")):
        row["Age_Days"] = round(request["age_years"] * DAYS_PER_YEAR)

    # Employment, as in apply_sentinel_and_placeholder_fixes(): the 365243 sentinel
    # becomes a blank plus the retired/unemployed flag
    not_employed = bool(request.get("not_employed"))
    row[SENTINEL_FLAG_COL] = int(not_employed)
    if not not_employed and not _blank(request.get("years_employed")):
        row[SENTINEL_COL] = round(request["years_employed"] * DAYS_PER_YEAR)

    # As in the cleaning step: the flag follows Car_Owned and car_age is kept as
    # given (a car age for a non-owner is rejected in validation, not here)
    row[HAS_CAR_AGE_COL] = int(row[CAR_OWNED_COL] == 1)

    if _blank(request.get("application_day")):
        row["Application_Process_Day"] = application_day(submitted_at)
    if _blank(request.get("application_hour")):
        row["Application_Process_Hour"] = submitted_at.hour

    frame = pd.DataFrame([row], columns=list(input_columns))
    for column in frame.columns:
        if column in TEXT_COLUMNS:
            frame[column] = frame[column].astype("str").where(frame[column].notna(), np.nan)
        elif column in INTEGER_COLUMNS:
            frame[column] = frame[column].astype("int64")
        else:
            frame[column] = frame[column].astype("float64")
    return frame


def to_request(model_row: pd.Series) -> dict:
    """Form fields for one row of the cleaned data (the reverse of build_model_row)."""
    def value(column):
        v = model_row[column]
        return None if pd.isna(v) else (v.item() if hasattr(v, "item") else v)

    request = {field: value(column) for field, column in DIRECT_FIELDS.items()}
    for field, column in FLAG_FIELDS.items():
        request[field] = None if value(column) is None else bool(value(column))
    for field, (column, (yes, _), _) in TAG_FIELDS.items():
        request[field] = value(column) == yes

    request["age_years"] = None if value("Age_Days") is None else value("Age_Days") / DAYS_PER_YEAR
    request["not_employed"] = value(SENTINEL_FLAG_COL) == 1
    days_employed = value(SENTINEL_COL)
    request["years_employed"] = None if days_employed is None else days_employed / DAYS_PER_YEAR
    return request
