"""Tests for the request adapter: form fields -> the model's one-row table."""

import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.adapter import application_day, build_model_row, to_request
from src.config import RAW_DATA_PATH, TARGET

COLUMNS = [
    "Client_Income", "Car_Owned", "Age_Days", "Employed_Days", "car_age", "Mobile_Tag",
    "Client_Permanent_Match_Tag", "Client_Gender", "Application_Process_Day", "Application_Process_Hour",
    "Is_Retired_Or_Unemployed", "has_car_age",
]
MONDAY_10AM = datetime(2026, 10, 5, 10, 30)


def test_age_and_employment_are_converted_to_days():
    row = build_model_row({"age_years": 40, "years_employed": 2, "not_employed": False}, COLUMNS, MONDAY_10AM)

    assert row.loc[0, "Age_Days"] == 14610  # 40 x 365.25
    assert row.loc[0, "Employed_Days"] == 730  # 2 x 365.25, rounded
    assert row.loc[0, "Is_Retired_Or_Unemployed"] == 0


def test_not_employed_gives_a_blank_and_the_flag_like_the_cleaning_step():
    row = build_model_row({"not_employed": True, "years_employed": 5}, COLUMNS, MONDAY_10AM)

    assert np.isnan(row.loc[0, "Employed_Days"])
    assert row.loc[0, "Is_Retired_Or_Unemployed"] == 1


def test_has_car_age_follows_car_owned_like_the_cleaning_step():
    owner = build_model_row({"car_owned": True, "car_age": 7}, COLUMNS, MONDAY_10AM)
    non_owner = build_model_row({"car_owned": False}, COLUMNS, MONDAY_10AM)
    unknown = build_model_row({"car_age": 7}, COLUMNS, MONDAY_10AM)

    assert owner.loc[0, "car_age"] == 7 and owner.loc[0, "has_car_age"] == 1
    assert non_owner.loc[0, "has_car_age"] == 0
    assert unknown.loc[0, "car_age"] == 7 and unknown.loc[0, "has_car_age"] == 0


def test_left_out_fields_are_blank_or_the_usual_answer():
    row = build_model_row({}, COLUMNS, MONDAY_10AM)

    assert np.isnan(row.loc[0, "Client_Income"]) and pd.isna(row.loc[0, "Client_Gender"])
    assert row.loc[0, "Mobile_Tag"] == 1
    assert row.loc[0, "Client_Permanent_Match_Tag"] == "Yes"


def test_application_day_and_hour_come_from_the_submission_time():
    row = build_model_row({}, COLUMNS, MONDAY_10AM)

    assert row.loc[0, "Application_Process_Day"] == 1  # Sunday = 0, so Monday = 1
    assert row.loc[0, "Application_Process_Hour"] == 10
    assert application_day(datetime(2026, 10, 4)) == 0  # a Sunday


def test_columns_follow_the_given_order():
    assert list(build_model_row({}, COLUMNS, MONDAY_10AM).columns) == COLUMNS


@pytest.mark.skipif(not RAW_DATA_PATH.exists(), reason="raw data not on this machine")
def test_training_rows_survive_a_round_trip_through_the_form_fields():
    from src.data.data_cleaning import clean_and_split

    train_df, _ = clean_and_split()
    X = train_df.drop(columns=[TARGET])
    sample = X.sample(20, random_state=0)

    for _, original in sample.iterrows():
        rebuilt = build_model_row(to_request(original), list(X.columns), MONDAY_10AM).iloc[0]
        for column in X.columns:
            if pd.isna(original[column]):
                assert pd.isna(rebuilt[column]), column
            else:
                assert rebuilt[column] == original[column], column
    assert (build_model_row(to_request(sample.iloc[0]), list(X.columns)).dtypes == X.dtypes).all()
