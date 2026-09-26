"""Shared constants for the loan default pipeline. Single source of truth
so notebooks, tests and the backend never drift apart on column names or
thresholds.
"""

import os

RANDOM_STATE = 42

TARGET = "Default"
ID_COL = "ID"

RAW_DATA_PATH = os.path.join("data", "raw", "Train_Dataset.csv")
PROCESSED_DATA_DIR = os.path.join("data", "processed")

# Columns found to be numeric but stored as text due to stray characters
# ('$', 'x', '&', '#VALUE!', '@', '#') - see 01_data_understanding.ipynb
NUMERIC_TEXT_COLS = [
    "Client_Income",
    "Credit_Amount",
    "Loan_Annuity",
    "Population_Region_Relative",
    "Age_Days",
    "Employed_Days",
    "Registration_Days",
    "ID_Days",
    "Score_Source_3",
]

# Employed_Days sentinel: 365243 marks retired/unemployed applicants (F4)
SENTINEL_COL = "Employed_Days"
SENTINEL_VALUE = 365243
SENTINEL_FLAG_COL = "Is_Retired_Or_Unemployed"

# Disguised missing values (placeholder strings), by column
PLACEHOLDER_TOKENS = {
    "Client_Gender": ["XNA"],
    "Accompany_Client": ["##"],
}
# Type_Organization's "XNA" is NOT a placeholder to remove - it's a
# legitimate "no employer" category, tied to the sentinel population.

# Corrupted numeric values found via EDA (exact value 100, should be <=1)
CORRUPTED_ABOVE_ONE_COLS = ["Score_Source_2", "Population_Region_Relative"]

# Own_House_Age is actually car age (filled only when Car_Owned == 1, not
# related to House_Own) - see team implementation plan finding F5
CAR_AGE_RENAME = {"Own_House_Age": "car_age"}
CAR_AGE_COL = "car_age"
HAS_CAR_AGE_COL = "has_car_age"
CAR_OWNED_COL = "Car_Owned"

DROP_COLS = [ID_COL]
# Mobile_Tag is constant (1 for all but one row) - not hardcoded here,
# it is expected to be caught and dropped by importance-based feature
# selection instead.

SCORE_COLS = ["Score_Source_1", "Score_Source_2", "Score_Source_3"]

# Missing-value handling
MISSING_FLAG_THRESHOLD = 0.05  # add a "_was_missing" flag above this rate

# Outlier treatment
LOG_TRANSFORM_COLS = ["Client_Income"]

# Rare-category grouping (fit on train only)
RARE_CATEGORY_THRESHOLD = 0.01  # categories below 1% of train rows -> "Other"
RARE_CATEGORY_COLS = [
    "Type_Organization",
    "Client_Education",
    "Client_Income_Type",
    "Client_Occupation",
]

# Feature selection
LOW_IMPORTANCE_THRESHOLD = 0.001
REDUNDANT_COLS_TO_DROP = ["Age_Days", "Loan_Annuity"]

TRAIN_TEST_SPLIT_RATIO = 0.2
