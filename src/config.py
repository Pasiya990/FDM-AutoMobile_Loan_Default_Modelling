"""Shared constants for the loan default pipeline. Single source of truth
so src/data, src/preprocessing, notebooks and tests never drift apart on
column names, thresholds or the random seed.
"""

from pathlib import Path

RANDOM_STATE = 42

TARGET = "Default"
ID_COL = "ID"

RAW_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "Train_Dataset.csv"

# Columns found to be numeric but stored as text due to stray characters
# ('$', 'x', '&', '#VALUE!', '@', '#') - see notebooks/01_data_understanding.ipynb
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

# Employed_Days sentinel: 365243 marks retired/unemployed applicants
SENTINEL_COL = "Employed_Days"
SENTINEL_VALUE = 365243
SENTINEL_FLAG_COL = "Is_Retired_Or_Unemployed"

# Corrupted numeric values found via EDA (exact value 100, should be <=1)
CORRUPTED_ABOVE_ONE_COLS = ["Score_Source_2", "Population_Region_Relative"]

# Own_House_Age is actually car age (filled only when Car_Owned == 1, not
# related to House_Own) - see notebooks/03_data_preprocessing.ipynb
CAR_AGE_COL = "car_age"
HAS_CAR_AGE_COL = "has_car_age"
CAR_OWNED_COL = "Car_Owned"

SCORE_COLS = ["Score_Source_1", "Score_Source_2", "Score_Source_3"]

# Numeric -> median (fit on train), with a "_was_missing" flag when the
# missing rate exceeds this threshold
MISSING_FLAG_THRESHOLD = 0.05

# Denominators in the ratio features: a zero or negative value would produce
# infinity, so it is treated as missing and imputed instead
POSITIVE_ONLY_COLS = ["Client_Income", "Loan_Annuity"]

# Left un-encoded until feature engineering's rare-category grouping runs
COLS_RESERVED_FOR_FEATURE_ENGINEERING = [
    "Type_Organization",
    "Client_Education",
    "Client_Income_Type",
    "Client_Occupation",
]

# Categories below this fraction of training rows -> "Other"
RARE_CATEGORY_MIN_FREQ = 0.01

# RandomForest importance below this threshold -> feature dropped
LOW_IMPORTANCE_THRESHOLD = 0.001

# Risk bands on the model's risk score. "High" starts at the operating
# threshold saved with the model (metadata.json), so High means "flagged".
# "Low" is every score below this value; the rest is "Medium". Chosen from the
# out-of-fold training predictions, see src/models/risk_bands.py.
LOW_RISK_UPPER_BOUND = 0.25
