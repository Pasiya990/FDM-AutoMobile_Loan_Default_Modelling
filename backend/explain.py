"""The main factors behind one risk score.

XGBoost computes exact TreeSHAP contributions itself (pred_contribs=True): for
one applicant, each model feature gets a contribution in log-odds, and the
contributions plus a base value add up to the model's raw score. So no extra
library is needed.

The model sees 50-odd transformed features (one-hot columns, ratios, missing
flags), which mean little to a loan officer. Contributions are therefore summed
into readable factors - all Type_Organization_* columns into "Employer type",
the scores and their summaries into "External credit scores", and so on - and
the largest few are returned with the direction of their effect.

The contributions explain the model's score; they are not causes of default.
"""

import numpy as np
import pandas as pd
import xgboost

TOP_FACTORS = 4

# Model column (before one-hot encoding or added suffixes) -> readable factor
FACTORS = {
    "Score_Source_1": "External credit scores",
    "Score_Source_2": "External credit scores",
    "Score_Source_3": "External credit scores",
    "score_mean": "External credit scores",
    "score_min": "External credit scores",
    "scores_available": "External credit scores",
    "Credit_Amount": "Loan size compared with income",
    "Loan_Annuity": "Loan size compared with income",
    "credit_to_income": "Loan size compared with income",
    "annuity_to_income": "Loan size compared with income",
    "Credit_Loan_Ratio": "Loan size compared with income",
    "Client_Income": "Income",
    "income_per_member": "Income",
    "Client_Income_Type": "Income type",
    "Age_Years": "Age",
    "Age_Days": "Age",
    "Employed_Days": "Time in current job",
    "Is_Retired_Or_Unemployed": "Employment status",
    "Client_Occupation": "Occupation",
    "Type_Organization": "Employer type",
    "Client_Education": "Education",
    "Client_Gender": "Gender",
    "Client_Marital_Status": "Marital status",
    "Child_Count": "Family size",
    "Client_Family_Members": "Family size",
    "has_children": "Family size",
    "Car_Owned": "Car ownership and age",
    "car_age": "Car ownership and age",
    "has_car_age": "Car ownership and age",
    "Bike_Owned": "Other assets",
    "House_Own": "Other assets",
    "Client_Housing_Type": "Housing type",
    "Population_Region_Relative": "Region",
    "Cleint_City_Rating": "Region",
    "Registration_Days": "Recent registration, ID or phone changes",
    "ID_Days": "Recent registration, ID or phone changes",
    "Phone_Change": "Recent registration, ID or phone changes",
    "Mobile_Tag": "Contact details given",
    "Homephone_Tag": "Contact details given",
    "Workphone_Working": "Contact details given",
    "contact_count": "Contact details given",
    "Client_Permanent_Match_Tag": "Address matches",
    "Client_Contact_Work_Tag": "Address matches",
    "Credit_Bureau": "Credit bureau enquiries",
    "Social_Circle_Default": "Defaults in social circle",
    "Active_Loan": "Other active loans",
    "Loan_Contract_Type": "Contract type",
    "Accompany_Client": "Came to apply with",
    "Application_Process_Day": "Application day and hour",
    "Application_Process_Hour": "Application day and hour",
}
ONE_HOT_PREFIXES = sorted((c for c in FACTORS if c[0].isupper()), key=len, reverse=True)


def factor_of(feature: str) -> str:
    """Readable factor for one model feature, e.g. 'Type_Organization_XNA' -> 'Employer type'."""
    base = feature.removesuffix("_was_missing")
    if base in FACTORS:
        return FACTORS[base]
    for prefix in ONE_HOT_PREFIXES:
        if base.startswith(prefix + "_"):
            return FACTORS[prefix]
    return "Other details"


def contributions(pipeline, model_row: pd.DataFrame) -> pd.Series:
    """Per-feature contributions (log-odds) for one applicant, plus the base value as 'base'."""
    features = pipeline[:-1].transform(model_row)
    booster = pipeline[-1].get_booster()
    values = booster.predict(xgboost.DMatrix(features), pred_contribs=True)[0]
    return pd.Series(values, index=list(features.columns) + ["base"])


def top_factors(pipeline, model_row: pd.DataFrame, n=TOP_FACTORS) -> list:
    """The `n` readable factors that moved this applicant's score most, largest first."""
    per_feature = contributions(pipeline, model_row).drop("base")
    per_factor = per_feature.groupby(per_feature.index.map(factor_of)).sum()
    strongest = per_factor.reindex(per_factor.abs().sort_values(ascending=False).index).head(n)
    return [
        {
            "factor": name,
            "effect": "raises risk" if impact > 0 else "lowers risk",
            "impact": round(float(impact), 3),
        }
        for name, impact in strongest.items()
        if not np.isclose(impact, 0)
    ]
