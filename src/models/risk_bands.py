"""Risk bands for the final model's risk score.

    Low     score <  LOW_RISK_UPPER_BOUND
    Medium  LOW_RISK_UPPER_BOUND <= score < operating threshold
    High    score >= operating threshold (the same applicants the model flags)

The operating threshold comes from models/final/metadata.json, so the bands and
the flag can never disagree. The Low cut-off was chosen from the out-of-fold
training predictions (experiments/results/final_oof_predictions.csv), not the
test set. The score comes from a class-weighted model, so it is a risk score,
not a probability: describe a band by how often its applicants defaulted.

Usage (from the project root): python -m src.models.risk_bands
"""

import pandas as pd

from src.config import LOW_RISK_UPPER_BOUND

BANDS = ("Low", "Medium", "High")


def risk_band(score, operating_threshold, low_upper_bound=LOW_RISK_UPPER_BOUND):
    """Band name for one risk score."""
    if not 0 <= low_upper_bound < operating_threshold <= 1:
        raise ValueError("Need 0 <= low_upper_bound < operating_threshold <= 1")
    if score >= operating_threshold:
        return "High"
    if score >= low_upper_bound:
        return "Medium"
    return "Low"


def band_summary(y_true, scores, operating_threshold, low_upper_bound=LOW_RISK_UPPER_BOUND):
    """Share of applicants, default rate and share of all defaulters in each band."""
    bands = pd.Series([risk_band(s, operating_threshold, low_upper_bound) for s in scores])
    y = pd.Series(list(y_true))
    rows = []
    for band in BANDS:
        in_band = bands == band
        rows.append(
            {
                "band": band,
                "share_of_applicants": in_band.mean(),
                "default_rate": y[in_band].mean() if in_band.any() else float("nan"),
                "share_of_defaulters": y[in_band].sum() / y.sum(),
            }
        )
    return pd.DataFrame(rows).set_index("band")


if __name__ == "__main__":
    import json

    from src.models.select_final import OOF_PATH
    from src.models.train_final import FINAL_DIR, METADATA_FILE

    threshold = json.loads((FINAL_DIR / METADATA_FILE).read_text(encoding="utf-8"))["operating_threshold"]
    oof = pd.read_csv(OOF_PATH)
    print(band_summary(oof["y_true"], oof["xgboost_tuned"], threshold).round(3))
