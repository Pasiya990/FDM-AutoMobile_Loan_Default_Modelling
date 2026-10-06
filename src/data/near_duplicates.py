"""Near-duplicate applicant detection and removal.

The raw data contains the same application more than once with small
differences - typically one copy has a few values blanked to NaN - so
remove_duplicate_rows() (exact matches only) does not catch them. About 20%
of rows sit in such a group, and their labels agree in >99.9% of groups. If
the copies land on both sides of a CV fold or the train/test split, a model
that memorises rows (e.g. a fully grown Random Forest) predicts the second
copy almost perfectly, inflating its scores.

Detection is two-step:
  1. Candidate pairs: rows that agree on ALL columns of ANY near-unique
     "fingerprint" key (several keys, so a copy with one key column blanked
     is still found).
  2. Confirmation: a candidate pair is linked only if the two rows never
     disagree on a column where both have a value. This rejects pairs that
     share a fingerprint by coincidence, or the same person's separate loan
     application (different Credit_Amount/Loan_Annuity).
Confirmed pairs are merged with union-find into applicant groups.

Runs on the raw columns (after fix_dtypes, before the sentinel/placeholder
fixes), so derived flags such as has_car_age cannot disagree just because
their source value was blanked in one copy.
"""

import itertools

import numpy as np
import pandas as pd

from src.config import ID_COL, TARGET

# Each key is a set of high-cardinality columns that, jointly, identify one
# application. A pair is a candidate if ALL columns of ANY key match exactly.
FINGERPRINT_KEYS = [
    ["Age_Days", "ID_Days", "Registration_Days"],
    ["Age_Days", "Credit_Amount", "Phone_Change"],
    ["ID_Days", "Registration_Days", "Score_Source_2"],
    ["Age_Days", "Employed_Days", "Client_Income"],
    ["Registration_Days", "Phone_Change", "Loan_Annuity"],
]

# A fingerprint value shared by more rows than this is not near-unique
MAX_KEY_GROUP_SIZE = 50


def _find(parent, i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


def _candidate_pairs(df: pd.DataFrame, keys) -> np.ndarray:
    """Positional (i, j) pairs, i < j, that share every column of some key."""
    pairs = set()
    for key in keys:
        sub = df[key].reset_index(drop=True).dropna()
        sub = sub[sub.duplicated(keep=False)]  # only rows sharing their fingerprint
        for idx in sub.groupby(key, sort=False).indices.values():
            if len(idx) <= MAX_KEY_GROUP_SIZE:
                pairs.update(itertools.combinations(sorted(sub.index[idx]), 2))
    return np.array(sorted(pairs), dtype=int).reshape(-1, 2)


def _compatible(df: pd.DataFrame, pairs: np.ndarray) -> np.ndarray:
    """True where the two rows never disagree on a column both have a value in."""
    cols = [c for c in df.columns if c not in (ID_COL, TARGET)]
    a = df[cols].iloc[pairs[:, 0]].reset_index(drop=True).astype(object)
    b = df[cols].iloc[pairs[:, 1]].reset_index(drop=True).astype(object)
    conflicts = (a.notna() & b.notna() & (a != b)).sum(axis=1)
    return (conflicts == 0).to_numpy()


def assign_applicant_groups(df: pd.DataFrame, keys=FINGERPRINT_KEYS) -> pd.Series:
    """Return a group id per row (aligned to df.index); copies share an id."""
    n = len(df)
    parent = np.arange(n)
    pairs = _candidate_pairs(df, keys)
    if len(pairs):
        pairs = pairs[_compatible(df, pairs)]
    for i, j in pairs:
        ri, rj = _find(parent, i), _find(parent, j)
        if ri != rj:
            parent[rj] = ri
    roots = np.array([_find(parent, i) for i in range(n)])
    return pd.Series(pd.factorize(roots)[0], index=df.index, name="applicant_group")


def remove_near_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Keep one row per applicant group: the most complete copy (fewest
    missing values), ties broken by original row order."""
    groups = assign_applicant_groups(df)
    missing = df.isna().sum(axis=1).to_numpy()
    order = np.lexsort((np.arange(len(df)), missing, groups.to_numpy()))
    keep = order[~pd.Series(groups.to_numpy()[order]).duplicated().to_numpy()]
    return df.iloc[np.sort(keep)].reset_index(drop=True)
