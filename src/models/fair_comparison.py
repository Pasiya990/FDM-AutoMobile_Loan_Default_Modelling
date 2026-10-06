"""Fair Random Forest vs XGBoost comparison, on the leak-free split.

Why this exists: the untuned Random Forest looked far better than tuned
XGBoost. Two causes:
  1. Near-duplicate applicants (~20% of rows) sat on both sides of the CV
     folds and the train/test split; the fully grown RF memorised them.
     Fixed at the source - clean_and_split() now keeps one row per applicant.
  2. XGBoost had no class weighting, was tuned on ROC-AUC with unshuffled
     folds, and was judged at threshold 0.5, while the RF had balanced class
     weights and a tuned 0.21 threshold.

Here both models get the same treatment:
  - the same folds as evaluate_model() (stratified, shuffled, seed 42), with
    the project pipeline's preprocessing fitted on each training fold only;
  - a random search for each (XGBoost: imbalance weight searched, early
    stopping on an inner split of the training fold; RF: regularised trees),
    both scored by PR-AUC, ROC-AUC reported alongside;
  - the same operating-threshold rule on out-of-fold predictions
    (highest threshold with recall >= 0.60, as in notebook 06b).
The held-out test set is NOT touched here; train_final.py scores the selected
model on it once.

DATA_MODE=original reruns the RF and XGBoost baselines on the old, leaky
split (near-duplicates kept), to show how much the leak inflated them.

Usage (from the project root):
    python -m src.models.fair_comparison [stage ...]
    stages: cache rf xgb_base xgb_search rf_search oof   (default: all)
"""

import json
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import ParameterSampler, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from src.config import RANDOM_STATE, TARGET
from src.data.data_cleaning import clean_and_split
from src.models.xgboost_model import build_xgboost, imbalance_ratio
from src.preprocessing.feature_engineering import build_feature_engineering_steps
from src.preprocessing.preprocessing import build_preprocessing_steps

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_MODE = os.environ.get("DATA_MODE", "dedup")  # "dedup" (leak-free) or "original"
CACHE_DIR = PROJECT_ROOT / "data" / "processed" / f"fair_cv_cache_{DATA_MODE}"
RESULTS_ROOT = PROJECT_ROOT / "experiments" / "results" / "fair_comparison"
OUT_DIR = RESULTS_ROOT / DATA_MODE
SEARCH_DIR = RESULTS_ROOT / "dedup"  # searches always run on the leak-free split
SELECTION_PATH = SEARCH_DIR / "selection.json"

CV_FOLDS = 5
TARGET_RECALL = 0.60
N_SEARCH_ITER = 20
EARLY_STOPPING_ROUNDS = 50
MAX_TREES = 3000

RF_DEFAULT = dict(n_estimators=300, max_depth=None, min_samples_leaf=1, max_features="sqrt", class_weight="balanced")
XGB_DEFAULT = dict(max_depth=6, learning_rate=0.05, min_child_weight=1, subsample=0.8, colsample_bytree=0.8,
                   gamma=0, reg_alpha=0, reg_lambda=1, spw_factor=1.0)

XGB_SEARCH_SPACE = {
    "max_depth": [2, 3, 4, 5, 6, 8],
    "learning_rate": [0.02, 0.05, 0.1],
    "min_child_weight": [1, 5, 10, 20, 50],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.4, 0.5, 0.6, 0.8, 1.0],
    "gamma": [0, 0.1, 0.5, 1.0],
    "reg_alpha": [0, 0.1, 1.0, 5.0],
    "reg_lambda": [1, 2, 5, 10, 20],
    # multiplier on negatives/positives: 0 -> unweighted, 1 -> fully balanced
    "spw_factor": [0.0, 0.3, 0.5, 1.0],
}

RF_SEARCH_SPACE = {
    "n_estimators": [300, 500],
    "max_depth": [None, 10, 14, 18, 24],
    "min_samples_leaf": [1, 5, 10, 20, 50],
    "max_features": ["sqrt", 0.3, 0.5],
    "class_weight": ["balanced", "balanced_subsample"],
}


def build_preprocessor() -> Pipeline:
    return Pipeline(build_preprocessing_steps() + build_feature_engineering_steps())


def build_rf(**params) -> RandomForestClassifier:
    return RandomForestClassifier(**{**RF_DEFAULT, **params}, random_state=RANDOM_STATE, n_jobs=-1)


def spw_for(y, spw_factor) -> float:
    return 1.0 if spw_factor == 0 else imbalance_ratio(y) * spw_factor


# ---------------------------------------------------------------- metrics
def fold_metrics(y, proba, threshold=0.5):
    pred = (proba >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y, proba),
        "pr_auc": average_precision_score(y, proba),
        "recall": recall_score(y, pred, zero_division=0),
        "precision": precision_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
    }


def summarise_folds(name, rows):
    df = pd.DataFrame(rows)
    out = {"model_name": name, "threshold": 0.5}
    for m in ["roc_auc", "pr_auc"]:
        out[f"{m}_mean"] = df[m].mean()
        out[f"{m}_std"] = df[m].std(ddof=0)
    for m in ["recall", "precision", "f1"]:
        out[f"{m}_mean"] = df[m].mean()
    return out


def threshold_for_recall(y, proba, target=TARGET_RECALL):
    """Project rule (06b): highest threshold whose recall is still >= target."""
    _, recall, thresholds = precision_recall_curve(y, proba)
    eligible = np.where(recall[:-1] >= target)[0]
    return float(thresholds[eligible[-1]])


def threshold_max_f1(y, proba):
    precision, recall, thresholds = precision_recall_curve(y, proba)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    return float(thresholds[int(np.nanargmax(f1[:-1]))])


def at_threshold(y, proba, threshold):
    y = np.asarray(y)
    pred = (np.asarray(proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "threshold": round(threshold, 4),
        "recall": tp / (tp + fn),
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "f1": f1_score(y, pred, zero_division=0),
        "share_flagged": pred.mean(),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


# ---------------------------------------------------------------- folds
def stage_cache():
    """Fit preprocessing on each fold's training rows only; cache the transformed folds."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    train_df, _ = clean_and_split(drop_near_duplicates=(DATA_MODE == "dedup"))
    X, y = train_df.drop(columns=[TARGET]), train_df[TARGET]
    splits = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE).split(X, y)
    for k, (tr, va) in enumerate(splits):
        path = CACHE_DIR / f"fold{k}.joblib"
        if path.exists():
            continue
        t = time.time()
        pre = build_preprocessor().fit(X.iloc[tr], y.iloc[tr])
        joblib.dump(
            {
                "X_tr": pre.transform(X.iloc[tr]).astype("float32"),
                "y_tr": y.iloc[tr].to_numpy(),
                "X_va": pre.transform(X.iloc[va]).astype("float32"),
                "y_va": y.iloc[va].to_numpy(),
            },
            path,
        )
        print(f"fold {k} cached in {time.time() - t:.0f}s", flush=True)


def load_folds():
    return [joblib.load(CACHE_DIR / f"fold{k}.joblib") for k in range(CV_FOLDS)]


def fit_xgb_early_stopping(params, X, y, spw):
    """Early stopping on an inner 10% split of the TRAINING fold, never the scored fold.

    Stops on ROC-AUC, not logloss: with scale_pos_weight the probabilities are
    deliberately shifted, so unweighted logloss keeps "improving" for
    thousands of trees while ranking quality falls (fold 0, baseline params:
    logloss stopped at 2,457 trees, val ROC-AUC 0.706; auc at 126, 0.741).
    """
    X_fit, X_es, y_fit, y_es = train_test_split(X, y, test_size=0.1, stratify=y, random_state=RANDOM_STATE)
    model = build_xgboost(scale_pos_weight=spw, n_estimators=MAX_TREES, eval_metric="auc",
                          early_stopping_rounds=EARLY_STOPPING_ROUNDS, **params)
    model.fit(X_fit, y_fit, eval_set=[(X_es, y_es)], verbose=False)
    return model


def cv_xgb(cand, folds, oof=False):
    params = {k: v for k, v in cand.items() if k != "spw_factor"}
    rows, iters, probas = [], [], []
    for f in folds:
        m = fit_xgb_early_stopping(params, f["X_tr"], f["y_tr"], spw_for(f["y_tr"], cand["spw_factor"]))
        p = m.predict_proba(f["X_va"])[:, 1]
        rows.append(fold_metrics(f["y_va"], p))
        iters.append(m.best_iteration + 1)
        probas.append(p)
    return rows, iters, probas


def cv_rf(params, folds):
    rows, probas = [], []
    for f in folds:
        p = build_rf(**params).fit(f["X_tr"], f["y_tr"]).predict_proba(f["X_va"])[:, 1]
        rows.append(fold_metrics(f["y_va"], p))
        probas.append(p)
    return rows, probas


# ---------------------------------------------------------------- stages
def stage_rf():
    """Untuned RF exactly as src/models/random_forest.py."""
    t = time.time()
    rows, probas = cv_rf({}, load_folds())
    joblib.dump({"rows": rows, "probas": probas}, OUT_DIR / "rf_default.joblib")
    print("rf default:", summarise_folds("rf", rows), f"({time.time() - t:.0f}s)", flush=True)


def stage_xgb_base():
    """Original XGBoost baseline params (300 trees), unweighted vs weighted."""
    folds, out = load_folds(), {}
    for label, use_spw in [("xgb_baseline_no_weight", False), ("xgb_baseline_weighted", True)]:
        rows = []
        for f in folds:
            m = build_xgboost(scale_pos_weight=spw_for(f["y_tr"], float(use_spw))).fit(f["X_tr"], f["y_tr"])
            rows.append(fold_metrics(f["y_va"], m.predict_proba(f["X_va"])[:, 1]))
        out[label] = rows
        print(label, summarise_folds(label, rows), flush=True)
    joblib.dump(out, OUT_DIR / "xgb_baseline.joblib")


def _run_search(name, space, default, evaluate):
    """Random search, resumable from its CSV log; candidate 0 is the default config."""
    log_path = SEARCH_DIR / f"{name}_search_log.csv"
    done = pd.read_csv(log_path) if log_path.exists() else pd.DataFrame()
    candidates = [default] + list(ParameterSampler(space, n_iter=N_SEARCH_ITER, random_state=RANDOM_STATE))
    folds = load_folds()
    for i, cand in enumerate(candidates):
        if len(done) and (done["candidate"] == i).any():
            continue
        t = time.time()
        rows, extra = evaluate(cand, folds)
        s = summarise_folds(f"{name}_cand{i}", rows)
        rec = {"candidate": i, **{k: cand[k] for k in space}, **extra, **s, "seconds": round(time.time() - t, 1)}
        done = pd.concat([done, pd.DataFrame([rec])], ignore_index=True)
        done.to_csv(log_path, index=False)
        print(f"{name} cand {i}: pr_auc={s['pr_auc_mean']:.4f} roc_auc={s['roc_auc_mean']:.4f} "
              f"{extra} ({time.time() - t:.0f}s) {cand}", flush=True)


def stage_xgb_search():
    def evaluate(cand, folds):
        rows, iters, _ = cv_xgb(cand, folds)
        return rows, {"best_iter_mean": float(np.mean(iters))}
    _run_search("xgb", XGB_SEARCH_SPACE, XGB_DEFAULT, evaluate)


def stage_rf_search():
    def evaluate(cand, folds):
        rows, _ = cv_rf(cand, folds)
        return rows, {}
    _run_search("rf", RF_SEARCH_SPACE, {**RF_DEFAULT}, evaluate)


def _clean_value(v):
    if isinstance(v, float) and np.isnan(v):
        return None  # max_depth=None round-trips through CSV as NaN
    if isinstance(v, (np.integer,)) or (isinstance(v, float) and v.is_integer()):
        return int(v)
    return v.item() if isinstance(v, np.generic) else v


def best_from_log(name, space):
    log = pd.read_csv(SEARCH_DIR / f"{name}_search_log.csv")
    best = log.sort_values("pr_auc_mean", ascending=False).iloc[0]
    params = {}
    for k in space:
        v = best[k]
        if isinstance(v, str):
            try:
                v = float(v)  # max_features "0.3" -> 0.3; "sqrt" stays a string
            except ValueError:
                pass
        params[k] = _clean_value(v)
    if "spw_factor" in params:
        params["spw_factor"] = float(params["spw_factor"])
    if "max_features" in params and isinstance(params["max_features"], int):
        params["max_features"] = float(params["max_features"])
    return params, best


def stage_oof():
    """OOF predictions for every finalist on the same folds, CV table, and both threshold rules."""
    folds = load_folds()
    y_oof = np.concatenate([f["y_va"] for f in folds])
    rf_params, _ = best_from_log("rf", RF_SEARCH_SPACE)
    xgb_params, _ = best_from_log("xgb", XGB_SEARCH_SPACE)

    finalists = {}
    rf_cv = joblib.load(OUT_DIR / "rf_default.joblib")
    finalists["random_forest_default"] = (rf_cv["rows"], rf_cv["probas"], {**RF_DEFAULT})
    rows, probas = cv_rf(rf_params, folds)
    finalists["random_forest_tuned"] = (rows, probas, rf_params)
    rows, iters, probas = cv_xgb(xgb_params, folds)
    xgb_n_estimators = int(round(np.mean(iters) * 1.1))  # full train split is 25% larger than a fold
    finalists["xgboost_tuned"] = (rows, probas, {**xgb_params, "n_estimators": xgb_n_estimators})

    base = joblib.load(OUT_DIR / "xgb_baseline.joblib")
    cv_table = pd.DataFrame(
        [summarise_folds(n, r) for n, (r, _, _) in finalists.items()]
        + [summarise_folds(n, base[n]) for n in ("xgb_baseline_no_weight", "xgb_baseline_weighted")]
    ).sort_values("pr_auc_mean", ascending=False)

    oof_rows, thresholds = [], {}
    for name, (_, probas, _) in finalists.items():
        p = np.concatenate(probas)
        thresholds[name] = threshold_for_recall(y_oof, p)
        for rule, t in [(f"recall>={TARGET_RECALL}", thresholds[name]), ("max_f1", threshold_max_f1(y_oof, p))]:
            oof_rows.append({"model": name, "rule": rule, "oof_roc_auc": roc_auc_score(y_oof, p),
                             "oof_pr_auc": average_precision_score(y_oof, p), **at_threshold(y_oof, p, t)})
    oof_table = pd.DataFrame(oof_rows)

    cv_table.to_csv(OUT_DIR / "cv_comparison.csv", index=False)
    oof_table.to_csv(OUT_DIR / "oof_threshold_comparison.csv", index=False)
    pd.set_option("display.width", 220)
    print(cv_table.round(4).to_string(index=False))
    print(oof_table.round(3).to_string(index=False))

    if DATA_MODE != "dedup":
        return
    # Selection on CV only: highest mean PR-AUC among the finalists
    winner = cv_table[cv_table["model_name"].isin(finalists)].iloc[0]["model_name"]
    rule_row = oof_table[(oof_table["model"] == winner) & oof_table["rule"].str.startswith("recall")].iloc[0]
    cv_row = cv_table.set_index("model_name").loc[winner]
    selection = {
        "model": winner,
        "family": "xgboost" if winner.startswith("xgboost") else "random_forest",
        "params": finalists[winner][2],
        "selected_by": "highest mean 5-fold CV PR-AUC on the leak-free training split",
        "operating_threshold": float(thresholds[winner]),
        "target_recall": TARGET_RECALL,
        "oof_at_threshold": {k: float(rule_row[k]) for k in ("recall", "precision", "share_flagged")},
        "cv_reference": {k: float(cv_row[k]) for k in ("roc_auc_mean", "roc_auc_std", "pr_auc_mean", "pr_auc_std")},
    }
    SELECTION_PATH.write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    print("selected:", json.dumps(selection, indent=2))


STAGES = {"cache": stage_cache, "rf": stage_rf, "xgb_base": stage_xgb_base,
          "xgb_search": stage_xgb_search, "rf_search": stage_rf_search, "oof": stage_oof}

if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SEARCH_DIR.mkdir(parents=True, exist_ok=True)
    chosen = sys.argv[1:] or list(STAGES)
    for s in chosen:
        print(f"=== stage {s} ({DATA_MODE})", flush=True)
        STAGES[s]()
