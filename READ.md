# READ: Model Development to Final System (Whole Team)

**Vehicle Loan Default Prediction · IT3051 Mini Project 2026**

This file is the team's reference for everything after feature engineering: the shared protocol, every model, tuning,
the final model, the threshold and risk bands, the extra analyses (ablation, explanations, fairness) and the prediction
system. Every number comes from a file in `experiments/results/` or `models/final/metadata.json`. The reasons behind
each decision are in [docs/decision_log.md](docs/decision_log.md).

> **Protocol for every model number:** stratified, shuffled 5-fold cross-validation on the **cleaned** training split
> (85,756 rows, 8.1% default, near-duplicate applicants removed), `RANDOM_STATE = 42`. The held-out test set
> (21,440 rows) was used once, for the final model only.

---

## 1. Status at a glance

| Area | Status |
|---|---|
| Data cleaning, near-duplicate removal, stratified split | Done (`src/data/`) |
| Shared leak-free pipeline and CV harness | Done (`src/preprocessing/`, `src/evaluation/metrics.py`) |
| Seven algorithms, baselines and tuning | Done (`notebooks/05*`, `07*`) |
| Model comparison | Done (`notebooks/06b_model_comparison.ipynb`) |
| Final model: tuned XGBoost, test set used once | Done (`models/final/`) |
| Operating threshold (60% recall) and risk bands | Done |
| Ablation study (Stage 7: does feature engineering help?) | Done (`experiments/results/ablation.csv`) |
| Explanations (top factors per prediction) | Done (`backend/explain.py`) |
| Fairness check | Done; team decision open (section 11) |
| Backend (FastAPI) and frontend (React) | Done, 123 tests pass, checked on a clean machine |
| Technical report and presentation | Not started |

---

## 2. Data and the split

`clean_and_split()` in `src/data/data_cleaning.py`, run once before any modelling:

| Step | Rows |
|---|---|
| Raw file | 121,856 |
| Exact duplicates removed (ignoring ID) | -2,640 |
| Near-duplicate applicant copies removed (`src/data/near_duplicates.py`) | -12,020 |
| Remaining | 107,196 |
| Stratified 80 / 20 split | 85,756 train · 21,440 test, both 8.1% default |

**Why near-duplicates matter.** Many applicants appeared twice, identical except for a field left blank in one copy.
They survive exact-duplicate removal and ended up on both sides of the CV folds and the train/test split, so models
partly "recognised" applicants instead of predicting. The untuned Random Forest scored ROC-AUC 0.774 with them and
0.730 without them: about 0.04 of its score was leakage. All numbers in this file are on the cleaned split.

Row-level fixes in the same step: text numbers parsed, the `Employed_Days` code 365243 turned into a blank plus the
`Is_Retired_Or_Unemployed` flag, placeholders (`XNA`, `##`) and impossible values above 1 turned into blanks,
`Own_House_Age` reinterpreted as `car_age` with a `has_car_age` flag, `ID` dropped.

---

## 3. The shared pipeline and protocol

**Pipeline** (`build_pipeline(model)` in `src/preprocessing/feature_engineering.py`): one scikit-learn `Pipeline`
whose steps learn only from the training data they are fitted on, then the model as the last step:

input check → score summaries → imputation with missing flags → income ratios → log income → one-hot encoding →
rare-category grouping → derived features (age in years, credit/annuity ratio, ...) → importance-based feature
selection → scaling → model

Because every step is inside the pipeline, cross-validation refits it per fold (no leakage), and the backend applies
exactly the same steps to one new application.

**Fixed during backend work:** the one-hot encoder dropped the "first" category of whatever rows it was given. For a
single applicant that was the applicant's own category, so every category column became 0 and one-at-a-time scores
were off by up to 0.22. Batch results, and therefore every reported number, were unaffected (difference 0.0). Fixed in
`src/preprocessing/preprocessing.py`, with a test that single rows match the batch.

**Harness** (`evaluate_model` in `src/evaluation/metrics.py`): stratified 5-fold CV, shuffle, seed 42, a fresh clone of
the pipeline per fold; reports ROC-AUC, PR-AUC, and recall/precision/F1 at a threshold.

**Metrics.** ROC-AUC is the primary, threshold-free metric; PR-AUC is the tie-break because only 8% default.
Accuracy is not used: always predicting "no default" already scores 91.9%. Recall, precision and F1 at 0.5 are **not
comparable across models**, because models with and without class weights place their scores differently; the
operating threshold is chosen separately (section 7).

---

## 4. The models

| Rank | Model | Builder | CV ROC-AUC | CV PR-AUC |
|---|---|---|---|---|
| 1 | XGBoost (tuned) | `src/models/xgboost_model.py` | 0.745 ± 0.004 | 0.229 ± 0.007 |
| 2 | LightGBM (tuned) | `src/models/lightgbm_model.py` | 0.744 ± 0.004 | 0.226 ± 0.007 |
| 3 | LightGBM (baseline) | same | 0.742 ± 0.003 | 0.218 ± 0.005 |
| 4 | XGBoost (baseline) | same as tuned | 0.738 ± 0.005 | 0.221 ± 0.005 |
| 5 | Random Forest (tuned) | `src/models/random_forest.py` | 0.736 ± 0.004 | 0.215 ± 0.006 |
| 6 | Logistic Regression | `src/models/logistic_regression.py` | 0.733 ± 0.003 | 0.212 ± 0.008 |
| 7 | Random Forest (untuned) | same as tuned | 0.730 ± 0.003 | 0.209 ± 0.005 |
| 8 | Decision Tree | `src/models/decision_tree.py` | 0.708 ± 0.005 | 0.178 ± 0.002 |
| 9 | Gaussian Naive Bayes | `src/models/gaussian_naive_bayes.py` | 0.697 ± 0.003 | 0.181 ± 0.006 |

**Why they land where they do:**

- **Boosted trees (XGBoost, LightGBM)** build trees one after another, each correcting the previous ones' errors, and
  handle interactions and missing values well. They are usually strongest on tabular credit data, and are here.
- **Random Forest** averages many deep trees. Untuned, its trees grow fully and partly memorise the training data;
  tuning (depth 12, leaves of at least 20) regularised it and helped once the leak was removed.
- **Logistic Regression** draws a straight-line boundary, so it misses interactions. It stays close behind the trees,
  possibly because the strongest signal (the external scores) moves risk in one direction; this was not tested.
- **Decision Tree**: one depth-limited tree gives coarse, step-like scores.
- **Gaussian Naive Bayes** assumes the features are independent given the class; the scores and ratio features are
  strongly correlated, so the assumption breaks.

Class imbalance (8% default) is handled with class weights: `class_weight="balanced"` for Random Forest, Decision Tree
and (baseline) Logistic Regression, `scale_pos_weight` ≈ 11.35 (non-defaulters / defaulters) for XGBoost and a
similar weight for LightGBM.

---

## 5. Tuning

All searches: `RandomizedSearchCV`, stratified 5-fold CV, ROC-AUC, training split only.

| Model | Baseline ROC-AUC | Tuned ROC-AUC | Best settings |
|---|---|---|---|
| XGBoost (`07_hyperparameter_tuning.ipynb`) | 0.738 | 0.745 | 400 trees, depth 4, learning rate 0.05, subsample 0.7, column sample 0.7, gamma 0.5, `reg_alpha` 1, `reg_lambda` 10 |
| LightGBM (`lightgbm.ipynb`) | 0.742 | 0.744 | 300 trees, depth 4, 31 leaves, learning rate 0.05, column sample 0.7, `scale_pos_weight` 10 |
| Random Forest (`07b_random_forest_tuning.ipynb`) | 0.730 | 0.736 | 500 trees, depth 12, at least 20 per leaf, `sqrt` features, `balanced_subsample` |
| Logistic Regression (`07a_logistic_regression_tuning.ipynb`) | 0.733 | see notebook | C = 0.01, no class weight |

The best settings are all strongly regularised (shallow boosted trees, leaf limits, L1/L2 penalties): with a limited signal, overfitting is the main risk.

---

## 6. Final model selection

**XGBoost (tuned)** is the final model: it ranked first on the team's rule (highest mean CV ROC-AUC, PR-AUC as tie-break).

Be honest about the margin in the viva: XGBoost tuned leads LightGBM tuned by 0.0007 ROC-AUC, far inside the
fold-to-fold spread (about 0.004). The top two are statistically tied; XGBoost was kept because it was first in the
table, has the higher PR-AUC, and its explanations (section 10) need no extra library.

**How the final model was built.** `src/models/select_final.py` takes the notebook-07 settings, re-runs 5-fold CV
(ROC-AUC 0.7454 ± 0.0045, PR-AUC 0.228 ± 0.008, matching notebook 07), chooses the threshold from the out-of-fold
predictions, and writes `experiments/results/fair_comparison/dedup/selection.json`. Then `src/models/train_final.py`
fits on the full training split, scores the test set **once**, and saves `models/final/final_pipeline.joblib` and
`metadata.json`. (`src/models/fair_comparison.py` contains a longer search with early stopping; it was not run for the
final selection.)

---

## 7. Operating threshold and risk bands

**Threshold 0.522.** The highest threshold that still catches 60% of defaulters in the out-of-fold training
predictions (recall target agreed by the team; a missed defaulter costs far more than an extra review). It is above
0.5 because the class weight pushes scores up. At 0.522 on the training predictions: recall 0.600, precision 0.180,
27.0% of applicants flagged.

**Risk bands** (`src/models/risk_bands.py`, cut-offs saved in `metadata.json`):

| Band | Risk score | Applicants | Past default rate | Share of all defaulters |
|---|---|---|---|---|
| Low | below 0.25 | 28% | 2.3% | 8% |
| Medium | 0.25 to 0.52 | 45% | 5.8% | 32% |
| High (= flagged) | 0.52 and above | 27% | 18.0% | 60% |

The score is a **risk score, not a probability**: with class weights, 0.6 does not mean a 60% chance of default. The
interface says so.

---

## 8. Final test result (held-out test set, used once)

| ROC-AUC | PR-AUC | Recall | Precision | Flagged | Caught / missed defaulters | Good applicants flagged |
|---|---|---|---|---|---|---|
| 0.760 | 0.247 | 0.627 | 0.187 | 27.2% | 1,088 / 647 | 4,739 |

In plain words: the model catches about 6 in 10 defaulters by flagging about 27 in 100 applicants; about 1 in 5
flagged applicants really defaults. The test results are close to the CV estimates (0.745, 0.600, 0.180), so the
threshold held on unseen data.

---

## 9. Ablation study (Stage 7: does feature engineering help?)

`src/models/ablation.py`, same folds, change measured per fold against the full pipeline:

| Variant | ROC-AUC | Change | PR-AUC | Change |
|---|---|---|---|---|
| Full final pipeline (55 features) | 0.745 | | 0.228 | |
| Without engineered features | 0.739 | -0.007 ± 0.002 | 0.221 | -0.008 ± 0.002 |
| Without feature selection (102 features) | 0.746 | +0.000 ± 0.002 | 0.226 | -0.002 ± 0.002 |
| Without the three bureau scores | 0.691 | **-0.055 ± 0.006** | 0.173 | **-0.055 ± 0.006** |
| Without `Active_Loan` and `Social_Circle_Default` | 0.744 | -0.001 ± 0.002 | 0.227 | -0.001 ± 0.003 |

- The engineered features earn their place: a small loss, but consistent across folds.
- Feature selection halves the inputs at no cost in accuracy, so it is kept for simplicity.
- The model depends heavily on the external bureau scores: applicants without them are scored less reliably (a
  limitation for the report).
- The two columns with uncertain meaning make no measurable difference; they were kept to avoid refitting the final
  model and reusing the test set (decision log D12).

---

## 10. Explanations

`backend/explain.py` uses XGBoost's built-in exact TreeSHAP (`pred_contribs=True`): every model feature gets a
contribution, and the contributions add up to the model's score (tested). They are summed into readable factors
("External credit scores", "Loan size compared with income", "Employer type", ...) and the four strongest are returned
with "raises risk" or "lowers risk". The external scores are the strongest factor for most applicants, which agrees
with the ablation. The factors explain the model, not the causes of default.

---

## 11. Fairness check

`src/models/fairness.py`, on out-of-fold training predictions at the operating threshold:

| Attribute | Good applicants wrongly flagged (false-positive rate) | Precision |
|---|---|---|
| Age | 61-69: 10% · 21-30: **40%** | 13% to 19% |
| Gender | Male: 20% · Female: **33%** | 17% to 19% |
| Marital status | Widowed: 13% · Single: 30% | 17% to 19% |
| Education | Graduation: 13% · Secondary: 29% | 15% to 22% |
| Income type | Retired: 12% · Service: 31% | 14% to 19% |

- Every attribute has a gap above the plan's 10-point limit. The gaps largely follow real differences in default rates
  (21-30: 11.0% default vs 61-69: 5.2%; women 10.0% vs men 7.1%).
- Precision is similar across groups: a flag means about the same for everyone. But good applicants from some groups
  are sent to extra review far more often.
- **Without gender, marital status and age:** ROC-AUC 0.745 → 0.743 (-0.003); the gender gap falls below the limit
  (false-positive rates 22% vs 30%); the age gap does not shrink, because age is carried by other inputs (for example
  the external scores and the retired income type).
- **Open team decision:** keep the final model and report the gaps (recommended for this submission, since a person
  reviews every flag), or refit without gender and marital status (fairer between men and women, but uses the test
  set a second time).

---

## 12. The prediction system

| Part | Files | What it does |
|---|---|---|
| Model loading | `backend/model_store.py` | Loads the pipeline and metadata once; 503 if missing; warns on library-version mismatch |
| Input adapter | `backend/adapter.py` | Form fields (age in years, "not employed", yes/no) → the model's columns, repeating the cleaning rules |
| Validation | `backend/validation.py` | Required fields, types, limits and categories from the saved training schema; 422 with per-field details |
| API | `backend/app.py`, `backend/routes/prediction.py` | `/health`, `/schema`, `/examples`, `/predict`; also serves the web page |
| Explanations | `backend/explain.py` | Top factors per prediction |
| Frontend | `frontend/src/` (React, built to `frontend/dist/`) | Header, application form, example applicants, results beside the form, How to use page, print summary |

`/predict` returns the prediction in words, risk score, band, threshold, top factors, suggested action, model
version and a disclaimer. Run it with `uvicorn backend.app:app --reload` and open http://127.0.0.1:8000/ (README).

**Testing:** 123 automated tests (`pytest -q`), including: `/predict` equals the pipeline called directly on 100 real
test applications; every kind of invalid input; the model missing; single rows scored exactly like batches; SHAP
contributions adding up to the score. The whole suite was also run in a fresh Python environment from
`requirements.txt`, which found and fixed a missing `pyarrow` dependency and unpinned library versions.

---

## 13. Known issues and limitations

1. **Bureau-score dependence:** without the external scores, ranking quality drops sharply (section 9).
2. **Fairness gaps** in who gets flagged (section 11).
3. **Undocumented data:** the dataset's source, currency, period and some column meanings (`Active_Loan`,
   `Social_Circle_Default`, `Own_House_Age`) are not documented; there is no application date, so no time-based test.
4. **Day of application:** the data codes days 0 to 6; 0 is taken to be Sunday (it is the quietest day). An inference.
5. **Risk score, not probability:** scores are shifted by the class weight; calibration is future work.
6. **Result files:** `experiments.csv` and `xgboost_baseline_vs_tuned.csv` still contain two old XGBoost Baseline rows
   from before near-duplicate removal; `06b` keeps the most recent row per model, so its table is correct.
7. **Test-set use by teammates:** the Logistic Regression notebooks also report test-set results
   (`logistic_regression_test.csv`, `logistic_regression_tuned_test.csv`). They were not used to choose the final
   model; say so if asked.

---

## 14. Likely viva questions

**Why not accuracy?** Always predicting "no default" scores 91.9% and catches nobody.

**Why is the threshold not 0.5?** At 0.5 a model trained on 8% defaulters catches few of them. We chose the
threshold for 60% recall on training data only; it is 0.522 here because class weights push scores up.

**Why XGBoost?** Highest CV ROC-AUC under the team's rule, though statistically tied with LightGBM; boosted trees
model interactions and missing values well.

**Why did scores fall compared with earlier runs?** Near-duplicate applicants inflated them; removing them is the
honest estimate.

**How do you know the system uses the same preprocessing as training?** The saved pipeline contains every
preprocessing step, and a test shows `/predict` returns exactly the pipeline's own score on 100 real applications.

**Did feature engineering help?** Yes, modestly and consistently (-0.007 ROC-AUC without it); the bureau scores
matter most.

**Is the model fair?** It flags some groups more, largely following differences in past default rates; precision is
similar across groups. Removing gender narrows the gender gap; removing age does not narrow the age gap because other
inputs carry age information. A person reviews every flag; the tool never refuses automatically.

**Why was the test set used only once?** Choosing anything on it would make the reported score optimistic.
