# Decision Log

**Vehicle Loan Default Prediction · IT3051 Mini Project 2026**

One row per important decision: what was decided, the evidence for it, what else was considered, and where to find
it. Numbers are cross-validation results on the training split unless marked "test". The full story is in
[READ.md](../READ.md).

## Data and preprocessing

| ID | Decision | Evidence | Alternatives considered | Where |
|---|---|---|---|---|
| D1 | Remove 2,640 exact duplicate rows (ignoring `ID`) before splitting | Identical applications would sit on both sides of the split and inflate scores | Keep them | `src/data/data_cleaning.py` |
| D2 | Remove 12,020 near-duplicate applicant copies before splitting | Copies identical except for one blank field leaked across folds and the test split; untuned Random Forest ROC-AUC 0.774 with them, 0.730 without | Keep them; group-aware split | `src/data/near_duplicates.py` |
| D3 | `Employed_Days` code 365243 → blank plus `Is_Retired_Or_Unemployed` flag | The code marks retired or unemployed applicants, not 1,000 years of work | Treat as a number; drop the column | `src/data/data_cleaning.py` |
| D4 | Treat `Own_House_Age` as car age (`car_age`, `has_car_age`) | Filled only when `Car_Owned` = 1 and unrelated to `House_Own` | Use it as house age | notebook 03 |
| D5 | Stratified 80 / 20 split, seed 42; test set used once, for the final model | Keeps the 8.1% default rate in both parts; one use keeps the test estimate honest | Re-using the test set for choices | `src/data/data_cleaning.py`, `src/models/train_final.py` |
| D6 | Every learned preprocessing step inside one scikit-learn pipeline | Prevents leakage in CV; the backend applies the exact training steps to one application | Preprocess the whole dataset first | `src/preprocessing/` |
| D7 | Keep feature engineering and importance-based selection | Ablation: without engineered features ROC-AUC -0.007 ± 0.002; without selection +0.000 with 102 instead of 55 features | Raw features only; no selection | `src/models/ablation.py`, `experiments/results/ablation.csv` |

## Models and evaluation

| ID | Decision | Evidence | Alternatives considered | Where |
|---|---|---|---|---|
| D8 | ROC-AUC as the primary metric, PR-AUC as the tie-break; accuracy not used | 8.1% default: always predicting "no default" scores 91.9% accuracy | Accuracy; F1 at 0.5 | `src/evaluation/metrics.py` |
| D9 | Class weights for the 8% minority class | Balanced weights for the tree models, `scale_pos_weight` ≈ 11.35 for XGBoost (not varied in its search), 10 for tuned LightGBM; tuned Logistic Regression chose no weight | SMOTE (proposal mentioned it; not tested) | `src/models/` |
| D10 | Seven algorithms compared under one protocol | The assignment requires at least four; one protocol makes the comparison fair | Different folds per model | `notebooks/06b_model_comparison.ipynb` |
| D11 | Final model: tuned XGBoost | Highest CV ROC-AUC 0.745 (LightGBM tuned 0.744, a statistical tie; XGBoost has the higher PR-AUC, 0.229 vs 0.226) | LightGBM tuned; Random Forest | `src/models/select_final.py`, `selection.json` |
| D12 | Keep `Active_Loan` and `Social_Circle_Default` | Removing them changes ROC-AUC by -0.001 ± 0.002, within noise; removing them would mean refitting the final model and using the test set again | Drop them (the plan's rule for uncertain columns) | `experiments/results/ablation.csv` |

## Threshold, bands and system

| ID | Decision | Evidence | Alternatives considered | Where |
|---|---|---|---|---|
| D13 | Operating threshold 0.522, the highest that reaches 60% recall on out-of-fold training predictions | At 60% recall: precision 0.180, 27% flagged; test: recall 0.627, precision 0.187 | 0.5 (catches far fewer defaulters); a recall target of 0.70 | `selection.json`, `models/final/metadata.json` |
| D14 | Risk bands: Low below 0.25, Medium 0.25 to 0.52, High from 0.52 | Past default rates 2.3%, 5.8%, 18.0%; Low holds only 8% of defaulters | Low cut-off 0.20 | `src/models/risk_bands.py` |
| D15 | Show a "risk score", not a probability | Class weights shift the scores; 0.6 does not mean 60% | Calibrate the scores (future work) | `backend/messages.py` |
| D16 | Explanations from XGBoost's built-in TreeSHAP, grouped into readable factors | Exact contributions that add up to the score (tested); no extra library to pin | The `shap` package; permutation importance | `backend/explain.py` |
| D17 | Validation limits are the training extremes; typical ranges are hints only | Percentile limits would reject the 2% of real training applicants outside the range of each field, and more across all fields together | Percentile limits | `src/models/model_metadata.py` |
| D18 | Six required fields: age, income, credit amount, annuity, contract type, employment status | A loan officer always knows them; everything else may be blank and is imputed as in training | Require every field; require none | `backend/validation.py` |
| D19 | Decision support only; never automatic refusal | Proposal sections 3 and 10; about 81% of flagged test applicants repaid (precision 0.187) | Automatic decisions | `backend/messages.py`, frontend |
| D20 | Pin the library versions the saved model was trained with | A clean install failed to load the model without `pyarrow`, and pulled newer pandas and scikit-learn | Unpinned requirements | `requirements.txt` |

## Open decisions

| ID | Question | Options | Evidence | Recommendation |
|---|---|---|---|---|
| O1 | Fairness: gender, marital status and age | Keep the final model and report the gaps; or refit without gender and marital status | Gaps above 10 points for every attribute; without the three attributes ROC-AUC -0.003, gender gap below the limit, age gap unchanged | Keep for this submission, report the gaps and recommend the refit as future work |
