# READ: Model Development After Feature Engineering (Whole Team)

**Automobile Loan Default Prediction · IT3051 Mini Project 2026**

This file covers every model built after the feature-engineering stage, the shared protocol they were all scored
with, the comparison, and the open problems. It's written for the team viva, so it covers all models, not one
member's work.

> **Protocol for every number here:** stratified, shuffled 5-fold cross-validation on the training split
> (95,372 rows, 8.1% default), `RANDOM_STATE = 42`, default 0.5 threshold. Test-set numbers appear only where a
> notebook reports them, and they are labelled as reference only. No model was selected on the test set.

---

## 1. Status at a glance

| Model | Builder | Notebook | CV ROC-AUC | Protocol status |
|---|---|---|---|---|
| Logistic Regression | `src/models/logistic_regression.py` | `05a_logistic_regression.ipynb` | 0.734 ± 0.005 | verified |
| Decision Tree | `src/models/decision_tree.py` | `decision_tree.ipynb` | 0.708 ± 0.004 | **unverified** |
| Random Forest (untuned) | `src/models/random_forest.py` | `05_baseline_models_random_forest.ipynb` | 0.774 ± 0.004 | verified |
| Random Forest (tuned) | same builder | `07b_random_forest_tuning.ipynb` | 0.770 ± 0.004 | verified, not kept |
| Gaussian Naive Bayes | `src/models/gaussian_naive_bayes.py` | `05b_gaussian_naive_bayes.ipynb` | 0.697 ± 0.004 | verified |
| XGBoost (baseline) | `src/models/xgboost_model.py` | `05c_XGBoost_Baseline.ipynb` | 0.763 ± 0.004 | verified |
| XGBoost (tuned) | same builder | `07_hyperparameter_tuning.ipynb` | 0.771 ± 0.004 | verified |
| LightGBM | `src/models/lightgbm_model.py` | `lightgbm.ipynb` | **none** | **pending** |

| Area | Status |
|---|---|
| Pipeline refactor (fit/transform, one `build_pipeline`) | Done |
| Shared CV harness and logger | Done |
| Model comparison table | Done (`06b_model_comparison.ipynb`) |
| Comparison plots (ROC/PR overlay) | Pending |
| Leading-candidate decision | Proposed, needs team confirmation |
| Operating threshold | Not chosen (team decision) |
| Final model and test-set run | Not started (team decision) |

---

## 2. The shared pipeline and harness

### 2.1 Pipeline

The original pipeline recomputed statistics from both train and test on each call, so it couldn't transform a
single new application. It is now one fitted `sklearn.Pipeline`, built by `build_pipeline(model)`.

| File | Contains |
|---|---|
| [src/config.py](src/config.py) | Shared constants: `RANDOM_STATE`, `TARGET`, column lists, thresholds |
| [src/data/data_cleaning.py](src/data/data_cleaning.py) | Row-level cleaning before the split: numeric coercion, duplicate removal, sentinel and placeholder fixes, stratified split |
| [src/preprocessing/preprocessing.py](src/preprocessing/preprocessing.py) | Fit-on-train transformers: `ScoreSummaryAdder`, `MissingValueImputer`, `IncomeRatioAdder`, `IncomeLogTransformer`, `CategoricalOneHotEncoder` |
| [src/preprocessing/feature_engineering.py](src/preprocessing/feature_engineering.py) | `RareCategoryGrouper`, `DerivedFeatureAdder`, `ImportanceFeatureSelector`, `SelectiveStandardScaler`, `build_pipeline(model)` |

**Verified:** after transformation the training data has 95,372 rows and 46 features, with zero missing values.
The importance selector drops 58 features, and the top feature is `score_mean` (importance 0.227). A single new
application can be transformed and scored by the fitted pipeline.

**Tests:** [tests/test_pipeline.py](tests/test_pipeline.py) has 16 passing tests: the split, no-leakage checks,
single-row transformation, and unseen categories.

### 2.2 Evaluation harness

[src/evaluation/metrics.py](src/evaluation/metrics.py):

- `evaluate_model(pipeline, X, y, cv=5, threshold=0.5)` runs stratified, shuffled 5-fold CV with seed 42. It clones
  the pipeline for each fold, so no fitted state leaks between folds. It reports ROC-AUC, PR-AUC, recall, precision,
  F1, and the confusion matrix.
- `log_result_to_csv(results, path)` writes one row per model name and replaces any old row, so reruns don't
  duplicate.

---

## 3. Models, one by one

### 3.1 Logistic Regression (teammate, `05a_logistic_regression.ipynb`)

- **Builder:** `build_logistic_regression_pipeline()`: L2 regularisation, `C=1.0`, `class_weight="balanced"`,
  `max_iter=1000`, `lbfgs`. It is the untuned linear baseline.
- **CV result:** ROC-AUC 0.734 ± 0.005, PR-AUC 0.212 ± 0.005, recall 0.668, precision 0.153, F1 0.249.
- **Reference only, test set:** ROC-AUC 0.743, PR-AUC 0.215, recall 0.688. At 0.5 it caught 1,329 of 1,931
  defaulters and flagged 7,227 non-defaulters.
- **Decision recorded in the notebook:** keep as the untuned, interpretable linear baseline. It has much higher
  recall at 0.5 than Random Forest, but Random Forest ranks better overall.
- **Why it's behind:** a straight-line model can't draw curves or interactions, so it underfits.

### 3.2 Decision Tree (teammate, `decision_tree.ipynb`)

- **Builder:** `build_decision_tree_pipeline()`: `max_depth=6`, `min_samples_leaf=20`, `class_weight="balanced"`,
  `random_state=42`. The model is a single tree, so it's easy to read as rules.
- **Logged CV result:** ROC-AUC 0.708 ± 0.004, PR-AUC 0.181 ± 0.004, recall 0.607, precision 0.152, F1 0.242.
- **Reference only, test set:** ROC-AUC 0.706, PR-AUC 0.173, recall 0.605, precision 0.147.
- **Status: unverified.** The notebook's Finding describes 5-fold CV, but no cross-validation code exists in the
  repo. The logged row's source is unknown. The notebook also loads `X_train.joblib` and similar files, which
  aren't in `data/processed/`, so it can't run as-is.
- **Why it's behind:** a single tree with leaves of at least 20 gives coarse predictions. Unlimited depth overfits.

### 3.3 Random Forest (ours)

- **Builder:** `build_random_forest_pipeline()`: 300 trees, `class_weight="balanced"`.
- **Baseline notebook:** `05_baseline_models_random_forest.ipynb`.
  - CV result: ROC-AUC 0.774 ± 0.004, PR-AUC 0.358 ± 0.009, recall 0.163, precision 0.632, F1 0.260.
  - Figure: `experiments/figures/05_random_forest_roc_pr.png`.
  - The low recall at 0.5 is a property of the operating point, not of the ranking.
- **Tuning notebook:** `07b_random_forest_tuning.ipynb`.
  - 30 random configurations, stratified 5-fold CV, ROC-AUC. The untuned baseline is re-evaluated under the same
    folds, and the search space includes the baseline values.
  - Tuned (500 trees, depth 20, leaf size 1, `sqrt` features, balanced): ROC-AUC 0.770 ± 0.004, PR-AUC 0.339,
    recall 0.285, precision 0.381.
  - **Decision:** keep the untuned baseline. The ROC-AUC gap is -0.004, within one standard deviation.
  - **Caveat:** the exact baseline setting wasn't among the 30 samples, so this is not an exhaustive search.
  - **Superseded:** an earlier run chose an operating threshold of 0.309. That gained recall, but the gain came
    from the threshold, not the tuned parameters, and the baseline was never scored at that threshold.
- **Imbalance comparison:** balanced 0.764 > balanced_subsample 0.760 > none 0.753 (best search scores).

### 3.4 Gaussian Naive Bayes (ours, `05b_gaussian_naive_bayes.ipynb`)

- **Builder:** `build_gaussian_naive_bayes_pipeline()`.
- **CV result:** ROC-AUC 0.697 ± 0.004, PR-AUC 0.176 ± 0.009, recall 0.479, precision 0.170, F1 0.250.
- **Why it's weakest:** it assumes features are independent given the class. Our features are strongly
  correlated (`score_mean` and the `Score_Source` columns), so the assumption breaks.
- **Decision:** no tuning. It has one parameter, and tuning wouldn't change the ranking.

### 3.5 XGBoost baseline (teammate, `05c_XGBoost_Baseline.ipynb`)

- **Builder:** `build_xgboost()` in `src/models/xgboost_model.py`, wrapped in `build_pipeline`:
  300 trees, learning rate 0.05, max depth 6, subsample 0.8, column sample 0.8, `logloss`. It has no class weighting.
- **CV result:** ROC-AUC 0.763 ± 0.004, PR-AUC 0.261 ± 0.010, recall 0.022, precision 0.622, F1 0.042.
- **Why recall is so low at 0.5:** no class weighting, so the model rarely predicts default at that cut-off.
  Its probabilities still rank well (ROC-AUC 0.763).
- **Notebook conclusion:** the recorded baseline is the reference point for tuning.

### 3.6 XGBoost tuned (teammate, `07_hyperparameter_tuning.ipynb`)

- **Search:** `RandomizedSearchCV`, 30 configurations, stratified 5-fold CV, ROC-AUC. The search space covers trees,
  learning rate, depth, minimum child weight, subsample, column sample, gamma, and two regularisation terms.
- **Best parameters:** 700 trees, learning rate 0.03, depth 8, subsample 0.7, column sample 0.8, minimum child
  weight 1, gamma 0.5, `reg_alpha` 0.01, `reg_lambda` 2.0.
- **CV result:** ROC-AUC 0.771 ± 0.004, PR-AUC 0.307 ± 0.009, recall 0.045, precision 0.717, F1 0.085.
- **Caveat:** the notebook also reports results on the held-out test set. Those are reference only and weren't used
  to pick the parameters, but the team has not agreed a policy for test-set numbers.
- **Compared with the baseline:** ROC-AUC +0.008, which is about twice the standard deviation, and PR-AUC up from
  0.261 to 0.307. The ROC-AUC gain is real, but the recall at 0.5 is still very low.

### 3.7 LightGBM (teammate, `lightgbm.ipynb`)

- **Builder:** `build_lightgbm_pipeline()` in `src/models/lightgbm_model.py`: 100 trees, learning rate 0.05, max depth
  5, `scale_pos_weight=11.37`, `random_state=42`. A `FeatureNameCleaner` step renames columns, because LightGBM
  rejects colons and spaces.
- **Reference only, test set (from the notebook and docstring):** ROC-AUC 0.748, PR-AUC 0.215, recall 0.648,
  precision 0.168. Training takes about 2 seconds.
- **Status: pending.** There is no cross-validated result, so it can't be compared yet. The notebook also loads
  `X_train.joblib` and similar files that aren't in the repo.
- **Feature names:** the notebook's top features (`Credit_to_Annuity_Ratio`, `Mean_Bureau_Score`, `Age_Days`) don't
  match the column names our pipeline produces (`credit_to_income`, `score_mean`, `Age_Years`). The notebook was
  probably run on an older feature set, which needs checking.

---

## 4. Comparison

`notebooks/06b_model_comparison.ipynb` reads `experiments.csv`, `logistic_regression.csv`, and
`xgboost_baseline_vs_tuned.csv`. It keeps the most recent row per model name, and it doesn't use the test set.

| Rank | Model | ROC-AUC | PR-AUC | Recall (0.5) | Precision (0.5) | Protocol |
|---|---|---|---|---|---|---|
| 1 | Random Forest (untuned) | 0.774 ± 0.004 | 0.358 ± 0.009 | 0.163 | 0.632 | verified |
| 2 | XGBoost Tuned | 0.771 ± 0.004 | 0.307 ± 0.009 | 0.045 | 0.717 | verified |
| 3 | Random Forest (tuned) | 0.770 ± 0.004 | 0.339 ± 0.009 | 0.285 | 0.381 | verified |
| 4 | XGBoost Baseline | 0.763 ± 0.004 | 0.261 ± 0.010 | 0.022 | 0.622 | verified |
| 5 | Logistic Regression | 0.734 ± 0.005 | 0.212 ± 0.005 | 0.668 | 0.153 | verified |
| 6 | Decision Tree | 0.708 ± 0.004 | 0.181 ± 0.004 | 0.607 | 0.152 | **unverified** |
| 7 | Gaussian NB | 0.697 ± 0.004 | 0.176 ± 0.009 | 0.479 | 0.170 | verified |

**Not in the table:** LightGBM, because it has no cross-validated result. Its test-set numbers would mix protocols.

**Proposed shortlist (team to confirm):** Random Forest (untuned) and XGBoost Tuned. They are within one standard
deviation on ROC-AUC, and Random Forest leads clearly on PR-AUC. XGBoost Tuned's recall at 0.5 is very low, so the
operating threshold matters for it.

---

## 5. How to run it

```
pip install -r requirements.txt
pytest -q
```

Run notebooks from `notebooks/`, which finds the project root automatically.

| Notebook | Time |
|---|---|
| `05a_logistic_regression.ipynb` | not timed |
| `05_baseline_models_random_forest.ipynb` | about 3 minutes |
| `05b_gaussian_naive_bayes.ipynb` | about 1 minute |
| `05c_XGBoost_Baseline.ipynb` | not timed |
| `07b_random_forest_tuning.ipynb` | 35–45 minutes |
| `07_hyperparameter_tuning.ipynb` (XGBoost) | not timed (longer than a single baseline) |
| `06b_model_comparison.ipynb` | seconds |

**Test status:** 21 of 22 pass. The failure is `test_lightgbm_pipeline_fits_and_predicts`, because `lightgbm` is not in
`requirements.txt` yet.

---

## 6. Known issues

These are recorded so nobody is surprised. Several are unresolved.

1. **Decision Tree CV source unknown.** The logged row has no cross-validation code in the repo, and the teammate's
   `05_baseline_models.ipynb` that probably produced it is no longer in `notebooks/`.
2. **Decision Tree and LightGBM notebooks can't run as-is.** They load `X_train.joblib` and similar files that aren't
   in `data/processed/`.
3. **LightGBM has no CV result**, and its feature names don't match the current pipeline.
4. **Test-set numbers are reported in several notebooks** (Logistic Regression, Decision Tree, LightGBM, XGBoost
   tuning). They weren't used to select parameters, but the team hasn't agreed a rule for them.
5. **Duplicate rows in `experiments.csv`**: the Decision Tree appears twice, and an earlier Random Forest row was
   duplicated. `06b` deduplicates on read; the file itself isn't cleaned.
6. **`lightgbm` missing from `requirements.txt`.** One test fails until it's added.
7. **Teammate's `06_model_comparison.ipynb` is broken.** It raises `NameError: X_train`, covers XGBoost only, and has a
   hard-coded Mac path. `06b` was built instead, and the teammate's file was not touched.
8. **Naming:** `"XGBoost Baseline"` uses spaces and capitals. Flagged, not renamed.
9. **README is stale.** Its structure section doesn't list the new notebooks or modules.
10. **Backend, frontend, and `models/baseline|tuned|final/`** are still empty.

---

## 7. Viva guide: the whole story

Follow the story, not the file order: **protocol → pipeline → each model → what tuning changed → comparison →
what we keep, and what's still open.** For each model, spend about one minute: the builder, the CV result, and one
sentence on why it lands where it does.

### Step 1: the shared protocol (1 min)
Open [src/evaluation/metrics.py](src/evaluation/metrics.py). Point at `StratifiedKFold(shuffle=True, random_state=RANDOM_STATE)`
and `clone(pipeline)`.

> "Every model is scored by the same function, with the same folds, metric, and threshold. That's what makes the
> comparison fair."

### Step 2: the pipeline (1 min)
Open [src/preprocessing/feature_engineering.py](src/preprocessing/feature_engineering.py), `build_pipeline`.

> "Imputation, encoding, and feature selection learn from training data only, and the model is the last step. So one
> fitted object can score one new application."

### Step 3: the models (about 8 minutes)
Logistic Regression, Decision Tree, Random Forest, Gaussian NB, XGBoost baseline, XGBoost tuned, LightGBM. Use the
sections in this file. For LightGBM and the Decision Tree, say the status plainly.

### Step 4: tuning (2 min)
Random Forest: tuned didn't beat the baseline, so we kept the baseline. XGBoost: tuning gained about 0.008 ROC-AUC,
which is real, and it raised PR-AUC from 0.261 to 0.307.

### Step 5: the comparison (2 min)
Open `06b_model_comparison.ipynb`.

> "Random Forest ranks first on ROC-AUC and PR-AUC. XGBoost Tuned is close on ROC-AUC. We propose carrying both
> forward. The Decision Tree and LightGBM are pending verification."

---

## 8. Likely questions

**Why do all the models use the same protocol?**
So the comparison is fair. Otherwise a difference could come from the split, not the model.

**Why 5-fold CV and not the test set?**
The test set must stay unseen until the final model is chosen. Comparisons use CV on the training data only.

**Why is accuracy not reported?**
Predicting "no default" for everyone gives about 92% accuracy, so it doesn't measure anything useful.

**Why is Logistic Regression behind?**
It can't draw curves or interactions, so it underfits, even with balanced weights.

**Why does the Decision Tree have a question mark?**
Its logged numbers have no traceable cross-validation code, and its notebook can't run as-is. We'll confirm the source
or re-run it.

**Why is LightGBM not in the comparison?**
It has no cross-validated result yet, so adding its test-set numbers would mix protocols.

**Why did XGBoost tuning help but Random Forest tuning didn't?**
In our runs, XGBoost tuning improved ROC-AUC by about twice its standard deviation, while Random Forest's best sampled
configuration fell slightly below its untuned baseline. We haven't tested why. One possible reason is that the
XGBoost baseline used untuned settings, but that's a hypothesis, not a finding.

**Why does XGBoost have such low recall at 0.5?**
It has no class weighting, and its probabilities sit below 0.5 for most defaulters. Its ranking is still good.

**Why not tune Naive Bayes?**
It has one parameter and is clearly the weakest model, so tuning wouldn't change the outcome.

**Is XGBoost Tuned really close to Random Forest?**
On ROC-AUC, yes: 0.771 against 0.774, within one standard deviation. On PR-AUC, no: 0.307 against 0.358.

**What threshold will the final model use?**
Not chosen yet. Earlier, a threshold of 0.309 gave recall 0.65 but about 22,000 extra false alarms, and that gain came
from the threshold, not the tuning. The threshold is a team decision applied to all models.

---

## 9. Before you present: checklist

- [ ] Add `lightgbm` to `requirements.txt`, then `pip install -r requirements.txt`
- [ ] `pytest -q`: expect 22 of 22
- [ ] Open, in this order: `src/evaluation/metrics.py`, `src/preprocessing/feature_engineering.py`, then one notebook per model
- [ ] Notebooks already show their outputs. Don't re-run the tuning notebooks during the viva
- [ ] Know the Decision Tree and LightGBM status before saying anything about them
- [ ] Know the test-set reporting issue (section 6, item 4) in case it's raised
