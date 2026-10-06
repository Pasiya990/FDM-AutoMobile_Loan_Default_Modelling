# FDM-AutoMobile_Loan_Default_Modelling

End-to-end Data Mining &amp; Machine Learning project for IT3051, covering EDA, data preprocessing, feature engineering, model development, evaluation, hyperparameter tuning, and an integrated prediction system.

# IT3051 – Fundamentals of Data Mining Mini Project 2026

## 📌 Project Overview

This project is developed as part of the IT3051 – Fundamentals of Data Mining module. It applies an end-to-end data mining and machine learning workflow to solve a real-world prediction problem.

## 🎯 Objectives

- Understand and define the real-world problem
- Explore and understand the selected dataset
- Perform Exploratory Data Analysis (EDA)
- Clean and preprocess the data
- Perform feature engineering and feature selection
- Develop and compare multiple machine learning models
- Optimize the selected models using hyperparameter tuning
- Evaluate and select the final model
- Develop an end-to-end prediction system

## 📊 Dataset

**Dataset:** Automobile Loan Default Dataset

**Source:** kaggle - https://www.kaggle.com/datasets/saurabhbagchi/dish-network-hackathon/data?select=Train_Dataset.csv

**Target Variable:** Default

**Problem Type:** Classification

## 🔍 Exploratory Data Analysis

The project includes analysis of:

- Data structure and variable types
- Missing values
- Duplicate records
- Outliers
- Feature distributions
- Relationships between variables
- Class imbalance
- Potential data leakage

## 🛠️ Data Preprocessing

The preprocessing pipeline includes appropriate techniques such as:

- Handling missing values
- Removing/handling duplicates
- Treating outliers
- Encoding categorical variables
- Feature scaling
- Feature engineering
- Feature selection
- Train-test splitting

## 🤖 Machine Learning Models

At least four suitable machine learning algorithms are implemented and compared.

The models are evaluated using appropriate performance metrics and validation strategies.

## ⚙️ Model Optimization

Hyperparameter tuning and other optimization techniques are applied to improve model performance.

The final model is selected based on experimental results and the requirements of the problem.

## 🌐 Prediction System

The final trained model is integrated into a prediction system consisting of:

- **Frontend:** React (JavaScript, built with Vite), served by the backend
- **Backend:** FastAPI (Python)
- **Machine Learning Model:** tuned XGBoost inside the shared preprocessing pipeline
  (`models/final/final_pipeline.joblib`, details in `models/final/metadata.json`)

A loan officer enters an application and receives a **risk score**, a **risk band** (Low / Medium / High) and a
**suggested action**. The system supports decisions; it never refuses an application by itself.

| | Final model (held-out test set, used once) |
|---|---|
| ROC-AUC | 0.760 |
| PR-AUC | 0.247 |
| Operating threshold | 0.522 (chosen for 60% recall on training data) |
| Recall / precision at the threshold | 0.627 / 0.187 |
| Applicants flagged | 27.2% |

Risk bands (chosen on out-of-fold training predictions): **Low** below 0.25 (about 2% default), **Medium** 0.25 to
0.52 (about 6%), **High** from 0.52, the flagged applicants (about 18%). The risk score ranks applicants; it is not a
probability, because the model weights the default class.

## ▶️ Running the System

You need **Python 3.11 or later**. Node.js is only needed to change the web page.

```bash
# 1. Install the Python packages (from the project root)
pip install -r requirements.txt

# 2. Start the API, which also serves the web page
uvicorn backend.app:app --reload
#    (if "uvicorn" is not found: python -m uvicorn backend.app:app --reload)
```

Then open:

- **http://127.0.0.1:8000/** for the web page. "Try an example" fills the form with a real low-, medium- or high-risk
  applicant.
- **http://127.0.0.1:8000/docs** for the interactive API documentation.

Keep the terminal open while using the system; `Ctrl+C` stops it.

| Endpoint | Purpose |
|---|---|
| `GET /health` | Model status and version |
| `GET /schema` | Form fields, limits and allowed values |
| `GET /examples` | Three demo applicants |
| `POST /predict` | Scores one application; `422` with per-field details for invalid input, `503` if the model is missing |

### Tests

```bash
pytest -q
```

Most tests need the dataset at `data/raw/Train_Dataset.csv` (it is not stored in the repository; download
`Train_Dataset.csv` from the Kaggle link above).

### Changing the web page

```bash
cd frontend
npm install
npm run dev      # live-reloading page at http://localhost:5173, using the API started above
npm run build    # writes frontend/dist/, which the API serves
```

`frontend/dist/` is kept in git so the system runs with Python only: rebuild and commit it after changing the page.

### Rebuilding the final model (only if needed)

Needs the dataset in `data/raw/`.

```bash
python -m src.models.select_final        # cross-validation, threshold and selection.json (training data only)
python -m src.models.train_final --force # fits the final model and scores the held-out test set
python -m backend.examples               # refreshes the demo applicants
```

`train_final` refuses to run again without `--force`, because each run uses the held-out test set.

## 📁 Project Structure

```text
FDM-AutoMobile_Loan_Default_Modelling/
│
├── data/
│   ├── raw/                     # Train_Dataset.csv (not in git)
│   └── processed/
│
├── notebooks/                   # 01-04 data understanding to feature engineering,
│                                # 05* baselines, 06b comparison, 07* tuning
│
├── src/
│   ├── config.py                # shared constants (seed, columns, thresholds)
│   ├── data/                    # cleaning, near-duplicate removal, train/test split
│   ├── preprocessing/           # fit-on-train pipeline steps, build_pipeline()
│   ├── models/                  # one builder per model; select_final.py, train_final.py,
│   │                            # risk_bands.py, model_metadata.py
│   └── evaluation/              # shared cross-validation harness and logging
│
├── models/final/                # final_pipeline.joblib + metadata.json
│
├── experiments/
│   ├── results/                 # experiment tables, selection.json, out-of-fold predictions
│   └── figures/
│
├── backend/
│   ├── app.py                   # FastAPI app, serves the API and the web page
│   ├── routes/prediction.py     # /health, /schema, /examples, /predict
│   ├── model_store.py           # loads the model and metadata
│   ├── adapter.py               # form fields -> the model's columns
│   ├── validation.py            # checks an application against the saved schema
│   ├── messages.py              # labels, suggested actions, disclaimer
│   └── examples.py              # demo applicants (examples.json)
│
├── frontend/
│   ├── src/                     # React components
│   └── dist/                    # built page served by the backend
│
├── reports/
├── tests/
├── requirements.txt
└── README.md
```

## Understanding the Structure

* data/ → Where the data comes from.
* notebooks/ → Where you investigate and experiment.
* src/ → Where you keep reusable project code.
* models/ → Where trained models are stored.
* experiments/ → Where you keep evidence of your experiments and results.
* backend/ → Makes the ML model usable as a service.
* frontend/ → Lets a user interact with the system.
* reports/ → Contains the academic deliverables.
* tests/ → Checks that different parts of the system work correctly.

## How to understand the structure

The easiest way to think about it is as **one pipeline**:

```text
                    DATA
                     │
                     ▼
              ┌─────────────┐
              │   notebooks │
              └──────┬──────┘
                     │
        ┌────────────┴────────────┐
        ▼                         ▼
   Data Cleaning            EDA / Analysis
        │                         │
        └────────────┬────────────┘
                     ▼
             Preprocessing
                     │
                     ▼
          Feature Engineering
                     │
                     ▼
               ML Models
                     │
       ┌─────────────┼─────────────┐
       ▼             ▼             ▼
   Logistic       Random        XGBoost
  Regression      Forest
       │             │             │
       └─────────────┼─────────────┘
                     ▼
             Model Evaluation
                     │
                     ▼
            Hyperparameter Tuning
                     │
                     ▼
                Final Model
                     │
                     ▼
             ┌──────────────┐
             │    Backend   │
             └──────┬───────┘
                    │
                    ▼
             ┌──────────────┐
             │   Frontend   │
             └──────┬───────┘
                    │
                    ▼
             Loan Default
               Prediction
```
