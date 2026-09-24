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

**Problem Type:** [Classification / Regression]

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

- **Frontend:** [Technology]
- **Backend:** [Technology]
- **Machine Learning Model:** [Final Model]

Users can enter the required input values and receive a prediction from the trained model.

## 📁 Project Structure

```text
FDM-AutoMobile_Loan_Default_Modelling/
│
├── data/
│   ├── raw/
│   │   └── .gitkeep
│   └── processed/
│       └── .gitkeep
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_data_preprocessing.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_baseline_models.ipynb
│   ├── 06_model_comparison.ipynb
│   └── 07_hyperparameter_tuning.ipynb
│
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   └── data_cleaning.py
│   │
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── preprocessing.py
│   │   └── feature_engineering.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── logistic_regression.py
│   │   ├── decision_tree.py
│   │   ├── random_forest.py
│   │   └── xgboost_model.py
│   │
│   └── evaluation/
│       ├── __init__.py
│       ├── metrics.py
│       └── model_comparison.py
│
├── models/
│   ├── baseline/
│   ├── tuned/
│   └── final/
│
├── experiments/
│   ├── results/
│   └── figures/
│
├── backend/
│   ├── app.py
│   └── routes/
│       └── prediction.py
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
│
├── reports/
│   ├── dataset_proposal/
│   ├── technical_report/
│   └── presentation/
│
├── tests/
│
├── requirements.txt
├── .gitignore
└── README.md

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
