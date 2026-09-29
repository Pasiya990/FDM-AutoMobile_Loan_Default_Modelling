"""Decision Tree Baseline Model
Automobile Loan Default Prediction (SLIIT IT3051)
"""

import os
import time
import joblib
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


def train_decision_tree(X_train, y_train, max_depth=6, min_samples_leaf=20):
    """Initializes and trains a Decision Tree model.

    Parameters:
    - max_depth=6: Stops the tree from growing too deep (prevents overfitting).
    - min_samples_leaf=20: Requires at least 20 applicants per leaf.
    - class_weight='balanced': Gives more weight to defaulters (handles imbalanced data).
    - random_state=42: Makes results reproducible.
    """
    # 1. Create the Decision Tree model
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight="balanced",
        random_state=42,
    )

    # 2. Record start time and fit the model on training data
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time  # Training time in seconds

    return model, train_time
