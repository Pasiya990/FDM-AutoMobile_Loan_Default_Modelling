"""Stratified train/test split, done once, before any step that learns a
statistic from the data (imputation values, category lists, scalers).
Everyone reads train.csv/test.csv from data/processed/ - nobody re-splits.
"""

import os

import pandas as pd
from sklearn.model_selection import train_test_split

from . import config


def load_clean_rows(path=None):
    path = path or os.path.join(config.PROCESSED_DATA_DIR, "clean_rows.csv")
    return pd.read_csv(path, low_memory=False)


def split_data(df=None, output_dir=None, save=True):
    if df is None:
        df = load_clean_rows()

    train_df, test_df = train_test_split(
        df,
        test_size=config.TRAIN_TEST_SPLIT_RATIO,
        stratify=df[config.TARGET],
        random_state=config.RANDOM_STATE,
    )

    report = pd.DataFrame(
        [
            {
                "split": "train",
                "rows": len(train_df),
                "default_rate": round(train_df[config.TARGET].mean(), 4),
            },
            {
                "split": "test",
                "rows": len(test_df),
                "default_rate": round(test_df[config.TARGET].mean(), 4),
            },
        ]
    )

    if save:
        output_dir = output_dir or config.PROCESSED_DATA_DIR
        os.makedirs(output_dir, exist_ok=True)
        train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
        test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)
        report.to_csv(os.path.join(output_dir, "split_report.csv"), index=False)

    print(report.to_string(index=False))

    return train_df, test_df


if __name__ == "__main__":
    split_data()
