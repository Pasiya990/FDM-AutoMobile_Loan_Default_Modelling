"""Row-level data preparation: parse text-numeric columns and remove
duplicate rows. Runs once, before the train/test split, since removing
rows can't happen inside a scikit-learn transformer (it would change the
row count) and duplicates must be gone before the split so no copy can
land on both sides.
"""

import os

import pandas as pd

from . import config


def load_raw_data(path=None):
    path = path or config.RAW_DATA_PATH
    return pd.read_csv(path, low_memory=False)


def parse_numeric_text_columns(df):
    """Convert numeric-looking text columns to real numbers; invalid
    entries (stray characters like '$', 'x', '&', '#VALUE!') become
    missing. Returns (df, token_counts): token_counts maps column name to
    how many values were coerced to NaN.
    """
    df = df.copy()
    token_counts = {}
    for col in config.NUMERIC_TEXT_COLS:
        before = df[col].isna().sum()
        df[col] = pd.to_numeric(df[col], errors="coerce")
        after = df[col].isna().sum()
        token_counts[col] = int(after - before)
    return df, token_counts


def remove_duplicate_rows(df):
    """Remove exact duplicate rows, ignoring ID (which is unique per row
    and would otherwise hide every duplicate - see team plan finding F2).
    Returns (df, n_removed).
    """
    cols_no_id = [c for c in df.columns if c != config.ID_COL]
    before = len(df)
    df = df.drop_duplicates(subset=cols_no_id).reset_index(drop=True)
    after = len(df)
    return df, before - after


def prepare_data(input_path=None, output_dir=None, save=True):
    df = load_raw_data(input_path)
    rows_in = len(df)

    df, token_counts = parse_numeric_text_columns(df)
    df, n_duplicates_removed = remove_duplicate_rows(df)

    log_rows = [
        {"metric": "rows_in", "value": rows_in},
        {"metric": "duplicates_removed", "value": n_duplicates_removed},
        {"metric": "rows_out", "value": len(df)},
    ]
    for col, count in token_counts.items():
        log_rows.append({"metric": f"invalid_tokens_{col}", "value": count})
    log = pd.DataFrame(log_rows)

    if save:
        output_dir = output_dir or config.PROCESSED_DATA_DIR
        os.makedirs(output_dir, exist_ok=True)
        df.to_csv(os.path.join(output_dir, "clean_rows.csv"), index=False)
        log.to_csv(os.path.join(output_dir, "cleaning_log.csv"), index=False)

    print(f"Rows in: {rows_in:,}")
    print(f"Duplicate rows removed (ignoring {config.ID_COL}): {n_duplicates_removed:,}")
    print(f"Rows out: {len(df):,}")
    print()
    print("Invalid tokens converted to missing, per column:")
    for col, count in token_counts.items():
        print(f"  {col:30s}: {count}")

    return df


if __name__ == "__main__":
    prepare_data()
