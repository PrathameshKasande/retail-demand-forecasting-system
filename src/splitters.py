import pandas as pd


def time_based_train_val_split(df: pd.DataFrame, val_days: int = 14):
    """Splits dataset strictly on a chronological date boundary."""
    max_date = df["date"].max()
    split_date = max_date - pd.Timedelta(days=val_days)

    train_df = df[df["date"] <= split_date].copy()
    val_df = df[df["date"] > split_date].copy()
    return train_df, val_df