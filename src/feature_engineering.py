import pandas as pd
import numpy as np
import yaml


def generate_temporal_features(df: pd.DataFrame, config_path="config/config.yaml") -> pd.DataFrame:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    horizon = config["features"]["forecast_horizon"]
    lags = config["features"]["lag_days"]
    windows = config["features"]["rolling_windows"]

    df = df.sort_values(by=["StockCode", "date"]).copy()

    # Shifted lag variables
    for lag in lags:
        df[f"sales_lag_{lag}"] = df.groupby("StockCode")["sales"].shift(lag)

    # Shifted rolling mean and standard deviation
    for w in windows:
        df[f"sales_roll_mean_{w}"] = df.groupby("StockCode")["sales"].transform(
            lambda x: x.shift(horizon).rolling(w).mean()
        )
        df[f"sales_roll_std_{w}"] = df.groupby("StockCode")["sales"].transform(
            lambda x: x.shift(horizon).rolling(w).std()
        )
        # Shifted zero-demand frequency to track inactive spells
        df[f"zero_ratio_{w}"] = df.groupby("StockCode")["sales"].transform(
            lambda x: (x.shift(horizon) == 0).rolling(w).mean()
        )

    # Calendar features
    df["day_of_week"] = df["date"].dt.day_of_week.astype(np.int8)
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(np.int8)
    df["month"] = df["date"].dt.month.astype(np.int8)
    df["day_of_month"] = df["date"].dt.day.astype(np.int8)

    return df.dropna().reset_index(drop=True)