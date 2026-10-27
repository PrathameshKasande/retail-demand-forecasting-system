import os
import pandas as pd
import numpy as np
import yaml


def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def clip_outliers_iqr(df: pd.DataFrame) -> pd.DataFrame:
    """Clips daily sales outliers per SKU using IQR to prevent distribution skew."""
    clipped_groups = []
    for _, group in df.groupby("StockCode"):
        g = group.copy()
        q25 = g["sales"].quantile(0.25)
        q75 = g["sales"].quantile(0.75)
        iqr = q75 - q25
        upper_limit = q75 + 2.5 * iqr
        g["sales"] = np.clip(g["sales"], 0, upper_limit)
        clipped_groups.append(g)
    return pd.concat(clipped_groups).sort_values(by=["StockCode", "date"]).reset_index(drop=True)


def load_and_aggregate_data(config_path="config/config.yaml") -> pd.DataFrame:
    config = load_config(config_path)
    raw_path = config["data"]["raw_path"]

    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Missing {raw_path}. Place online_retail_II.csv in data/raw/")

    print(f"Reading dataset: {raw_path}...")
    df = pd.read_csv(
        raw_path,
        usecols=["Invoice", "StockCode", "Quantity", "InvoiceDate", "Price"]
    )

    # 1. Clean invalid cancellations
    df = df[~df["Invoice"].astype(str).str.startswith("C")]
    df = df[(df["Quantity"] > 0) & (df["Price"] > 0)].copy()

    # 2. Level-1 Outlier filter: Cap extreme single wholesale anomaly orders
    df = df[df["Quantity"] <= 200].copy()

    df["date"] = pd.to_datetime(df["InvoiceDate"]).dt.normalize()

    # 3. Filter top SKUs by repeat customer invoice frequency
    top_skus = df.groupby("StockCode")["Invoice"].nunique().nlargest(config["data"]["top_n_skus"]).index
    df = df[df["StockCode"].isin(top_skus)]

    # 4. Aggregate daily sales and price
    daily = (
        df.groupby(["StockCode", "date"])
        .agg(sales=("Quantity", "sum"), avg_price=("Price", "mean"))
        .reset_index()
    )

    # 5. Continuous daily grid
    all_skus = daily["StockCode"].unique()
    all_dates = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
    full_idx = pd.MultiIndex.from_product([all_skus, all_dates], names=["StockCode", "date"]).to_frame().reset_index(drop=True)

    daily_complete = pd.merge(full_idx, daily, on=["StockCode", "date"], how="left")
    daily_complete["sales"] = daily_complete["sales"].fillna(0).astype(np.int32)
    daily_complete["avg_price"] = daily_complete.groupby("StockCode")["avg_price"].transform(lambda x: x.ffill().bfill())
    daily_complete = daily_complete.sort_values(by=["StockCode", "date"]).reset_index(drop=True)

    # 6. Level-2 Outlier filter: 100% IQR clipping on aggregated daily volume
    daily_clean = clip_outliers_iqr(daily_complete)

    os.makedirs(os.path.dirname(config["data"]["processed_path"]), exist_ok=True)
    daily_clean.to_parquet(config["data"]["processed_path"], index=False)

    os.makedirs(os.path.dirname(config["data"]["sample_path"]), exist_ok=True)
    daily_clean.head(100).to_csv(config["data"]["sample_path"], index=False)

    print(f"Aggregation complete. Processed {len(daily_clean)} outlier-free records.")
    return daily_clean


if __name__ == "__main__":
    load_and_aggregate_data()