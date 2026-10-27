import os
import json
import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


def compute_all_metrics(actual: np.ndarray, predicted: np.ndarray, threshold: float = 0.5) -> dict:
    """Computes continuous regression and thresholded classification metrics."""
    mse = float(mean_squared_error(actual, predicted))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(actual, predicted))

    actual_sum = np.sum(actual)
    wmape = float((np.sum(np.abs(actual - predicted)) / actual_sum * 100)) if actual_sum > 0 else 0.0

    # Classification: Demand day (>0.5) vs Zero-sales day
    actual_bin = (actual > threshold).astype(int)
    predicted_bin = (predicted > threshold).astype(int)

    acc = float(accuracy_score(actual_bin, predicted_bin) * 100)
    prec = float(precision_score(actual_bin, predicted_bin, zero_division=0) * 100)
    rec = float(recall_score(actual_bin, predicted_bin, zero_division=0) * 100)
    f1 = float(f1_score(actual_bin, predicted_bin, zero_division=0) * 100)

    # Asymmetric business cost
    diff = actual - predicted
    stockout = np.sum(diff[diff > 0]) * 4.0
    holding = np.sum(np.abs(diff[diff < 0])) * 1.0
    cost = float(stockout + holding)

    return {
        "regression": {
            "MSE": round(mse, 4),
            "RMSE": round(rmse, 4),
            "MAE": round(mae, 4),
            "WMAPE_pct": round(wmape, 2)
        },
        "classification": {
            "Accuracy_pct": round(acc, 2),
            "Precision_pct": round(prec, 2),
            "Recall_pct": round(rec, 2),
            "F1_Score_pct": round(f1, 2)
        },
        "business": {
            "Inventory_Loss_USD": round(cost, 2)
        }
    }


def save_evaluation_report(metrics: dict, output_path: str = "models/evaluation_report.json"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"\n[Report Saved] -> {output_path}")