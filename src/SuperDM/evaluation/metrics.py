from __future__ import annotations

import numpy as np


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_true - y_pred)))

    per_dim_mse = np.mean((y_true - y_pred) ** 2, axis=0).tolist()

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "per_dimension_mse": per_dim_mse,
    }