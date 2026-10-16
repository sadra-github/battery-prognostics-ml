"""Regression metrics with explicit checks for invalid inputs."""
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """Compute MAE, RMSE, and R-squared for aligned numeric arrays."""
    true = np.asarray(y_true, dtype=float).reshape(-1)
    pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if true.size == 0:
        raise ValueError("Metric inputs must not be empty")
    if true.shape != pred.shape:
        raise ValueError("y_true and y_pred must have the same number of values")
    if not np.isfinite(true).all() or not np.isfinite(pred).all():
        raise ValueError("Metric inputs must contain only finite values")
    return {
        "mae": float(mean_absolute_error(true, pred)),
        "rmse": float(np.sqrt(mean_squared_error(true, pred))),
        "r2": float(r2_score(true, pred)) if true.size >= 2 else float("nan"),
    }
