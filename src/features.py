"""Small, dataset-agnostic feature helpers for battery-cycle tables."""
import numpy as np
import pandas as pd


def extract_degradation_trend(values, window: int = 5) -> pd.Series:
    """Return a centered rolling mean for an ordered degradation signal.

    The input must already be sorted by cycle. The rolling window is kept
    modest by default; choose it using the sampling density and protocol.
    """
    if window < 1:
        raise ValueError("window must be a positive integer")
    series = pd.Series(values, dtype="float64")
    return series.rolling(window=window, min_periods=1, center=True).mean()


def extract_impedance_features(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Summarize selected impedance columns with basic descriptive features.

    Returns one row of mean, standard deviation, minimum, and maximum per
    requested column. It does not infer equivalent-circuit parameters.
    """
    if not columns:
        raise ValueError("Provide at least one impedance column")
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Impedance columns not found: {missing}")
    numeric = frame[columns].apply(pd.to_numeric, errors="coerce")
    out = {}
    for column in columns:
        values = numeric[column].dropna()
        if values.empty:
            out[f"{column}_mean"] = np.nan
            out[f"{column}_std"] = np.nan
            out[f"{column}_min"] = np.nan
            out[f"{column}_max"] = np.nan
        else:
            out[f"{column}_mean"] = float(values.mean())
            out[f"{column}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            out[f"{column}_min"] = float(values.min())
            out[f"{column}_max"] = float(values.max())
    return pd.DataFrame([out])
