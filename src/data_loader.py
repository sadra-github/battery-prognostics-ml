"""Basic loading and validation for cycle-level battery tables."""
from pathlib import Path
from typing import Iterable

import pandas as pd


def load_cycle_csv(path: str | Path) -> pd.DataFrame:
    """Load a CSV file and reject empty tables.

    Column interpretation is deliberately left to the caller because battery
    datasets use different names and units.
    """
    file_path = Path(path).expanduser()
    if not file_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {file_path}")
    frame = pd.read_csv(file_path)
    if frame.empty:
        raise ValueError(f"CSV contains no rows: {file_path}")
    if frame.columns.duplicated().any():
        duplicates = frame.columns[frame.columns.duplicated()].tolist()
        raise ValueError(f"Duplicate column names: {duplicates}")
    return frame


def validate_columns(frame: pd.DataFrame, required: Iterable[str]) -> None:
    """Raise a clear error if required columns are missing."""
    required = list(required)
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def summarize_missingness(frame: pd.DataFrame) -> pd.DataFrame:
    """Return per-column missing counts and percentages."""
    counts = frame.isna().sum()
    summary = pd.DataFrame({"missing_count": counts,
                            "missing_percent": counts / max(len(frame), 1) * 100})
    return summary.sort_values("missing_count", ascending=False)
