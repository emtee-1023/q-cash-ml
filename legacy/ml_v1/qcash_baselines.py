"""Naive forecast baselines for the Q-Cash revenue model.

These live in their own importable module rather than in the training script
because a joblib artifact stores the class by module path. If the class were
defined in the training script, loading the artifact in a separate prediction
process would fail with "module '__main__' has no attribute ...".

The empirical finding behind this module: on the Belgard Barbers data, no
machine-learning model beat a simple persistence forecast, so the production
artifact is one of these objects. They are still real estimators with a
`predict` method, so the prediction service treats them like any scikit-learn
model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class PersistenceBaseline:
    """Forecast next-30-day revenue from trailing revenue.

    Parameters
    ----------
    column:
        Name of the trailing-revenue column to read, for example `revenue_30d`.
    multiplier:
        Scale applied to that column. A 60-day total covers twice the window,
        so `revenue_60d` uses a multiplier of 0.5 to compare like with like.
    """

    def __init__(self, column: str, multiplier: float = 1.0) -> None:
        self.column = column
        self.multiplier = multiplier

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Return the forecast, one value per row of `frame`."""
        if isinstance(frame, np.ndarray):
            raise TypeError(
                f"{type(self).__name__} needs a DataFrame with column "
                f"{self.column!r}, not a numpy array."
            )
        if self.column not in frame.columns:
            raise KeyError(
                f"Input frame is missing required column {self.column!r}. "
                f"Available columns: {sorted(frame.columns)}"
            )
        values = frame[self.column].to_numpy(dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError(
                f"Column {self.column!r} contains non-finite values. Impute or "
                "drop those rows before predicting."
            )
        return self.multiplier * values

    def __repr__(self) -> str:
        return f"PersistenceBaseline({self.column!r}, multiplier={self.multiplier})"
