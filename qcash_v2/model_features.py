"""Shared model-feature definitions for training and serving.

Keeping these in one place guarantees that the engine computes exactly the same
columns the model was trained on. The enhanced frame adds the 90-day moving
average and scale-free momentum features that let a tree model anchor on the
moving average (ratio target) instead of extrapolating a raw level.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLS = [
    "rev_sum_7", "rev_sum_30", "rev_sum_90", "txn_count_30",
    "freq_per_active_week", "active_days_30", "adr_pct", "mean_txn_minor",
    "cv_daily_rev_30", "trend_pct", "hhi_30", "top1_share_30", "tenure_days",
]
MA_DIVISOR = 3.0


def feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    d = df[FEATURE_COLS].copy()
    sector_dummies = pd.get_dummies(df["sector"], prefix="sector", dtype=float)
    return pd.concat([d, sector_dummies], axis=1)


def enhanced_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Base features + 90-day moving average and scale-free momentum features."""
    base = feature_frame(df)
    rev7 = df["rev_sum_7"].to_numpy(dtype=float)
    rev30 = df["rev_sum_30"].to_numpy(dtype=float)
    rev90 = df["rev_sum_90"].to_numpy(dtype=float)
    ma = rev90 / MA_DIVISOR
    eps = 1.0
    out = base.copy()
    out["ma_90"] = ma
    out["mom_week"] = rev7 / (rev30 * 7.0 / 30.0 + eps)
    out["mom_month"] = rev30 / (ma + eps)
    out["mom_week_q"] = rev7 / (ma * 7.0 / 30.0 + eps)
    out["month"] = df["as_of_date"].dt.month.astype(float).to_numpy()
    out["quarter"] = df["as_of_date"].dt.quarter.astype(float).to_numpy()
    return out
