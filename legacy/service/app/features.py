"""Feature engineering for the Q-Cash synthetic revenue model.

Computes the same 90-day historical features used to train the served model,
from a business's trailing daily revenue and transaction counts. The logic
mirrors ``qcash_synthetic_v1/scripts/build_supervised_dataset.py`` so that
served predictions are computed identically to the training features.

Every feature is computed strictly from days *before* the information cutoff
date (``as_of_date`` = the day after the latest input day), so the forecast
uses no future information.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

WINDOWS = (7, 14, 30, 60, 90)
MIN_HISTORY_DAYS = 90


def _trailing(df: pd.DataFrame, as_of: pd.Timestamp, days: int) -> pd.DataFrame:
    lo = as_of - pd.Timedelta(days=days)
    return df[(df["transaction_date"] >= lo) & (df["transaction_date"] < as_of)]


def _weekend_share(df: pd.DataFrame) -> float:
    if len(df) == 0 or df["revenue_kes"].sum() == 0:
        return 0.0
    weekend = df["transaction_date"].dt.weekday >= 5
    return float(df.loc[weekend, "revenue_kes"].sum() / df["revenue_kes"].sum())


def _cv(values: pd.Series) -> float:
    mean = values.mean()
    if not np.isfinite(mean) or mean <= 0:
        return 0.0
    return float(values.std() / mean)


def compute_features(daily: pd.DataFrame) -> dict[str, float]:
    """Return the served model's feature row for one business.

    ``daily`` must contain at least 90 days of ``transaction_date``,
    ``revenue_kes`` and ``transaction_count``. The latest date is treated as
    "today"; features are computed up to and including it, and the 30-day
    forecast window follows.
    """
    required = {"transaction_date", "revenue_kes", "transaction_count"}
    missing = required.difference(daily.columns)
    if missing:
        raise ValueError(f"Input is missing required columns: {sorted(missing)}")

    df = daily.copy()
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    df = df.sort_values("transaction_date").reset_index(drop=True)

    if len(df) < MIN_HISTORY_DAYS:
        raise ValueError(
            f"At least {MIN_HISTORY_DAYS} days of history are required, got {len(df)}."
        )

    as_of = df["transaction_date"].max() + pd.Timedelta(days=1)

    row: dict[str, float] = {}

    # revenue / transaction-count rolling aggregates
    for w in WINDOWS:
        win = _trailing(df, as_of, w)
        row[f"revenue_{w}d"] = float(win["revenue_kes"].sum())
        row[f"tx_count_{w}d"] = float(win["transaction_count"].sum())

    w30 = _trailing(df, as_of, 30)
    w60 = _trailing(df, as_of, 60)
    w90 = _trailing(df, as_of, 90)

    rev30_daily = w30["revenue_kes"]
    active30 = (w30["transaction_count"] > 0)
    tx30 = w30["transaction_count"].sum()
    rev30 = w30["revenue_kes"].sum()

    row["active_days_30d"] = float(active30.sum())
    row["avg_daily_revenue_30d"] = float(rev30_daily.mean()) if len(rev30_daily) else 0.0
    row["median_daily_revenue_30d"] = float(rev30_daily.median()) if len(rev30_daily) else 0.0
    row["std_daily_revenue_30d"] = float(rev30_daily.std()) if len(rev30_daily) else 0.0
    row["cv_daily_revenue_30d"] = _cv(rev30_daily)
    row["zero_revenue_day_rate_30d"] = float(1.0 - active30.mean()) if len(w30) else 0.0
    row["weekend_revenue_share_30d"] = _weekend_share(w30)
    row["average_transaction_value_30d"] = float(rev30 / tx30) if tx30 > 0 else 0.0
    row["transaction_frequency_30d"] = float(tx30 / active30.sum()) if active30.sum() > 0 else 0.0

    rev60 = w60["revenue_kes"].sum()
    rev90 = w90["revenue_kes"].sum()
    avg30 = rev30 / 30.0 if rev30 > 0 else 0.0
    avg90 = rev90 / 90.0 if rev90 > 0 else 0.0
    row["revenue_trend_30_vs_90"] = float(avg30 / avg90 - 1.0) if avg90 > 0 else 0.0
    row["recent_growth_rate_30d"] = float(rev30 / rev60 - 1.0) if rev60 > 0 else 0.0
    row["recent_decline_rate_30d"] = float(max(0.0, -(rev30 / rev60 - 1.0))) if rev60 > 0 else 0.0

    row["day_of_week"] = float(as_of.weekday())
    row["month_num"] = float(as_of.month)
    row["quarter"] = float(as_of.quarter)

    return row
