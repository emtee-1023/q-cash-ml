"""Build the supervised synthetic ML dataset from synthetic daily metrics.

For each business and weekly observation date we construct:

  * historical (leakage-safe) features computed strictly from days *before*
    ``as_of_date``;
  * several forecast targets computed strictly from the 30 days *after*
    ``as_of_date``.

Targets
-------
    target_next_30d_revenue_kes        total revenue over the next 30 days
    minimum_future_30d_revenue         smallest daily revenue in the next 30 days
    lower_percentile_future_30d_revenue 10th percentile of next-30-day daily revenue
    future_30d_revenue_volatility      std of daily revenue over the next 30 days

The future-only targets never enter the historical features, so there is no
leakage by construction. Consecutive weekly rows share overlapping 30-day
target windows; the experiment handles this with an embargo gap.

Usage
-----
    python build_supervised_dataset.py \
        --daily ../data/synthetic_daily_metrics.csv \
        --out ../data/synthetic_supervised_weekly.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_OUT_DIR = Path(__file__).resolve().parent.parent / "data"

REVENUE_COL = "revenue_kes"
TX_COL = "transaction_count"
ACTIVE_COL = "active_day"

WINDOWS = (7, 14, 30, 60, 90)
MIN_HISTORY_DAYS = 90
OBSERVATION_STEP_DAYS = 7
FUTURE_DAYS = 30
LOWER_PERCENTILE = 10


def _trailing(df: pd.DataFrame, as_of: pd.Timestamp, days: int) -> pd.DataFrame:
    """Rows strictly before as_of, within the last ``days`` days."""
    lo = as_of - pd.Timedelta(days=days)
    return df[(df["transaction_date"] >= lo) & (df["transaction_date"] < as_of)]


def _future(df: pd.DataFrame, as_of: pd.Timestamp, days: int) -> pd.DataFrame:
    """Rows strictly after as_of, within the next ``days`` days."""
    hi = as_of + pd.Timedelta(days=days)
    return df[(df["transaction_date"] > as_of) & (df["transaction_date"] <= hi)]


def _weekend_share(df: pd.DataFrame) -> float:
    if len(df) == 0 or df[REVENUE_COL].sum() == 0:
        return 0.0
    weekend = df["transaction_date"].dt.weekday >= 5
    return float(df.loc[weekend, REVENUE_COL].sum() / df[REVENUE_COL].sum())


def _cv(values: pd.Series) -> float:
    mean = values.mean()
    if not np.isfinite(mean) or mean <= 0:
        return 0.0
    return float(values.std() / mean)


def build_features_for_business(
    biz_id: str, daily: pd.DataFrame, as_of_dates: pd.DatetimeIndex
) -> pd.DataFrame:
    """Build one row per as_of date for a single business."""
    daily = daily.sort_values("transaction_date").reset_index(drop=True)
    rows: list[dict] = []

    for as_of in as_of_dates:
        past = daily[daily["transaction_date"] < as_of]
        if len(past) == 0:
            continue

        row = {
            "business_id": biz_id,
            "as_of_date": as_of,
            "data_source": "synthetic",
            "archetype": daily["archetype"].iloc[0],
        }

        # --- revenue / transaction rolling aggregates ---------------------
        for w in WINDOWS:
            win = _trailing(daily, as_of, w)
            row[f"revenue_{w}d"] = float(win[REVENUE_COL].sum())
            row[f"tx_count_{w}d"] = float(win[TX_COL].sum())

        w30 = _trailing(daily, as_of, 30)
        w60 = _trailing(daily, as_of, 60)
        w90 = _trailing(daily, as_of, 90)

        # --- behavioural features (30-day window) -------------------------
        rev30_daily = w30[REVENUE_COL]
        row["active_days_30d"] = float(w30[ACTIVE_COL].sum())
        row["avg_daily_revenue_30d"] = float(rev30_daily.mean()) if len(rev30_daily) else 0.0
        row["median_daily_revenue_30d"] = float(rev30_daily.median()) if len(rev30_daily) else 0.0
        row["std_daily_revenue_30d"] = float(rev30_daily.std()) if len(rev30_daily) else 0.0
        row["cv_daily_revenue_30d"] = _cv(rev30_daily)
        row["zero_revenue_day_rate_30d"] = float(
            1.0 - w30[ACTIVE_COL].mean()) if len(w30) else 0.0
        row["weekend_revenue_share_30d"] = _weekend_share(w30)

        tx30 = w30[TX_COL].sum()
        rev30 = w30[REVENUE_COL].sum()
        row["average_transaction_value_30d"] = float(rev30 / tx30) if tx30 > 0 else 0.0
        active30 = w30[ACTIVE_COL].sum()
        row["transaction_frequency_30d"] = float(tx30 / active30) if active30 > 0 else 0.0

        # --- trend / growth / seasonality ----------------------------------
        rev60 = w60[REVENUE_COL].sum()
        rev90 = w90[REVENUE_COL].sum()
        avg30 = rev30 / 30.0 if rev30 > 0 else 0.0
        avg90 = rev90 / 90.0 if rev90 > 0 else 0.0
        row["revenue_trend_30_vs_90"] = float(avg30 / avg90 - 1.0) if avg90 > 0 else 0.0
        row["recent_growth_rate_30d"] = float(rev30 / rev60 - 1.0) if rev60 > 0 else 0.0
        row["recent_decline_rate_30d"] = float(max(0.0, -(rev30 / rev60 - 1.0))) if rev60 > 0 else 0.0

        row["day_of_week"] = int(as_of.weekday())
        row["month_num"] = int(as_of.month)
        row["quarter"] = int(as_of.quarter)

        # --- future-only targets ------------------------------------------
        fut = _future(daily, as_of, FUTURE_DAYS)
        fut_rev = fut[REVENUE_COL]
        row["target_next_30d_revenue_kes"] = float(fut_rev.sum())
        row["minimum_future_30d_revenue"] = float(fut_rev.min()) if len(fut_rev) else 0.0
        row["lower_percentile_future_30d_revenue"] = float(
            np.percentile(fut_rev, LOWER_PERCENTILE)) if len(fut_rev) else 0.0
        row["future_30d_revenue_volatility"] = float(fut_rev.std()) if len(fut_rev) else 0.0

        rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--daily", type=Path,
                        default=DEFAULT_OUT_DIR / "synthetic_daily_metrics.csv")
    parser.add_argument("--out", type=Path,
                        default=DEFAULT_OUT_DIR / "synthetic_supervised_weekly.csv")
    args = parser.parse_args()

    daily = pd.read_csv(args.daily, parse_dates=["transaction_date"])

    # Common weekly grid aligned to Sundays across the population.
    grid = pd.date_range(daily["transaction_date"].min(),
                         daily["transaction_date"].max(), freq="7D")

    frames = []
    for biz_id, sub in daily.groupby("business_id"):
        frame = build_features_for_business(biz_id, sub, grid)
        frames.append(frame)

    supervised = pd.concat(frames, ignore_index=True)

    # Drop early rows that lack the minimum history (before features are stable)
    # and trailing rows that lack a full 30-day future window.
    first_valid = grid[0] + pd.Timedelta(days=MIN_HISTORY_DAYS)
    last_valid = grid[-1] - pd.Timedelta(days=FUTURE_DAYS)
    supervised = supervised[
        (supervised["as_of_date"] >= first_valid)
        & (supervised["as_of_date"] <= last_valid)
    ].reset_index(drop=True)

    supervised = supervised.sort_values(["business_id", "as_of_date"]).reset_index(drop=True)
    supervised.to_csv(args.out, index=False)

    print(f"Wrote {args.out.name}: {len(supervised)} rows, "
          f"{supervised['business_id'].nunique()} businesses")
    print(f"  columns: {list(supervised.columns)}")
    print(f"  targets: target_next_30d_revenue_kes, minimum_future_30d_revenue, "
          f"lower_percentile_future_30d_revenue, future_30d_revenue_volatility")


if __name__ == "__main__":
    main()
