"""Feasibility test: how accurate is a forecast with only N days of history?

For each history length H we build a simple, robust run-rate forecast from the
last H days of eligible revenue and score it against the realised next-30-day
revenue. This quantifies the accuracy penalty of a short warm-up (e.g. KopoKopo's
14 days) and informs how much to shrink the advance for short-history businesses.

Run-rate forecast:  forecast_30d = (sum of eligible revenue over last H days) / H * 30
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import DATA_DIR

HORIZON = 30
STEP = 7
H_VALUES = [7, 14, 21, 30, 45, 60, 90]


def _metrics(y: np.ndarray, p: np.ndarray) -> dict:
    ape = np.abs(p - y) / np.maximum(y, 1) * 100
    return {
        "mae": float(np.mean(np.abs(p - y))),
        "median_ape": float(np.median(ape)),
        "within25": float(np.mean(ape <= 25) * 100),
        "within40": float(np.mean(ape <= 40) * 100),
        "r2": float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)),
        "n": int(len(y)),
    }


def main() -> None:
    daily = pd.read_csv(DATA_DIR / "daily.csv", parse_dates=["date"])
    print(f"{'H (days)':>9} {'n obs':>7} {'MAE (KSh)':>11} {'median APE':>11} "
          f"{'within ±25%':>12} {'within ±40%':>12} {'R2':>6}")
    print("-" * 74)
    for H in H_VALUES:
        ys, ps = [], []
        for _, sub in daily.groupby("business_id"):
            sub = sub.sort_values("date").reset_index(drop=True)
            dates = sub["date"].to_numpy(dtype="datetime64[D]")
            rev = sub["eligible_revenue_minor"].to_numpy(dtype=float)
            start = dates[0]
            cut = start + np.timedelta64(H, "D")
            last_cut = dates[-1] - np.timedelta64(HORIZON, "D")
            while cut <= last_cut:
                hi = int(np.searchsorted(dates, cut, "left"))
                lo = int(np.searchsorted(dates, cut - np.timedelta64(H, "D"), "left"))
                if hi <= lo:
                    cut += np.timedelta64(STEP, "D")
                    continue
                run_rate = rev[lo:hi].sum() / H * HORIZON
                fhi = int(np.searchsorted(dates, cut + np.timedelta64(HORIZON, "D"), "left"))
                target = rev[hi:fhi].sum()
                ys.append(target)
                ps.append(run_rate)
                cut += np.timedelta64(STEP, "D")
        m = _metrics(np.array(ys), np.array(ps))
        print(f"{H:>9} {m['n']:>7} {m['mae']/100:>11,.0f} {m['median_ape']:>10.0f}% "
              f"{m['within25']:>11.0f}% {m['within40']:>11.0f}% {m['r2']:>6.2f}")


if __name__ == "__main__":
    main()
