"""Leakage-safe feature engineering for the Q-Cash v2 model.

Every feature is computed strictly from rows *before* the cut-off date. The
target is the sum of eligible Paybill revenue (integer minor units) over the 30
days *after* the cut-off.  No feature reads a future row, encodes the target,
or exposes the business identity.

Feature definitions follow the engine guide section 4.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import DATA_DIR, load_config

WINDOWS = (7, 30, 90)
HORIZON = 30


def _cumsum(arr: np.ndarray) -> np.ndarray:
    out = np.empty(len(arr) + 1, dtype=np.float64)
    out[0] = 0.0
    np.cumsum(arr, dtype=np.float64, out=out[1:])
    return out


def _window_sum(cum: np.ndarray, lo: int, hi: int) -> float:
    if hi <= lo:
        return 0.0
    return float(cum[hi] - cum[lo])


def _cv(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    mean = float(values.mean())
    if not np.isfinite(mean) or mean <= 0:
        return 0.0
    std = float(values.std())
    return std / mean


def build_features(daily: pd.DataFrame, businesses: pd.DataFrame) -> pd.DataFrame:
    """Return one row per (business, weekly cut-off) with features + target."""
    cfg = load_config()
    feat_cfg = cfg["features"]
    step = feat_cfg["observation_step_days"]
    warmup = feat_cfg["warmup_days"]

    sector_map = businesses.set_index("business_id")["sector"].to_dict()
    size_map = businesses.set_index("business_id")["size_class"].to_dict()
    active_map = businesses.set_index("business_id")["paybill_active_from"].to_dict()

    rows: list[dict] = []
    for biz_id, sub in daily.groupby("business_id", sort=True):
        sub = sub.sort_values("date").reset_index(drop=True)
        dates = sub["date"].to_numpy(dtype="datetime64[D]")
        rev = sub["eligible_revenue_minor"].to_numpy(dtype=np.float64)
        tx = sub["eligible_tx_count"].to_numpy(dtype=np.float64)
        sum_sq = sub["sum_sq_minor"].to_numpy(dtype=np.float64)
        max_tx = sub["max_tx_minor"].to_numpy(dtype=np.float64)
        true_level = sub["true_level_minor"].to_numpy(dtype=np.float64)
        shock = sub["shock"].to_numpy(dtype=np.float64)
        dow = sub["date"].dt.weekday.to_numpy()

        cum_rev = _cumsum(rev)
        cum_tx = _cumsum(tx)
        cum_sq = _cumsum(sum_sq)

        start_date = np.datetime64(pd.Timestamp(active_map.get(biz_id, dates[0])).to_pydatetime().date())
        first_cut = start_date + np.timedelta64(warmup, "D")
        last_date = dates[-1]
        last_cut = last_date - np.timedelta64(HORIZON, "D")

        cut = first_cut
        while cut <= last_cut:
            hi = int(np.searchsorted(dates, cut, side="left"))  # rows strictly before cut
            if hi < warmup:
                cut += np.timedelta64(step, "D")
                continue

            rev_30 = _window_sum(cum_rev, max(0, np.searchsorted(dates, cut - np.timedelta64(30, "D"), "left")), hi)
            rev_90 = _window_sum(cum_rev, max(0, np.searchsorted(dates, cut - np.timedelta64(90, "D"), "left")), hi)
            rev_7 = _window_sum(cum_rev, max(0, np.searchsorted(dates, cut - np.timedelta64(7, "D"), "left")), hi)

            lo30 = int(np.searchsorted(dates, cut - np.timedelta64(30, "D"), "left"))
            win_rev = rev[lo30:hi]
            win_tx = tx[lo30:hi]
            tx_30 = float(win_tx.sum())
            active_days = float((win_rev > 0).sum())

            mean_txn = (rev_30 / tx_30) if tx_30 > 0 else 0.0
            cv = _cv(win_rev)

            # trend: mean of last 14d vs prior 14d
            lo14 = int(np.searchsorted(dates, cut - np.timedelta64(14, "D"), "left"))
            lo28 = int(np.searchsorted(dates, cut - np.timedelta64(28, "D"), "left"))
            mean_14 = float(rev[lo14:hi].mean()) if lo14 < hi else 0.0
            mean_prior = float(rev[lo28:lo14].mean()) if lo28 < lo14 else 0.0
            trend_pct = (mean_14 - mean_prior) / mean_prior * 100.0 if mean_prior > 0 else 0.0
            trend_pct = float(np.clip(trend_pct, -100.0, 100.0))

            # HHI and top1 share over 30d
            sq_30 = _window_sum(cum_sq, lo30, hi)
            hhi = (sq_30 / (rev_30 * rev_30)) if rev_30 > 0 else 1.0
            hhi = float(np.clip(hhi, 0.0, 1.0))
            top1 = float(max_tx[lo30:hi].max()) if lo30 < hi else 0.0
            top1_share = (top1 / rev_30) if rev_30 > 0 else 0.0

            # active weeks over 30d
            weekdays = pd.DatetimeIndex(dates[lo30:hi]).isocalendar().week.to_numpy()
            active_weeks = len(np.unique(weekdays[win_rev[lo30:hi] > 0])) if lo30 < hi else 0
            freq_per_active_week = (tx_30 / active_weeks) if active_weeks > 0 else 0.0

            # operational recency: days since the last day with revenue
            active_before = np.where(rev[:hi] > 0)[0]
            if active_before.size:
                days_since_last_tx = int((dates[hi - 1] - dates[active_before[-1]]).astype(int))
            else:
                days_since_last_tx = int((dates[hi - 1] - dates[0]).astype(int)) + 1

            # activity over the last 7 days (foot-traffic momentum)
            lo7 = int(np.searchsorted(dates, cut - np.timedelta64(7, "D"), "left"))
            active_days_7 = float((rev[lo7:hi] > 0).sum())

            # weekend revenue share over 30d
            weekend_mask = dow[lo30:hi] >= 5
            weekend_rev_share_30 = (
                float(rev[lo30:hi][weekend_mask].sum() / rev_30) if rev_30 > 0 else 0.0
            )

            # tenure
            first_receipt = int(np.argmax(rev > 0)) if (rev > 0).any() else 0
            tenure_days = float((dates[hi - 1] - dates[first_receipt]).astype(int) + 1) if hi > 0 else 0.0

            # target: next 30d eligible revenue
            fhi = int(np.searchsorted(dates, cut + np.timedelta64(HORIZON, "D"), "left"))
            target = _window_sum(cum_rev, hi, min(fhi, len(dates)))

            # oracle forecast: E[observed revenue next 30d | latent state at cut]
            oracle = _oracle_forecast(true_level, shock, hi, min(fhi, len(dates)), cfg)

            rows.append({
                "business_id": biz_id,
                "sector": sector_map.get(biz_id, "unknown"),
                "size_class": size_map.get(biz_id, "unknown"),
                "as_of_date": pd.Timestamp(cut),
                "rev_sum_7": rev_7,
                "rev_sum_30": rev_30,
                "rev_sum_90": rev_90,
                "txn_count_30": tx_30,
                "freq_per_active_week": freq_per_active_week,
                "active_days_30": active_days,
                "adr_pct": (active_days / 30.0 * 100.0),
                "mean_txn_minor": mean_txn,
                "cv_daily_rev_30": cv,
                "trend_pct": trend_pct,
                "hhi_30": hhi,
                "top1_share_30": top1_share,
                "tenure_days": tenure_days,
                "days_since_last_tx": days_since_last_tx,
                "active_days_7": active_days_7,
                "weekend_rev_share_30": weekend_rev_share_30,
                "target_minor": target,
                "oracle_forecast_minor": oracle,
            })
            cut += np.timedelta64(step, "D")

    out = pd.DataFrame(rows)
    out["as_of_date"] = pd.to_datetime(out["as_of_date"])
    return out


def _oracle_forecast(true_level: np.ndarray, shock: np.ndarray, hi: int, fhi: int, cfg: dict) -> float:
    """Irreducible-error floor: best forecast that knows the latent state.

    The oracle observes the deterministic level and the *current* latent shock
    but not the future shock innovations. For each future day d=1..30 it predicts
    true_level[t] * E[exp(shock_t) | shock at cut-off].
    """
    if hi >= fhi or hi >= len(true_level):
        return 0.0
    rho = cfg["generator"]["shock_rho"]
    sigma = cfg["generator"]["shock_sigma"]
    shock_now = shock[hi - 1] if hi > 0 else 0.0
    total = 0.0
    for t in range(hi, fhi):
        d = t - hi + 1
        rho_d = rho ** d
        var = (sigma ** 2) * (1.0 - rho ** (2 * d)) / (1.0 - rho * rho)
        expected_exp = np.exp(rho_d * shock_now + 0.5 * var)
        total += true_level[t] * expected_exp
    return float(total)


if __name__ == "__main__":
    daily = pd.read_csv(DATA_DIR / "daily.csv", parse_dates=["date"])
    businesses = pd.read_csv(DATA_DIR / "businesses.csv")
    df = build_features(daily, businesses)
    df.to_csv(DATA_DIR / "supervised.csv", index=False)
    print(f"Wrote supervised.csv: {len(df)} rows, {df['business_id'].nunique()} businesses")
    print(df[["rev_sum_30", "target_minor", "oracle_forecast_minor"]].describe().round(0).to_string())
