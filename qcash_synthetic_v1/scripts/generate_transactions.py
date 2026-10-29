"""Synthetic MSME transaction generator for the Q-Cash POC.

Generates transaction-level M-Pesa Till revenue records for a population of
synthetic businesses. Each business is drawn from one of eight archetypes that
are *not* copies of the real barbershop: they model distinct, controlled
cash-flow behaviours so we can test whether transaction-behaviour features
distinguish reliable businesses from volatile or declining ones.

Everything here is a simulation assumption for a proof of concept. It must
never be presented as measured Kenyan MSME statistics.

Reproducibility
---------------
A single master seed drives every random draw. Each business gets a derived
seed so that adding/removing a business does not change the others, and the
whole population is regenerated identically for a given master seed.

Outputs
-------
    data/synthetic_transactions.csv   transaction-level records
    data/synthetic_daily_metrics.csv  per-business daily aggregates
    data/synthetic_metadata.json      generation provenance + assumptions

Usage
-----
    python generate_transactions.py \
        --n-per-archetype 8 --seed 42 \
        --start 2024-01-01 --end 2026-09-30
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_OUT_DIR = Path(__file__).resolve().parent.parent / "data"

# Python weekday: Monday=0 .. Sunday=6
FLAT_WEEKLY = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0}
FLAT_MONTHLY = {m: 1.0 for m in range(1, 13)}


def _weekly(overrides: dict[int, float] | None = None) -> dict[int, float]:
    result = dict(FLAT_WEEKLY)
    if overrides:
        result.update(overrides)
    return result


def _monthly(overrides: dict[int, float] | None = None) -> dict[int, float]:
    result = dict(FLAT_MONTHLY)
    if overrides:
        result.update(overrides)
    return result


# ---------------------------------------------------------------------------
# Archetype definitions.
#
# ``baseline_daily`` is the mean daily revenue for an *active* day at the start
# of the window; the growth/decline rate is applied day-over-day. ``daily_cv``
# is the coefficient of variation of the multiplicative noise applied to daily
# revenue. ``weekly``/``monthly`` are multiplicative seasonality profiles.
# ---------------------------------------------------------------------------
ARCHETYPES: dict[str, dict] = {
    "A_stable_barbershop": {
        "label": "Stable barbershop",
        "baseline_daily": 4000.0,
        "avg_tx_value": 400.0,
        "daily_cv": 0.15,
        "growth_daily": 0.0,
        "zero_day_prob": 0.01,
        "high_day_prob": 0.03,
        "high_day_mult": 1.5,
        "weekly": _weekly({5: 1.15, 6: 0.90}),
        "monthly": _monthly(),
        "tx_amount_cv": 0.30,
    },
    "B_volatile_retail": {
        "label": "Highly volatile retail",
        "baseline_daily": 5000.0,
        "avg_tx_value": 900.0,
        "daily_cv": 0.60,
        "growth_daily": 0.0,
        "zero_day_prob": 0.05,
        "high_day_prob": 0.08,
        "high_day_mult": 3.5,
        "weekly": _weekly(),
        "monthly": _monthly(),
        "tx_amount_cv": 0.55,
    },
    "C_growing": {
        "label": "Growing business",
        "baseline_daily": 2500.0,
        "avg_tx_value": 300.0,
        "daily_cv": 0.20,
        "growth_daily": 0.0012,
        "zero_day_prob": 0.02,
        "high_day_prob": 0.03,
        "high_day_mult": 1.6,
        "weekly": _weekly({5: 1.10}),
        "monthly": _monthly(),
        "tx_amount_cv": 0.35,
    },
    "D_declining": {
        "label": "Declining business",
        "baseline_daily": 8000.0,
        "avg_tx_value": 500.0,
        "daily_cv": 0.18,
        "growth_daily": -0.0011,
        "zero_day_prob": 0.02,
        "high_day_prob": 0.02,
        "high_day_mult": 1.4,
        "weekly": _weekly(),
        "monthly": _monthly(),
        "tx_amount_cv": 0.30,
    },
    "E_weekly_seasonal": {
        "label": "Strong weekly seasonality",
        "baseline_daily": 5000.0,
        "avg_tx_value": 600.0,
        "daily_cv": 0.20,
        "growth_daily": 0.0,
        "zero_day_prob": 0.02,
        "high_day_prob": 0.03,
        "high_day_mult": 1.5,
        "weekly": _weekly({0: 0.55, 1: 0.65, 2: 0.85, 3: 0.95, 4: 1.30, 5: 1.70, 6: 1.15}),
        "monthly": _monthly(),
        "tx_amount_cv": 0.35,
    },
    "F_monthly_seasonal": {
        "label": "Strong seasonal business",
        "baseline_daily": 4500.0,
        "avg_tx_value": 500.0,
        "daily_cv": 0.15,
        "growth_daily": 0.0,
        "zero_day_prob": 0.02,
        "high_day_prob": 0.03,
        "high_day_mult": 1.5,
        "weekly": _weekly(),
        "monthly": _monthly({
            1: 0.75, 2: 0.85, 3: 1.00, 4: 1.35, 5: 0.70, 6: 0.90,
            7: 0.90, 8: 1.40, 9: 1.00, 10: 1.00, 11: 1.00, 12: 1.60,
        }),
        "tx_amount_cv": 0.30,
    },
    "G_intermittent": {
        "label": "Intermittent / irregular MSME",
        "baseline_daily": 3000.0,
        "avg_tx_value": 700.0,
        "daily_cv": 1.0,
        "growth_daily": 0.0,
        "zero_day_prob": 0.40,
        "high_day_prob": 0.05,
        "high_day_mult": 6.0,
        "weekly": _weekly(),
        "monthly": _monthly(),
        "tx_amount_cv": 0.50,
    },
    "H_high_volume": {
        "label": "Stable high-volume business",
        "baseline_daily": 15000.0,
        "avg_tx_value": 400.0,
        "daily_cv": 0.10,
        "growth_daily": 0.0,
        "zero_day_prob": 0.005,
        "high_day_prob": 0.02,
        "high_day_mult": 1.3,
        "weekly": _weekly({5: 1.05, 6: 1.05}),
        "monthly": _monthly(),
        "tx_amount_cv": 0.25,
    },
    "I_micro_vendor": {
        # Informal vegetable/fruit stall ("mama mboga"): small, frequent,
        # low-value transactions, market days up, many low/zero days.
        "label": "Micro vendor (mama mboga)",
        "baseline_daily": 450.0,
        "avg_tx_value": 45.0,
        "daily_cv": 0.50,
        "growth_daily": 0.0,
        "zero_day_prob": 0.18,
        "high_day_prob": 0.04,
        "high_day_mult": 2.0,
        "weekly": _weekly({1: 1.4, 5: 1.5, 6: 0.9}),
        "monthly": _monthly(),
        "tx_amount_cv": 0.50,
    },
    "J_micro_service": {
        # Casual / gig worker (boda boda, fundi): fewer, larger transactions,
        # irregular days and occasional large jobs.
        "label": "Micro service (boda boda / fundi)",
        "baseline_daily": 900.0,
        "avg_tx_value": 250.0,
        "daily_cv": 0.85,
        "growth_daily": 0.0,
        "zero_day_prob": 0.25,
        "high_day_prob": 0.05,
        "high_day_mult": 3.5,
        "weekly": _weekly({5: 1.2, 6: 1.1}),
        "monthly": _monthly(),
        "tx_amount_cv": 0.60,
    },
}


def _derive_seed(master_seed: int, business_index: int) -> int:
    """Deterministic per-business seed from the master seed + index."""
    return int(master_seed + 1_000_003 * (business_index + 1))


def _level_series(params: dict, dates: pd.DatetimeIndex, rng: np.random.Generator) -> np.ndarray:
    """Expected daily revenue before noise: level x trend x seasonality."""
    baseline = params["baseline_daily"]
    growth = params["growth_daily"]
    t = np.arange(len(dates))
    level = baseline * np.power(1.0 + growth, t)
    month_mult = np.array([params["monthly"][d.month] for d in dates])
    dow_mult = np.array([params["weekly"][d.weekday()] for d in dates])
    return level * month_mult * dow_mult


def _daily_revenue(params: dict, dates: pd.DatetimeIndex, rng: np.random.Generator) -> np.ndarray:
    """Simulate the daily revenue series for one business."""
    base = _level_series(params, dates, rng)
    cv = params["daily_cv"]
    sigma = np.sqrt(np.log(1.0 + cv * cv))
    mu = -0.5 * sigma * sigma
    noise = rng.lognormal(mu, sigma, len(dates))
    revenue = base * noise

    zero_mask = rng.random(len(dates)) < params["zero_day_prob"]
    revenue[zero_mask] = 0.0

    high_mask = rng.random(len(dates)) < params["high_day_prob"]
    revenue[high_mask] *= params["high_day_mult"]

    return np.maximum(revenue, 0.0)


def _transactions_for_day(
    business_id: str,
    date: pd.Timestamp,
    revenue: float,
    params: dict,
    rng: np.random.Generator,
    customer_pool: np.ndarray,
    tx_counter: int,
) -> tuple[list[dict], int]:
    """Transaction records for one day, rescaled to sum to the day's revenue."""
    rows: list[dict] = []
    if revenue <= 0:
        return rows, tx_counter

    avg_value = params["avg_tx_value"]
    n_tx = int(rng.poisson(revenue / avg_value))
    n_tx = max(n_tx, 1)

    per_tx_mean = revenue / n_tx
    sigma = params["tx_amount_cv"]
    mu = np.log(per_tx_mean) - 0.5 * sigma * sigma
    amounts = rng.lognormal(mu, sigma, n_tx)
    amounts = amounts / amounts.sum() * revenue
    amounts = np.clip(amounts, 1.0, None)

    customers = rng.choice(customer_pool, size=n_tx, replace=True)
    seconds = rng.integers(0, 24 * 3600, size=n_tx)

    for i, amount in enumerate(amounts):
        t = pd.Timestamp(date).normalize() + pd.Timedelta(seconds=int(seconds[i]))
        rows.append({
            "business_id": business_id,
            "transaction_id": f"{business_id}-T{tx_counter:07d}",
            "transaction_date": date.date().isoformat(),
            "transaction_time": t.strftime("%H:%M:%S"),
            "transaction_amount_kes": round(float(amount), 2),
            "transaction_type": "till_payment_received",
            "is_credit": 1,
            "customer_id_hash": customers[i],
            "archetype": params["_archetype_key"],
            "data_source": "synthetic",
        })
        tx_counter += 1
    return rows, tx_counter


def _customer_pool(rng: np.random.Generator, size: int = 180) -> np.ndarray:
    return np.array([f"{int(x):x}" for x in rng.integers(0, 2**48, size=size)])


def generate_business(
    key: str,
    index: int,
    master_seed: int,
    dates: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate one business's transaction and daily-metric frames."""
    template = ARCHETYPES[key]
    rng = np.random.default_rng(_derive_seed(master_seed, index))

    params = dict(template)
    params["baseline_daily"] = float(template["baseline_daily"] * rng.lognormal(0.0, 0.12))
    params["daily_cv"] = float(np.clip(template["daily_cv"] * rng.lognormal(0.0, 0.10), 0.03, 1.5))
    params["_archetype_key"] = key

    business_id = f"SYN_{index:03d}"
    revenue = _daily_revenue(params, dates, rng)
    customer_pool = _customer_pool(rng)

    tx_rows: list[dict] = []
    tx_counter = 0
    for date, rev in zip(dates, revenue):
        day_rows, tx_counter = _transactions_for_day(
            business_id, date, rev, params, rng, customer_pool, tx_counter
        )
        tx_rows.extend(day_rows)

    tx_df = pd.DataFrame(tx_rows)

    daily = tx_df.groupby("transaction_date", as_index=False).agg(
        revenue_kes=("transaction_amount_kes", "sum"),
        transaction_count=("transaction_id", "count"),
        avg_transaction_kes=("transaction_amount_kes", "mean"),
        median_transaction_kes=("transaction_amount_kes", "median"),
        std_transaction_kes=("transaction_amount_kes", "std"),
        max_transaction_kes=("transaction_amount_kes", "max"),
        unique_customers=("customer_id_hash", "nunique"),
    )
    daily["transaction_date"] = pd.to_datetime(daily["transaction_date"])
    daily = daily.sort_values("transaction_date")

    full = pd.DataFrame({"transaction_date": dates})
    daily = full.merge(daily, on="transaction_date", how="left")
    for col in [
        "revenue_kes", "transaction_count", "avg_transaction_kes",
        "median_transaction_kes", "std_transaction_kes",
        "max_transaction_kes", "unique_customers",
    ]:
        daily[col] = daily[col].fillna(0.0)

    daily["business_id"] = business_id
    daily["archetype"] = key
    daily["data_source"] = "synthetic"
    daily["active_day"] = (daily["transaction_count"] > 0).astype(int)
    daily["day_of_week"] = daily["transaction_date"].dt.weekday
    daily["weekday"] = daily["transaction_date"].dt.day_name()
    daily["month"] = daily["transaction_date"].dt.month
    daily["quarter"] = daily["transaction_date"].dt.quarter
    daily["year"] = daily["transaction_date"].dt.year

    return tx_df, daily


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-per-archetype", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2026-09-30")
    parser.add_argument("--archetypes", default=None,
                        help="Comma-separated archetype keys; default = all.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    keys = list(ARCHETYPES.keys()) if args.archetypes is None else args.archetypes.split(",")
    unknown = [k for k in keys if k not in ARCHETYPES]
    if unknown:
        raise SystemExit(f"Unknown archetypes: {unknown}. Valid: {list(ARCHETYPES)}")

    dates = pd.date_range(args.start, args.end, freq="D")
    started = datetime.now(timezone.utc)

    tx_frames: list[pd.DataFrame] = []
    daily_frames: list[pd.DataFrame] = []
    for key in keys:
        for j in range(args.n_per_archetype):
            index = len(daily_frames)
            tx, daily = generate_business(key, index, args.seed, dates)
            tx_frames.append(tx)
            daily_frames.append(daily)

    tx_all = pd.concat(tx_frames, ignore_index=True)
    daily_all = pd.concat(daily_frames, ignore_index=True)

    tx_path = args.out_dir / "synthetic_transactions.csv"
    daily_path = args.out_dir / "synthetic_daily_metrics.csv"
    tx_all.to_csv(tx_path, index=False)
    daily_all.to_csv(daily_path, index=False)

    metadata = {
        "data_source": "synthetic",
        "created_utc": started.isoformat(),
        "master_seed": args.seed,
        "n_businesses": int(daily_all["business_id"].nunique()),
        "n_transactions": int(len(tx_all)),
        "n_archetypes": len(keys),
        "archetypes": {k: {"label": ARCHETYPES[k]["label"],
                          "n_businesses": args.n_per_archetype} for k in keys},
        "date_range": [args.start, args.end],
        "disclaimer": (
            "Synthetic simulation data for a proof of concept. Parameters are "
            "simulation assumptions, NOT measured Kenyan MSME statistics, and "
            "results do not establish generalisation to real businesses."
        ),
        "assumptions": [
            "Daily revenue follows a log-normal multiplicative noise model with "
            "per-archetype coefficient of variation.",
            "Zero/low-revenue and unusually-high-revenue days are explicit "
            "Bernoulli events with per-archetype probabilities.",
            "Growth/decline is geometric day-over-day.",
            "Weekly and monthly seasonality are multiplicative profiles.",
            "Transaction counts are Poisson with mean revenue / average "
            "transaction value; per-transaction amounts are log-normal and "
            "rescaled so each day's transactions sum to that day's revenue.",
        ],
    }
    meta_path = args.out_dir / "synthetic_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2))

    print(f"Wrote {tx_path.name}: {len(tx_all):,} transactions "
          f"across {metadata['n_businesses']} businesses")
    print(f"Wrote {daily_path.name}: {len(daily_all):,} business-days")
    print(f"Wrote {meta_path.name}")


if __name__ == "__main__":
    main()
