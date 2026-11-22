"""Deterministic synthetic MSME generator for the Q-Cash v2 rebuild.

The generator produces a panel of synthetic Kenyan MSME businesses whose
Paybill receipts follow a structured, *honest* data-generating process:

  * a latent AR(1) demand shock that no model can observe (irreducible error),
  * per-sector ticket-size, weekday and annual seasonality,
  * a digital share (only part of sales reach the Paybill),
  * micro businesses whose tickets go down to **1 shilling** (100 minor units),
  * non-revenue events (refunds, reversals, own transfers, capital injections,
    duplicate callbacks, outages) that a transaction-intelligence stage must
    strip out before any revenue is counted.

Every money value is an integer number of minor units (1 KSh = 100). The
generator is deterministic given ``master_seed``: the same seed and config
reproduce byte-identical files.

Outputs (in ``data/``):
    businesses.csv     one row per business + latent ground truth
    transactions.csv   observed Paybill transactions (integer minor units)
    daily.csv          per-business daily aggregates (eligible + oracle columns)
    metadata.json      provenance
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from common import DATA_DIR, derive_seed, load_config

# ---------------------------------------------------------------------------
# Sector definitions.  ``median_ticket_ksh`` and ``sigma_ticket`` parameterise a
# lognormal ticket-size distribution.  ``base_daily_ksh`` is the mean daily
# *true* sales at the start of the window.  ``weekday`` is a multiplicative
# day-of-week profile (Monday=0 .. Sunday=6).
# ---------------------------------------------------------------------------
SECTORS: dict[str, dict] = {
    "barbershop": {
        "median_ticket_ksh": 300.0, "sigma_ticket": 0.35,
        "base_daily_ksh": 4200.0, "weekday": {5: 1.25, 6: 0.85, 0: 0.90},
        "micro_item_share": 0.0,
    },
    "salon": {
        "median_ticket_ksh": 500.0, "sigma_ticket": 0.35,
        "base_daily_ksh": 5200.0, "weekday": {5: 1.30, 6: 0.80, 0: 0.85},
        "micro_item_share": 0.0,
    },
    "pharmacy": {
        "median_ticket_ksh": 350.0, "sigma_ticket": 0.40,
        "base_daily_ksh": 8200.0, "weekday": {},
        "micro_item_share": 0.0,
    },
    "general_retail": {
        "median_ticket_ksh": 80.0, "sigma_ticket": 0.50,
        "base_daily_ksh": 4000.0, "weekday": {5: 1.15},
        "micro_item_share": 0.10,
    },
    "greengrocer": {
        "median_ticket_ksh": 25.0, "sigma_ticket": 0.70,
        "base_daily_ksh": 800.0, "weekday": {1: 1.35, 5: 1.45, 6: 0.95},
        "micro_item_share": 0.35,       # single sukuma / onion / tomato -> ~1 KSh
    },
    "hardware": {
        "median_ticket_ksh": 900.0, "sigma_ticket": 0.50,
        "base_daily_ksh": 9000.0, "weekday": {5: 1.10},
        "micro_item_share": 0.0,
    },
    "food_vendor": {
        "median_ticket_ksh": 150.0, "sigma_ticket": 0.45,
        "base_daily_ksh": 3500.0, "weekday": {5: 1.20, 6: 1.10},
        "micro_item_share": 0.0,
    },
    "secondhand_clothing": {
        "median_ticket_ksh": 600.0, "sigma_ticket": 0.55,
        "base_daily_ksh": 5000.0, "weekday": {5: 1.25, 6: 1.15},
        "micro_item_share": 0.0,
    },
}

# annual seasonality profile (week of year -> multiplier), gentle
_WEEKLY_SEASON = {
    48: 1.25, 49: 1.30, 50: 1.35, 51: 1.40, 52: 1.30,   # December holidays
    0: 1.10, 1: 1.00,
}


def _annual_factor(week_of_year: int) -> float:
    return float(_WEEKLY_SEASON.get(week_of_year, 1.0))


def _weekday_factor(sector: str, dow: int) -> float:
    return float(SECTORS[sector]["weekday"].get(dow, 1.0))


def _sector_median_ticket_minor(sector: str) -> float:
    return SECTORS[sector]["median_ticket_ksh"] * 100.0


def _mean_of_lognormal(median: float, sigma: float) -> float:
    return median * np.exp(0.5 * sigma * sigma)


def _generate_business_meta(index: int, sector: str, cfg: dict, rng: np.random.Generator) -> dict:
    """Business-level latent parameters (unobserved by the model)."""
    g = cfg["generator"]
    digital_share = float(rng.uniform(g["digital_share_min"], g["digital_share_max"]))
    latent_quality = float(rng.lognormal(0.0, 0.15))
    base_daily = float(SECTORS[sector]["base_daily_ksh"] * rng.lognormal(0.0, 0.20))
    # gentle trend; a few businesses get a structural regime change
    trend = float(rng.normal(0.0, 0.0003))
    if rng.random() < 0.05:
        trend = float(rng.choice([-0.0025, 0.0025]))
    return {
        "business_id": f"SYN_{index:03d}",
        "sector": sector,
        "size_class": _size_class(base_daily),
        "open_date": cfg["generator"]["start_date"],
        "paybill_active_from": cfg["generator"]["start_date"],
        "digital_share": digital_share,
        "latent_quality": latent_quality,
        "base_daily_ksh": base_daily,
        "trend": trend,
        "churn_date": None,
    }


def _size_class(base_daily_ksh: float) -> str:
    if base_daily_ksh < 1500:
        return "micro"
    if base_daily_ksh < 6000:
        return "small"
    return "medium"


def generate_business(
    business: dict,
    dates: pd.DatetimeIndex,
    cfg: dict,
    master_seed: int,
    index: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate one business -> (transactions DataFrame, daily DataFrame)."""
    g = cfg["generator"]
    sector = business["sector"]
    rng = np.random.default_rng(derive_seed(master_seed, index))

    median_ticket_minor = _sector_median_ticket_minor(sector)
    sigma_ticket = SECTORS[sector]["sigma_ticket"]
    mean_ticket_minor = _mean_of_lognormal(median_ticket_minor, sigma_ticket)

    base = business["base_daily_ksh"] * 100.0  # minor units
    trend = business["trend"]
    quality = business["latent_quality"]
    digital_share = business["digital_share"]

    n = len(dates)
    t = np.arange(n)
    rho = g["shock_rho"]
    sigma_shock = g["shock_sigma"]

    # latent AR(1) shock
    eps = rng.normal(0.0, sigma_shock, n)
    shock = np.empty(n)
    shock[0] = eps[0] / np.sqrt(1.0 - rho * rho)
    for i in range(1, n):
        shock[i] = rho * shock[i - 1] + eps[i]

    month = np.array([d.month for d in dates])
    dow = np.array([d.weekday() for d in dates])
    week = np.array([int(d.isocalendar()[1]) % 52 for d in dates])

    det_level = base * np.power(1.0 + trend, t) * quality
    payday_boost = float(g.get("payday_boost", 0.0))
    for d in range(n):
        det_level[d] *= _weekday_factor(sector, dow[d]) * _annual_factor(week[d])
        # end-of-month payday cycle: salaried customers spend more from the 25th
        # to the 5th. Modest, so it shapes the calendar without dominating the
        # inherent demand noise.
        dom = dates[d].day
        if dom >= 25 or dom <= 5:
            det_level[d] *= (1.0 + payday_boost)
    # gentle month-of-year effect (December + school terms) folded into annual

    # temporary closure spells (stock-out, illness, travel, renovation): whole
    # days with zero receipts, which create genuine gaps in trading activity.
    # Gated behind a positive probability so a zero config consumes no RNG and
    # reproduces the baseline stream exactly.
    closure_days = np.zeros(n, dtype=bool)
    start_prob = float(g.get("closure_start_prob", 0.0))
    if start_prob > 0:
        cmin = int(g.get("closure_min_days", 3))
        cmax = int(g.get("closure_max_days", 14))
        i = 0
        while i < n:
            if rng.random() < start_prob:
                dur = int(rng.integers(cmin, cmax + 1))
                closure_days[i:i + dur] = True
                i += dur
            else:
                i += 1

    # expected *observed* Paybill revenue per day (before shock / noise)
    true_level_minor = det_level * digital_share

    # realised arrivals -> observed paybill transactions
    tx_rows: list[dict] = []
    daily_rows: list[dict] = []
    tx_counter = 0
    outage_days = set(np.where(rng.random(n) < g["outage_rate"])[0])

    for d in range(n):
        lam = det_level[d] * np.exp(shock[d]) / mean_ticket_minor
        n_arrivals = int(rng.poisson(lam)) if lam > 0 else 0
        if n_arrivals == 0 or closure_days[d]:
            day_rev = 0
            day_tx = 0
            day_sq = 0
            day_max = 0
        else:
            amounts = _sample_tickets(sector, n_arrivals, median_ticket_minor, sigma_ticket, g, rng)
            paybill_mask = rng.random(n_arrivals) < digital_share
            observed = amounts[paybill_mask]
            # outages kill the whole day's paybill receipts
            if d in outage_days:
                observed = np.array([], dtype=np.int64)
            day_rev = int(observed.sum())
            day_tx = int(observed.size)
            day_sq = int(np.sum(observed.astype(np.int64) ** 2))
            day_max = int(observed.max()) if observed.size else 0
            for amt in observed:
                tx_rows.append({
                    "ref": f"SYN_{index:03d}-{tx_counter:08d}",
                    "business_id": business["business_id"],
                    "customer_id": None,
                    "occurred_at": pd.Timestamp(dates[d]).isoformat(),
                    "amount_minor": int(amt),
                    "type": "credit",
                    "channel": "paybill",
                    "is_reversal_of": None,
                    "reference": None,
                    "payer_hash": None,
                    "category": "revenue_eligible",
                })
                tx_counter += 1

        daily_rows.append({
            "business_id": business["business_id"],
            "date": pd.Timestamp(dates[d]),
            "true_level_minor": float(true_level_minor[d]),
            "shock": float(shock[d]),
            "eligible_revenue_minor": day_rev,
            "eligible_tx_count": day_tx,
            "sum_sq_minor": day_sq,
            "max_tx_minor": day_max,
        })

    tx_df = pd.DataFrame(tx_rows)
    daily_df = pd.DataFrame(daily_rows)
    return tx_df, daily_df


def _sample_tickets(
    sector: str,
    n: int,
    median_ticket_minor: float,
    sigma: float,
    cfg: dict,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample n ticket amounts (integer minor units), floor at min_amount_minor."""
    mu = np.log(median_ticket_minor)
    amounts = rng.lognormal(mu, sigma, n)
    micro_share = SECTORS[sector]["micro_item_share"]
    if micro_share > 0:
        is_micro = rng.random(n) < micro_share
        # micro items: a few shillings, uniformly 1..15 KSh
        micro = rng.integers(1, 16, int(is_micro.sum())) * 100
        amounts[is_micro] = micro.astype(float)
    min_amt = cfg["min_amount_minor"]
    round_to = cfg["round_to_minor"]
    rounded = np.maximum(min_amt, np.rint(amounts / round_to) * round_to).astype(np.int64)
    return rounded


def inject_non_revenue(
    tx_df: pd.DataFrame,
    businesses: pd.DataFrame,
    dates: pd.DatetimeIndex,
    cfg: dict,
    master_seed: int,
) -> pd.DataFrame:
    """Inject refunds, reversals, own transfers, capital injections and duplicates.

    Each injection is a debit (or flagged credit) that must be excluded by the
    transaction-intelligence stage. All injected rows carry a ground-truth
    ``category`` so classification can be verified.
    """
    g = cfg["generator"]
    rng = np.random.default_rng(master_seed + 999_983)
    out = tx_df.copy()
    out["amount_minor"] = out["amount_minor"].astype(np.int64)
    credit_idx = out.index.to_numpy()
    extra: list[dict] = []

    def next_ref(counter: list[int]) -> str:
        ref = f"INJ-{counter[0]:09d}"
        counter[0] += 1
        return ref

    counter = [0]

    # own transfers (credit that is really the owner's own money)
    n_own = int(len(out) * g["own_transfer_rate"])
    if n_own:
        own_idx = rng.choice(credit_idx, size=n_own, replace=False)
        out.loc[own_idx, "category"] = "non_revenue_own_transfer"
        out.loc[own_idx, "reference"] = "own_transfer"
        # payer hash equals the business's own registered number hash
        out.loc[own_idx, "payer_hash"] = out.loc[own_idx, "business_id"].map(
            lambda b: f"own-{b}")

    # reversals: matching debit within 72h, flagged
    n_rev = int(len(out) * g["reversal_rate"])
    if n_rev:
        rev_idx = rng.choice(credit_idx, size=n_rev, replace=False)
        for i in rev_idx:
            row = out.loc[i]
            out.loc[i, "category"] = "non_revenue_reversal"
            extra.append({
                "ref": next_ref(counter),
                "business_id": row["business_id"],
                "customer_id": row["customer_id"],
                "occurred_at": row["occurred_at"],
                "amount_minor": -int(row["amount_minor"]),
                "type": "debit",
                "channel": "paybill",
                "is_reversal_of": row["ref"],
                "reference": "reversal",
                "payer_hash": row["payer_hash"],
                "category": "non_revenue_reversal",
            })

    # refunds: debit matching a credit, no reversal flag
    n_ref = int(len(out) * g["refund_rate"])
    if n_ref:
        ref_idx = rng.choice(credit_idx, size=n_ref, replace=False)
        for i in ref_idx:
            row = out.loc[i]
            out.loc[i, "category"] = "non_revenue_refund"
            extra.append({
                "ref": next_ref(counter),
                "business_id": row["business_id"],
                "customer_id": row["customer_id"],
                "occurred_at": row["occurred_at"],
                "amount_minor": -int(row["amount_minor"]),
                "type": "debit",
                "channel": "paybill",
                "is_reversal_of": None,
                "reference": "refund",
                "payer_hash": row["payer_hash"],
                "category": "non_revenue_refund",
            })

    # capital injections: occasional large credits
    for biz_id, biz in businesses.set_index("business_id").iterrows():
        biz_tx = out[out["business_id"] == biz_id]
        if biz_tx.empty:
            continue
        med = float(biz_tx["amount_minor"].median())
        biz_dates = pd.to_datetime(biz_tx["occurred_at"]).dt.date.unique()
        n_cap = max(0, int(round(len(biz_dates) * g["capital_injection_rate"])))
        for _ in range(n_cap):
            mult = float(rng.uniform(g["capital_mult_min"], g["capital_mult_max"]))
            amt = int(round(med * mult))
            d = pd.Timestamp(rng.choice(biz_dates))
            extra.append({
                "ref": next_ref(counter),
                "business_id": biz_id,
                "customer_id": None,
                "occurred_at": d.isoformat(),
                "amount_minor": amt,
                "type": "credit",
                "channel": "paybill",
                "is_reversal_of": None,
                "reference": str(rng.choice(["loan", "grant", "sacco", "capital"])),
                "payer_hash": None,
                "category": "non_revenue_capital",
            })

    # duplicate callbacks: same ref delivered twice
    n_dup = int(len(out) * g["duplicate_rate"])
    if n_dup:
        dup_idx = rng.choice(credit_idx, size=n_dup, replace=False)
        for i in dup_idx:
            row = out.loc[i]
            extra.append({
                "ref": row["ref"],  # same ref -> duplicate
                "business_id": row["business_id"],
                "customer_id": row["customer_id"],
                "occurred_at": row["occurred_at"],
                "amount_minor": int(row["amount_minor"]),
                "type": "credit",
                "channel": "paybill",
                "is_reversal_of": None,
                "reference": row["reference"],
                "payer_hash": row["payer_hash"],
                "category": "duplicate",
            })

    if extra:
        out = pd.concat([out, pd.DataFrame(extra)], ignore_index=True)
    return out


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()

    cfg = load_config()
    g = cfg["generator"]
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)

    dates = pd.date_range(g["start_date"], g["end_date"], freq="D")
    master_seed = args.seed if args.seed is not None else g["master_seed"]

    # business metadata
    sectors = list(SECTORS.keys())
    business_rows: list[dict] = []
    idx = 0
    for sector in sectors:
        for _ in range(g["n_businesses_per_sector"]):
            rng = np.random.default_rng(derive_seed(master_seed, idx))
            business_rows.append(_generate_business_meta(idx, sector, cfg, rng))
            idx += 1
    businesses = pd.DataFrame(business_rows)

    # churn: ~8% of businesses close mid-panel
    rng = np.random.default_rng(master_seed + 777)
    churn_idx = rng.choice(len(businesses), size=max(1, int(len(businesses) * 0.08)), replace=False)
    for i in churn_idx:
        close = pd.Timestamp(dates[len(dates) // 2 + int(rng.integers(0, 100))])
        businesses.loc[i, "churn_date"] = close.isoformat()

    tx_frames, daily_frames = [], []
    for index, business in businesses.iterrows():
        tx, daily = generate_business(business, dates, cfg, master_seed, index)
        tx_frames.append(tx)
        daily_frames.append(daily)

    tx_all = pd.concat(tx_frames, ignore_index=True)
    daily_all = pd.concat(daily_frames, ignore_index=True)

    # churn: drop transactions/days after the churn date
    churn = businesses.set_index("business_id")["churn_date"]
    for biz_id, close in churn.dropna().items():
        close_ts = pd.Timestamp(close)
        tx_all = tx_all[
            ~((tx_all["business_id"] == biz_id) & (pd.to_datetime(tx_all["occurred_at"]) > close_ts))
        ]
        daily_all = daily_all[
            ~((daily_all["business_id"] == biz_id) & (daily_all["date"] > close_ts))
        ]

    tx_all = inject_non_revenue(tx_all, businesses, dates, cfg, master_seed)
    tx_all = tx_all.sort_values(["business_id", "occurred_at", "ref"]).reset_index(drop=True)

    businesses.to_csv(out_dir / "businesses.csv", index=False)
    tx_all.to_csv(out_dir / "transactions.csv", index=False)
    daily_all.to_csv(out_dir / "daily.csv", index=False)

    metadata = {
        "created_utc": started.isoformat(),
        "generator_version": "v2.0",
        "master_seed": master_seed,
        "n_businesses": int(businesses["business_id"].nunique()),
        "n_transactions": int(len(tx_all)),
        "n_daily_rows": int(len(daily_all)),
        "date_range": [g["start_date"], g["end_date"]],
        "sectors": sectors,
        "min_amount_minor": g["min_amount_minor"],
        "min_amount_ksh": g["min_amount_minor"] / 100.0,
        "category_counts": tx_all["category"].value_counts().to_dict(),
        "disclaimer": (
            "Synthetic data for pipeline validation. Parameters are assumptions, "
            "NOT measured Kenyan MSME statistics."
        ),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    # summary printout
    print("Generated synthetic population")
    print(f"  businesses      : {metadata['n_businesses']}")
    print(f"  transactions    : {metadata['n_transactions']:,}")
    print(f"  daily rows      : {metadata['n_daily_rows']:,}")
    print(f"  min amount      : {metadata['min_amount_ksh']} KSh "
          f"({metadata['min_amount_minor']} minor units)")
    print("  category counts :", metadata["category_counts"])
    amt = tx_all["amount_minor"].astype(np.int64)
    amt_pos = amt[amt > 0]
    print(f"  amount_minor quantiles (positive): "
          f"{np.percentile(amt_pos, [1, 5, 50, 95, 99]).astype(int).tolist()}")
    n_low = int((amt_pos <= 100).sum())
    print(f"  transactions <= 1 KSh: {n_low}")


if __name__ == "__main__":
    main()
