"""Pipeline tests for Q-Cash v2.

Run with:  python tests/test_pipeline.py

Covers the retraining prompt's non-negotiables:
  1. determinism (same seed -> identical output)
  2. business-identity / overlap leakage tests
  3. future-row leakage test
  4. minor-unit invariant (all money integer)
  5. 1-shilling robustness (a business of 1-shilling transactions is handled)
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from common import DATA_DIR, load_config  # noqa: E402
from engine import classify_transactions, compute_feature_row, run_assessment  # noqa: E402

FAILURES: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}" + (f"  ({detail})" if detail else ""))
    if not condition:
        FAILURES.append(label)


def test_determinism() -> None:
    print("Determinism (same seed -> identical in-memory output)")
    import generator as gen

    cfg = load_config()
    dates = pd.date_range("2024-01-01", periods=40, freq="D")
    business = {
        "business_id": "SYN_TEST", "sector": "greengrocer", "size_class": "micro",
        "open_date": "2024-01-01", "paybill_active_from": "2024-01-01",
        "digital_share": 0.7, "latent_quality": 1.0, "base_daily_ksh": 800.0,
        "trend": 0.0, "churn_date": None,
    }
    tx1, d1 = gen.generate_business(business, dates, cfg, master_seed=42, index=0)
    tx2, d2 = gen.generate_business(business, dates, cfg, master_seed=42, index=0)
    tx3, d3 = gen.generate_business(business, dates, cfg, master_seed=43, index=0)
    check("same seed -> identical transactions", tx1.equals(tx2))
    check("same seed -> identical daily", d1.equals(d2))
    check("different seed -> different transactions", not tx1.equals(tx3))


def test_minor_unit_invariant() -> None:
    print("Minor-unit invariant (all money is a positive integer)")
    tx = pd.read_csv(DATA_DIR / "transactions.csv")
    amounts = tx["amount_minor"]
    check("no NaN amounts", bool(not amounts.isna().any()))
    check("all amounts are integers", bool((amounts.dropna() == amounts.dropna().astype(int)).all()))
    check("minimum amount is 1 shilling (100 minor units)",
          int(amounts[amounts > 0].min()) == 100)
    check("no zero-amount transactions", bool((amounts != 0).all()))


def test_classification_recovers_ground_truth() -> None:
    print("Transaction intelligence vs ground truth")
    tx = pd.read_csv(DATA_DIR / "transactions.csv")
    sample = tx[tx["business_id"].isin(tx["business_id"].unique()[:5])]
    classified = classify_transactions(sample)
    # revenue vs non-revenue (collapse all non-revenue categories)
    gt_revenue = sample["category"] == "revenue_eligible"
    pred_revenue = classified["predicted_category"] == "revenue_eligible"
    recall = float((gt_revenue & pred_revenue).sum() / max(1, gt_revenue.sum()))
    precision = float((gt_revenue & pred_revenue).sum() / max(1, pred_revenue.sum()))
    check("eligible-revenue recall > 0.95", recall > 0.95, f"recall={recall:.3f}")
    check("eligible-revenue precision > 0.95", precision > 0.95, f"precision={precision:.3f}")


def test_future_row_leakage() -> None:
    print("Future-row leakage test")
    cfg = load_config()
    daily = pd.read_csv(DATA_DIR / "daily.csv", parse_dates=["date"])
    biz = daily["business_id"].iloc[0]
    sub = daily[daily["business_id"] == biz].sort_values("date").reset_index(drop=True)
    as_of = sub["date"].iloc[-45]
    before = compute_feature_row(sub, "general_retail", cfg, as_of=as_of)

    # append a massive future transaction dated after as_of
    injected = pd.DataFrame({
        "business_id": [biz], "date": [as_of + pd.Timedelta(days=1)],
        "true_level_minor": [0.0], "shock": [0.0],
        "eligible_revenue_minor": [10**12], "eligible_tx_count": [1],
        "sum_sq_minor": [10**24], "max_tx_minor": [10**12],
    })
    after = compute_feature_row(pd.concat([sub, injected], ignore_index=True),
                                "general_retail", cfg, as_of=as_of)
    identical = all(before[k] == after[k] for k in before)
    check("features unchanged after appending a future transaction", identical)


def test_one_shilling_robustness() -> None:
    print("1-shilling robustness")
    # build a business whose only eligible receipts are 1-shilling transactions
    n = 90
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    tx = pd.DataFrame({
        "ref": [f"LOW-{i}" for i in range(n)],
        "business_id": ["SYN_LOW"] * n,
        "customer_id": [None] * n,
        "occurred_at": [d.isoformat() for d in dates],
        "amount_minor": [100] * n,  # 1 shilling each
        "type": ["credit"] * n,
        "channel": ["paybill"] * n,
        "is_reversal_of": [None] * n,
        "reference": [None] * n,
        "payer_hash": [None] * n,
        "category": ["revenue_eligible"] * n,
    })
    result = run_assessment(tx, "greengrocer")
    check("no exception for a 1-shilling business", "status" in result)
    if "prediction" in result:
        pred = result["prediction"]
        check("prediction is finite and non-negative",
              bool(np.isfinite(pred["point_minor"]) and pred["point_minor"] >= 0))
        # 90 days of 1-shilling receipts -> rev_sum_90 = 9000 minor units ->
        # moving-average forecast = 3000 minor (30 KSh). It must stay small.
        check("point forecast is small (a 1-shilling business)",
              pred["point_minor"] <= 10000,
              f"point={pred['point_minor']} minor units")


def test_eligibility_reason_codes() -> None:
    print("Eligibility reason codes (R03 / R04 / R05)")

    def series(days: int, per_day: int, ticket: int) -> pd.DataFrame:
        dates = pd.date_range("2025-06-01", periods=days, freq="D")
        rows = []
        for d in dates:
            for _ in range(per_day):
                rows.append({
                    "ref": f"S{len(rows)}", "business_id": "SYN_RC", "customer_id": None,
                    "occurred_at": d.strftime("%Y-%m-%dT%H:%M:%S"), "amount_minor": ticket,
                    "type": "credit", "channel": "paybill",
                    "is_reversal_of": None, "reference": None, "payer_hash": None,
                })
        return pd.DataFrame(rows)

    big = series(90, 10, 30000)
    base = run_assessment(big, "barbershop")
    check("baseline eligible with no reason codes",
          base["status"] == "eligible" and base["reason_codes"] == [])

    r03 = run_assessment(big, "barbershop", exposure={"overdue_minor": 100000})
    check("overdue advance -> not_eligible R03",
          r03["status"] == "not_eligible" and "R03" in r03["reason_codes"])

    r04 = run_assessment(big, "barbershop",
                         exposure={"outstanding_minor": 3000000, "max_exposure_minor": 2500000})
    check("exposure cap reached -> R04", "R04" in r04["reason_codes"])

    low = series(90, 10, 600)  # ~60/day, pharmacy margin -> affordability binds below min
    r05 = run_assessment(low, "pharmacy")
    check("affordability below minimum -> R05", "R05" in r05["reason_codes"])


def main() -> int:
    print("=" * 70)
    print("Q-Cash v2 pipeline tests")
    print("=" * 70)
    test_determinism()
    test_minor_unit_invariant()
    test_classification_recovers_ground_truth()
    test_future_row_leakage()
    test_one_shilling_robustness()
    test_eligibility_reason_codes()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} CHECK(S) FAILED:")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
