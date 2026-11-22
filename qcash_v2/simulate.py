"""Repayment simulation for the Q-Cash v2 decision engine.

For each business-month in the test partitions, run the assessment at the
cut-off (using only data before it), then simulate collection from the actual
future eligible receipts:

    raw            = E_t * recovery_rate_bps / 10000
    daily_headroom = E_t * daily_cap_bps / 10000
    C_t            = min(raw, B_t, daily_headroom)     (fee first)

This is a *policy* simulation on synthetic data, not a lending result.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from common import DATA_DIR, OUTPUTS_DIR, load_config
from engine import compute_feature_row, decide, predict, reliability_score, repayment_terms
from splits import build_partitions


def simulate_collection(
    advance_minor: int,
    fee_minor: int,
    recovery_bps: int,
    daily_cap_bps: int,
    term_days: int,
    max_repayment_days: int,
    future_rev_minor: np.ndarray,
) -> dict:
    total_due = advance_minor + fee_minor
    balance = total_due
    fee_remaining = fee_minor
    collected = 0
    settled_day = None
    capped_days = 0
    for t in range(min(len(future_rev_minor), max_repayment_days)):
        if balance <= 0:
            break
        e = int(future_rev_minor[t])
        raw = e * recovery_bps // 10000
        day_cap = e * daily_cap_bps // 10000
        c = min(raw, balance, day_cap)
        if c == day_cap and raw > day_cap:
            capped_days += 1
        fee_comp = min(fee_remaining, c)
        fee_remaining -= fee_comp
        balance -= c
        collected += c
        if balance <= 0:
            settled_day = t + 1
            break

    if settled_day is not None and settled_day <= term_days:
        outcome = "settled"
    elif settled_day is not None:
        outcome = "extended"
    else:
        outcome = "fallback"

    return {
        "outcome": outcome,
        "days_to_settle": settled_day,
        "collected_minor": int(collected),
        "shortfall_minor": int(balance),
        "recovery_ratio": float(collected / total_due) if total_due else 0.0,
        "capped_days": capped_days,
        "total_due_minor": int(total_due),
    }


def run_simulation() -> pd.DataFrame:
    cfg = load_config()
    daily = pd.read_csv(DATA_DIR / "daily.csv", parse_dates=["date"])
    businesses = pd.read_csv(DATA_DIR / "businesses.csv")
    supervised = pd.read_csv(DATA_DIR / "supervised.csv", parse_dates=["as_of_date"])
    parts = build_partitions(supervised)

    sector_map = businesses.set_index("business_id")["sector"].to_dict()
    daily_by_biz = {b: g.sort_values("date") for b, g in daily.groupby("business_id")}

    rows: list[dict] = []
    for obs in [parts.test_temporal, parts.test_business]:
        for _, row in obs.iterrows():
            biz = row["business_id"]
            as_of = row["as_of_date"]
            sector = sector_map.get(biz, "general_retail")
            sub = daily_by_biz[biz]
            past = sub[sub["date"] < as_of]
            if len(past) < cfg["engine"]["min_history_days"]:
                continue
            feats = compute_feature_row(past, sector, cfg)
            point, lower, _ = predict(feats, sector, as_of=as_of)
            rel = reliability_score(feats, None)
            decision = decide(point, lower, rel, sector)
            if decision["status"] != "eligible":
                rows.append({"business_id": biz, "as_of_date": as_of, "sector": sector,
                             "status": decision["status"], "outcome": None,
                             "binding_constraint": decision["binding_constraint"],
                             "amount_minor": 0})
                continue
            terms = repayment_terms(rel["band"])
            future = sub[(sub["date"] > as_of) & (sub["date"] <= as_of + pd.Timedelta(days=60))]
            future_rev = future["eligible_revenue_minor"].to_numpy(dtype=np.int64)

            # normal scenario
            res = simulate_collection(
                decision["amount_minor"], decision["fee_minor"],
                terms["recovery_rate_bps"], terms["daily_cap_bps"],
                terms["term_days"], terms["max_repayment_days"], future_rev,
            )
            # stress scenario: revenue reduced by 30%
            stress_res = simulate_collection(
                decision["amount_minor"], decision["fee_minor"],
                terms["recovery_rate_bps"], terms["daily_cap_bps"],
                terms["term_days"], terms["max_repayment_days"],
                (future_rev * 0.7).astype(np.int64),
            )
            rows.append({
                "business_id": biz, "as_of_date": as_of, "sector": sector,
                "band": rel["band"], "status": decision["status"],
                "binding_constraint": decision["binding_constraint"],
                "amount_minor": decision["amount_minor"],
                "outcome": res["outcome"], "days_to_settle": res["days_to_settle"],
                "recovery_ratio": res["recovery_ratio"],
                "shortfall_minor": res["shortfall_minor"],
                "stress_outcome": stress_res["outcome"],
                "stress_recovery_ratio": stress_res["recovery_ratio"],
            })

    return pd.DataFrame(rows)


def main() -> None:
    cfg = load_config()
    df = run_simulation()
    df.to_csv(OUTPUTS_DIR / "simulation.csv", index=False)

    eligible = df[df["status"] == "eligible"]
    summary = {
        "n_assessed": int(len(df)),
        "n_eligible": int(len(eligible)),
        "n_ineligible": int((df["status"] != "eligible").sum()),
        "outcome_distribution": eligible["outcome"].value_counts().to_dict(),
        "median_days_to_settle": float(eligible["days_to_settle"].median()) if len(eligible) else None,
        "p10_days": float(eligible["days_to_settle"].quantile(0.10)) if len(eligible) else None,
        "p90_days": float(eligible["days_to_settle"].quantile(0.90)) if len(eligible) else None,
        "binding_constraint": eligible["binding_constraint"].value_counts().to_dict() if len(eligible) else {},
        "total_advanced_minor": int(eligible["amount_minor"].sum()),
        "total_shortfall_minor": int(eligible["shortfall_minor"].sum()),
        "loss_rate_pct": float(eligible["shortfall_minor"].sum() / max(1, eligible["amount_minor"].sum()) * 100),
        "stress_outcome_distribution": eligible["stress_outcome"].value_counts().to_dict(),
        "stress_loss_rate_pct": float(
            (eligible["amount_minor"].sum() * 0 - 0
             + (eligible["amount_minor"] * (1 - eligible["stress_recovery_ratio"])).sum())
            / max(1, eligible["amount_minor"].sum()) * 100
        ),
    }
    (OUTPUTS_DIR / "simulation_summary.json").write_text(json.dumps(summary, indent=2))

    print("Repayment simulation (synthetic):")
    print(f"  assessed={summary['n_assessed']}  eligible={summary['n_eligible']}")
    print("  outcomes:", summary["outcome_distribution"])
    print(f"  median days to settle: {summary['median_days_to_settle']}")
    print("  binding constraints:", summary["binding_constraint"])
    print(f"  total advanced: KSh {summary['total_advanced_minor']/100:,.0f}")
    print(f"  total shortfall: KSh {summary['total_shortfall_minor']/100:,.0f}")
    print(f"  loss rate: {summary['loss_rate_pct']:.2f}%")
    print("  stress (revenue -30%) outcomes:", summary["stress_outcome_distribution"])
    print(f"  stress loss rate: {summary['stress_loss_rate_pct']:.2f}%")


if __name__ == "__main__":
    main()
