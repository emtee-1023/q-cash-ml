"""Q-Cash ERA decision engine (minor-unit integer arithmetic throughout).

Implements the stages of the decision-engine guide:

  Stage 1  transaction intelligence (classification -> eligible revenue)
  Stage 2  feature construction (delegated to :mod:`features`)
  Stage 3  revenue prediction (loads the trained model artifact)
  Stage 4  business reliability score (six weighted components)
  Stage 5  ERA decision (advance rate -> affordability -> final amount)
  Stage 6  repayment terms (recovery rate, daily cap, fallback ladder)

All money is integer minor units; all rates are integer basis points.
"""

from __future__ import annotations

from typing import Any

import joblib
import numpy as np
import pandas as pd

from common import OUTPUTS_DIR, load_config
from model_features import MA_DIVISOR, enhanced_frame

CAPITAL_KEYWORDS = ("loan", "grant", "sacco", "capital")
OWN_TRANSFER_KEYWORDS = ("own_transfer", "transfer", "internal")

RELIABILITY_WEIGHTS_BPS = {
    "consistency": 2500, "volatility": 2000, "activity": 2000,
    "trend": 1000, "concentration": 1500, "repayment": 1000,
}


# ---------------------------------------------------------------------------
# Stage 1 — transaction intelligence
# ---------------------------------------------------------------------------

def classify_transactions(tx: pd.DataFrame) -> pd.DataFrame:
    """Classify each transaction and keep the raw record untouched.

    Returns a copy with a ``predicted_category`` column. The order of the rules
    matters (guide section 3.2).
    """
    out = tx.copy()
    refs = out["ref"].astype(str)
    dup = refs.duplicated(keep="first")

    def category(row: pd.Series) -> str:
        if row["type"] == "debit":
            if pd.notna(row["is_reversal_of"]):
                return "non_revenue_reversal"
            return "non_revenue_refund"
        # credit
        if dup[row.name]:
            return "duplicate"
        if pd.notna(row["reference"]) and any(
            k in str(row["reference"]).lower() for k in CAPITAL_KEYWORDS
        ):
            return "non_revenue_capital"
        if pd.notna(row["reference"]) and any(
            k in str(row["reference"]).lower() for k in OWN_TRANSFER_KEYWORDS
        ):
            return "non_revenue_own_transfer"
        if pd.notna(row["payer_hash"]) and str(row["payer_hash"]).startswith("own-"):
            return "non_revenue_own_transfer"
        return "revenue_eligible"

    out["predicted_category"] = out.apply(category, axis=1)

    # anomaly detection: robust z-score > 3.5 on trailing amounts -> ambiguous
    out["_amount_abs"] = out["amount_minor"].abs()
    credits = out[out["type"] == "credit"]["_amount_abs"]
    if len(credits) >= 10:
        med = credits.median()
        mad = (credits - med).abs().median()
        scale = mad if mad > 0 else credits.quantile(0.75) - credits.quantile(0.25)
        if scale > 0:
            z = 0.6745 * (out["_amount_abs"] - med) / scale
            ambiguous = (out["type"] == "credit") & (z > 3.5)
            out.loc[ambiguous, "predicted_category"] = "ambiguous"
    out = out.drop(columns=["_amount_abs"])
    return out


def eligible_daily(tx: pd.DataFrame) -> pd.DataFrame:
    """Aggregate eligible revenue into per-day minor-unit totals + HHI inputs."""
    elig = tx[tx["predicted_category"] == "revenue_eligible"].copy()
    if elig.empty:
        return pd.DataFrame(columns=["date", "eligible_revenue_minor", "eligible_tx_count",
                                     "sum_sq_minor", "max_tx_minor"])
    elig["date"] = pd.to_datetime(elig["occurred_at"]).dt.normalize()
    elig["amt"] = elig["amount_minor"].astype(np.int64)
    daily = elig.groupby("date", as_index=False).agg(
        eligible_revenue_minor=("amt", "sum"),
        eligible_tx_count=("amt", "count"),
        sum_sq_minor=("amt", lambda s: int((s.astype(np.int64) ** 2).sum())),
        max_tx_minor=("amt", "max"),
    )
    return daily.sort_values("date").reset_index(drop=True)


def _calendar_daily(daily: pd.DataFrame) -> pd.DataFrame:
    """Reindex eligible daily aggregates to a continuous calendar.

    Zero-revenue days are inserted (as the training pipeline does and the engine
    guide requires) so volatility/activity features and the observed-day count
    are measured over calendar time, not only over active days.
    """
    if daily.empty:
        return daily
    df = daily.copy()
    df["date"] = pd.to_datetime(df["date"])
    idx = pd.date_range(df["date"].min(), df["date"].max(), freq="D", name="date")
    out = df.set_index("date").reindex(idx)
    for col in ("eligible_revenue_minor", "eligible_tx_count", "sum_sq_minor", "max_tx_minor"):
        out[col] = out[col].fillna(0)
    return out.reset_index()


# ---------------------------------------------------------------------------
# Stage 2 + 3 — features and prediction
# ---------------------------------------------------------------------------

def compute_feature_row(
    daily: pd.DataFrame, sector: str, cfg: dict, as_of: pd.Timestamp | None = None
) -> dict[str, float]:
    """Compute one feature row (minor units) from eligible daily aggregates.

    ``as_of`` is the information cut-off; rows at or after it are excluded. If
    not given, the day after the latest row is used.
    """
    daily = daily.sort_values("date").reset_index(drop=True)
    dates = daily["date"].to_numpy(dtype="datetime64[D]")
    rev = daily["eligible_revenue_minor"].to_numpy(dtype=np.float64)
    tx = daily["eligible_tx_count"].to_numpy(dtype=np.float64)
    sum_sq = daily["sum_sq_minor"].to_numpy(dtype=np.float64)
    max_tx = daily["max_tx_minor"].to_numpy(dtype=np.float64)

    if as_of is None:
        cut = dates[-1] + np.timedelta64(1, "D")
    else:
        cut = np.datetime64(pd.Timestamp(as_of).to_pydatetime().date())
    hi = int(np.searchsorted(dates, cut, side="left"))

    def wsum(days: int) -> float:
        lo = int(np.searchsorted(dates, cut - np.timedelta64(days, "D"), "left"))
        return float(rev[lo:hi].sum()) if lo < hi else 0.0

    lo30 = int(np.searchsorted(dates, cut - np.timedelta64(30, "D"), "left"))
    win_rev = rev[lo30:hi]
    win_tx = tx[lo30:hi]
    rev_30 = float(win_rev.sum())
    tx_30 = float(win_tx.sum())
    active_days = float((win_rev > 0).sum())
    window_days = max(1, hi - lo30)  # calendar days actually observed in the window

    mean_txn = (rev_30 / tx_30) if tx_30 > 0 else 0.0
    mean_daily = float(win_rev.mean()) if win_rev.size else 0.0
    cv = (float(win_rev.std()) / mean_daily) if mean_daily > 0 else 0.0

    lo14 = int(np.searchsorted(dates, cut - np.timedelta64(14, "D"), "left"))
    lo28 = int(np.searchsorted(dates, cut - np.timedelta64(28, "D"), "left"))
    mean_14 = float(rev[lo14:hi].mean()) if lo14 < hi else 0.0
    mean_prior = float(rev[lo28:lo14].mean()) if lo28 < lo14 else 0.0
    trend_pct = (mean_14 - mean_prior) / mean_prior * 100.0 if mean_prior > 0 else 0.0
    trend_pct = float(np.clip(trend_pct, -100.0, 100.0))

    sq_30 = float(sum_sq[lo30:hi].sum())
    hhi = (sq_30 / (rev_30 * rev_30)) if rev_30 > 0 else 1.0
    hhi = float(np.clip(hhi, 0.0, 1.0))
    top1 = float(max_tx[lo30:hi].max()) if lo30 < hi else 0.0
    top1_share = (top1 / rev_30) if rev_30 > 0 else 0.0

    weekdays = pd.DatetimeIndex(dates[lo30:hi]).isocalendar().week.to_numpy()
    active_weeks = len(np.unique(weekdays[win_rev > 0])) if lo30 < hi else 0
    freq_per_active_week = (tx_30 / active_weeks) if active_weeks > 0 else 0.0

    first_receipt = int(np.argmax(rev > 0)) if (rev > 0).any() else 0
    tenure = float((dates[hi - 1] - dates[first_receipt]).astype(int) + 1) if hi > 0 else 0.0

    return {
        "rev_sum_7": wsum(7),
        "rev_sum_30": rev_30,
        "rev_sum_90": wsum(90),
        "txn_count_30": tx_30,
        "freq_per_active_week": freq_per_active_week,
        "active_days_30": active_days,
        "adr_pct": (active_days / window_days * 100.0),
        "mean_txn_minor": mean_txn,
        "cv_daily_rev_30": cv,
        "trend_pct": trend_pct,
        "hhi_30": hhi,
        "top1_share_30": top1_share,
        "tenure_days": tenure,
    }


def model_path():
    """Path to the trained artifact, overridable via QCASH_MODEL_PATH (e.g. in Docker)."""
    import os
    env = os.getenv("QCASH_MODEL_PATH")
    from pathlib import Path as _Path
    return _Path(env) if env else (OUTPUTS_DIR / "model.joblib")


def predict(
    point_features: dict[str, float],
    sector: str,
    as_of: pd.Timestamp | None = None,
) -> tuple[float, float, dict]:
    """Return (point_minor, lower_minor, bundle) using the trained artifact."""
    bundle = joblib.load(model_path())
    mtype = bundle.get("model_type")
    if mtype == "moving_average_90d":
        col = bundle["required_feature"]
        point = float(point_features.get(col, 0.0)) / float(bundle["divisor"])
    elif mtype == "xgboost_ratio_ma90":
        as_of_ts = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp("today")
        row = {**point_features, "sector": sector, "as_of_date": as_of_ts}
        df1 = pd.DataFrame([row])
        X = enhanced_frame(df1).reindex(columns=bundle["enh_columns"]).fillna(0.0).to_numpy(dtype=float)
        ma = float(point_features.get("rev_sum_90", 0.0)) / float(bundle["ma_divisor"])
        ratios = [float(np.clip(xgb.predict(X)[0], 0.0, None)) for xgb in bundle["estimators"]]
        point = ma * float(np.mean(ratios))
    else:
        cols = bundle["feature_columns"]
        row = {c: point_features.get(c, 0.0) for c in cols if not c.startswith("sector_")}
        for c in cols:
            if c.startswith("sector_"):
                row[c] = 1.0 if c == f"sector_{sector}" else 0.0
        X = pd.DataFrame([row]).reindex(columns=cols).fillna(0.0).to_numpy(dtype=float)
        point = float(bundle["estimator"].predict(X)[0])
    point = float(np.clip(point, 0.0, None))
    cv_error = float(bundle.get("cv_error", 0.0))
    k = float(bundle.get("error_margin_k", 1.0))
    lower = max(0.0, point * (1.0 - k * cv_error))
    return point, lower, bundle


# ---------------------------------------------------------------------------
# Stage 4 — reliability score
# ---------------------------------------------------------------------------

def reliability_score(features: dict[str, float], repayment: dict[str, Any] | None) -> dict:
    cfg = load_config()["engine"]
    weights = cfg["reliability_weights_bps"]

    # consistency: share of weeks with revenue >= 50% of median week (proxy: ADR)
    adr = float(features.get("adr_pct", 0.0))
    consistency = float(np.clip(adr, 0.0, 100.0))

    cv = float(features.get("cv_daily_rev_30", 0.0))
    volatility = 100.0 * max(0.0, 1.0 - cv / 1.0)

    activity = float(np.clip(features.get("adr_pct", 0.0), 0.0, 100.0))

    trend = float(np.clip(50.0 + features.get("trend_pct", 0.0), 0.0, 100.0))

    hhi = float(features.get("hhi_30", 1.0))
    n_tx = float(features.get("txn_count_30", 1.0))
    if n_tx > 1:
        hhi_norm = (hhi - 1.0 / n_tx) / (1.0 - 1.0 / n_tx)
    else:
        hhi_norm = 1.0
    concentration = 100.0 * (1.0 - float(np.clip(hhi_norm, 0.0, 1.0)))

    components = {
        "consistency": {"score": consistency, "weight_bps": weights["consistency"], "used": True},
        "volatility": {"score": volatility, "weight_bps": weights["volatility"], "used": True},
        "activity": {"score": activity, "weight_bps": weights["activity"], "used": True},
        "trend": {"score": trend, "weight_bps": weights["trend"], "used": True},
        "concentration": {"score": concentration, "weight_bps": weights["concentration"], "used": True},
    }

    # repayment component: unavailable unless >= 2 settled advances
    if repayment and int(repayment.get("settled_count", 0) or 0) >= 2:
        on_time = int(repayment.get("on_time_count", 0))
        settled = int(repayment.get("settled_count", 1))
        components["repayment"] = {
            "score": 100.0 * on_time / settled,
            "weight_bps": weights["repayment"], "used": True,
        }
    else:
        components["repayment"] = {"score": None, "weight_bps": weights["repayment"], "used": False}

    avail = {k: v for k, v in components.items() if v["used"]}
    total_bps = sum(v["weight_bps"] for v in avail.values())
    score = sum(v["weight_bps"] * v["score"] for v in avail.values()) / total_bps if total_bps else 0.0
    band = _band(score, cfg["band_cutoffs"])

    return {
        "score": round(float(score), 2),
        "band": band,
        "components": components,
        "weights_renormalised": bool(repayment and int(repayment.get("settled_count", 0) or 0) < 2),
    }


def _band(score: float, cutoffs: dict[str, int]) -> str:
    if score >= cutoffs["A"]:
        return "A"
    if score >= cutoffs["B"]:
        return "B"
    if score >= cutoffs["C"]:
        return "C"
    return "D"


def _history_tier(observed_days: int, tx_30: int, cfg: dict) -> str:
    """Pick the cold-start tier from the length of observed history."""
    tiers = cfg["engine"]["history_tiers"]
    tier = "starter"
    for name in ("starter", "limited", "standard", "full"):
        if observed_days >= tiers[name]["min_days"]:
            tier = name
    # The full 90-day-anchored model also needs enough recent transactions.
    if tier == "full" and tx_30 < cfg["engine"]["full_model_min_txn_30"]:
        tier = "standard"
    return tier


def _run_rate_forecast(daily: pd.DataFrame, observed_days: int) -> float:
    """Conservative run-rate forecast: mean calendar-day eligible revenue x 30.

    Used for short history (<90 days) where the 90-day-anchored model cannot be
    applied. Includes zero-revenue days, so it is deliberately conservative.
    """
    total = float(daily["eligible_revenue_minor"].sum())
    return (total / max(1, observed_days)) * 30.0


# ---------------------------------------------------------------------------
# Stage 5 — decision
# ---------------------------------------------------------------------------

def decide(
    point_minor: float,
    lower_minor: float,
    reliability: dict,
    sector: str,
    exposure: dict[str, Any] | None = None,
    business_profile: dict[str, Any] | None = None,
    advance_rate_bps: int | None = None,
    max_advance_minor: int | None = None,
) -> dict:
    cfg = load_config()["engine"]
    band = reliability["band"]
    if advance_rate_bps is None:
        advance_rate_bps = cfg["advance_rate_bps"][band]
    margin_bps = cfg["margin_factor_bps"].get(sector, 3500)
    fee_rate_bps = cfg["fee_rate_bps"]
    affordability_share_bps = cfg["affordability_share_bps"]
    term_days = cfg["term_days"]
    max_adv = cfg["max_advance_minor"] if max_advance_minor is None else int(max_advance_minor)
    min_adv = cfg["min_advance_minor"]

    exposure = exposure or {}
    outstanding = int(exposure.get("outstanding_minor", 0) or 0)
    max_exposure = int(exposure.get("max_exposure_minor", max_adv))
    owner_outstanding = int(exposure.get("owner_outstanding_minor", 0) or 0)

    a_rate = lower_minor * advance_rate_bps // 10000

    # affordability: capacity = lower * margin; then share, term, fee
    capacity = lower_minor * margin_bps // 10000
    a_affordable = (
        capacity * affordability_share_bps // 10000
        * term_days // 30
        * 10000 // (10000 + fee_rate_bps)
    )

    headroom = max(0, max_exposure - outstanding)
    owner_headroom = max(0, max_exposure - owner_outstanding)

    candidates = {
        "rate": a_rate,
        "policy_max": max_adv,
        "affordability": a_affordable,
        "exposure": headroom,
        "owner_exposure": owner_headroom,
    }
    binding = min(candidates, key=candidates.get)
    a_final = min(candidates.values())
    a_final = int(a_final // 5000 * 5000)  # round DOWN to KSh 50

    reason_codes: list[str] = []
    if a_final < min_adv:
        return {"status": "not_eligible", "reason_codes": ["R02"],
                "binding_constraint": binding, "amount_minor": 0,
                "advance_rate_bps": advance_rate_bps}

    fee = a_final * fee_rate_bps // 10000
    return {
        "status": "eligible",
        "amount_minor": a_final,
        "advance_rate_bps": advance_rate_bps,
        "binding_constraint": binding,
        "reason_codes": reason_codes,
        "fee_minor": int(fee),
        "total_repayable_minor": int(a_final + fee),
        "affordability_capacity_minor": int(capacity),
    }


# ---------------------------------------------------------------------------
# Stage 6 — repayment terms
# ---------------------------------------------------------------------------

def repayment_terms(band: str) -> dict:
    cfg = load_config()["engine"]
    return {
        "term_days": cfg["term_days"],
        "max_repayment_days": cfg["max_repayment_days"],
        "recovery_rate_bps": cfg["recovery_rate_bps"][band],
        "daily_cap_bps": cfg["daily_cap_bps"],
        "fallback": "extend_once_then_partner_review",
    }


# ---------------------------------------------------------------------------
# full assessment
# ---------------------------------------------------------------------------

def run_assessment(
    transactions: pd.DataFrame,
    sector: str,
    repayment: dict[str, Any] | None = None,
    exposure: dict[str, Any] | None = None,
    business_profile: dict[str, Any] | None = None,
) -> dict:
    cfg = load_config()
    classified = classify_transactions(transactions)
    daily = _calendar_daily(eligible_daily(classified))

    counts = classified["predicted_category"].value_counts().to_dict()
    anomalies = int(counts.get("ambiguous", 0))

    observed_days = int(len(daily))
    total_eligible_tx = int(daily["eligible_tx_count"].sum()) if observed_days else 0
    if observed_days:
        cut = pd.Timestamp(daily["date"].max()) + pd.Timedelta(days=1)
        last30 = daily[daily["date"] >= cut - pd.Timedelta(days=30)]
        tx_30 = int(last30["eligible_tx_count"].sum())
    else:
        cut = pd.Timestamp("today")
        tx_30 = 0

    # Absolute cold-start floor (KopoKopo-style): 14 observed calendar days and
    # at least `min_txn_window` eligible transactions. Below this -> no offer.
    if observed_days < cfg["engine"]["min_history_days"] or total_eligible_tx < cfg["engine"]["min_txn_window"]:
        return {"status": "insufficient_data", "reason_codes": ["R01"],
                "detail": {"observed_days": observed_days,
                           "required_days": cfg["engine"]["min_history_days"],
                           "eligible_tx": total_eligible_tx,
                           "required_txn": cfg["engine"]["min_txn_window"]}}

    tier = _history_tier(observed_days, tx_30, cfg)
    tier_cfg = cfg["engine"]["history_tiers"][tier]
    feats = compute_feature_row(daily, sector, cfg)
    rel = reliability_score(feats, repayment)

    if tier == "full":
        point, lower, bundle = predict(feats, sector, as_of=cut)
        model_version = bundle.get("model_version")
        forecast_method = "xgboost_ratio_ma90"
        decision = decide(point, lower, rel, sector, exposure, business_profile)
    else:
        # Short history: conservative run-rate forecast + tier haircut + small rate/cap.
        point = _run_rate_forecast(daily, observed_days)
        margin = float(tier_cfg["margin"])
        lower = max(0.0, point * (1.0 - margin))
        model_version = f"run_rate_30d[{tier}]"
        forecast_method = "run_rate_30d"
        decision = decide(
            point, lower, rel, sector, exposure, business_profile,
            advance_rate_bps=int(tier_cfg["advance_rate_bps"]),
            max_advance_minor=int(tier_cfg["max_advance_minor"]),
        )
    terms = repayment_terms(rel["band"])

    return {
        "status": decision["status"],
        "model_version": model_version,
        "history_tier": tier,
        "observed_days": observed_days,
        "intelligence": {
            "counts": {k: int(v) for k, v in counts.items()},
            "value_minor": {"eligible": int(daily["eligible_revenue_minor"].sum()) if len(daily) else 0},
            "anomalies_flagged": anomalies,
        },
        "prediction": {
            "horizon_days": 30,
            "point_minor": int(point),
            "lower_bound_minor": int(lower),
            "method": forecast_method,
        },
        "reliability": rel,
        "recommendation": decision,
        "terms": terms,
        "reason_codes": decision["reason_codes"],
    }


if __name__ == "__main__":
    # sanity: assess one business from the synthetic data
    tx = pd.read_csv("data/transactions.csv")
    businesses = pd.read_csv("data/businesses.csv")
    biz = businesses["business_id"].iloc[0]
    sector = businesses[businesses["business_id"] == biz]["sector"].iloc[0]
    sub = tx[tx["business_id"] == biz]
    result = run_assessment(sub, sector)
    import json
    print(json.dumps(result, indent=2, default=str))
