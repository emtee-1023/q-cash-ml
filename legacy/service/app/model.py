"""Loads the trained synthetic Q-Cash model and turns inputs into forecasts.

The artifact is a joblib bundle written by ``run_experiment.py``. It contains
the fitted estimator, the feature columns the estimator expects, imputation
medians, the limit-engine constants and provenance metadata.

This module hides all of that behind two entry points: :func:`predict_revenue`
(which takes an already-computed feature row) and :func:`limits_for` (which
turns a forecast + volatility into the four early-access limit strategies).
Feature engineering itself lives in :mod:`.features`.
"""

from __future__ import annotations

import threading
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .config import MODEL_PATH

_bundle: dict[str, Any] | None = None
_lock = threading.Lock()


def get_bundle() -> dict[str, Any]:
    """Load the artifact once, on first use, and cache it in memory."""
    global _bundle
    if _bundle is None:
        with _lock:
            if _bundle is None:
                if not MODEL_PATH.exists():
                    raise RuntimeError(
                        f"Model artifact not found at {MODEL_PATH}. Set MODEL_PATH to "
                        "the service_model_bundle.joblib file produced by run_experiment.py."
                    )
                _bundle = joblib.load(MODEL_PATH)
    return _bundle


def _model_label() -> str:
    return str(get_bundle().get("model_label", "unknown"))


def _error_band_kes() -> float:
    band = get_bundle().get("error_band_kes")
    return float(band) if isinstance(band, (int, float)) else 0.0


def predict_revenue(features: pd.DataFrame) -> np.ndarray:
    """Return the next-30-day revenue forecast (KSh) for each feature row."""
    bundle = get_bundle()
    columns = bundle["feature_columns"]
    missing = [c for c in columns if c not in features.columns]
    if missing:
        raise ValueError(
            "Input is missing required feature columns: " + ", ".join(sorted(missing))
        )
    X = features[columns].apply(pd.to_numeric, errors="coerce")
    medians = bundle.get("imputation_medians", {})
    if medians:
        X = X.fillna(pd.Series(medians))
    raw = bundle["estimator"].predict(X.to_numpy(dtype=float))
    predictions = np.nan_to_num(np.asarray(raw, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(predictions, 0.0, None)


def _stability_factor(cv: float) -> float:
    return float(np.clip(1.0 - cv, 0.35, 1.0))


def limits_for(
    prediction: float, cv_daily_revenue_30d: float, median_daily_revenue_30d: float
) -> dict[str, float]:
    """The four transparent early-access limit strategies, in KSh.

    Limits are sized to days of predicted *daily* revenue (short-term bridge
    loans) with a soft cap:

    * simple_revenue:       daily * cover_days.
    * conservative:         median daily revenue * cover_days.
    * revenue_stability:    daily * cover_days * stability_factor(cv).
    * cashflow_coverage:    daily * cover_days * repayment_rate.
    """
    bundle = get_bundle()
    c = bundle.get("limit_constants", {})
    cover_days = float(c.get("cover_days", 5.0))
    repayment_rate = float(c.get("repayment_rate", 0.5))
    soft_cap = float(c.get("soft_cap_kes", 15000.0))

    prediction = max(0.0, float(prediction))
    daily = prediction / 30.0

    def capped(value: float) -> float:
        return round(min(value, soft_cap), 2)

    return {
        "simple_revenue_kes": capped(daily * cover_days),
        "conservative_kes": capped(float(median_daily_revenue_30d) * cover_days),
        "revenue_stability_kes": capped(
            daily * cover_days * _stability_factor(cv_daily_revenue_30d)
        ),
        "cashflow_coverage_kes": capped(daily * cover_days * repayment_rate),
    }


def _risk_tier(cv: float) -> str:
    if cv < 0.3:
        return "low"
    if cv < 0.6:
        return "medium"
    return "high"


def decision_for(
    prediction: float, cv_daily_revenue_30d: float, median_daily_revenue_30d: float
) -> dict[str, Any]:
    """A single recommended borrowing amount with its reasoning.

    The recommendation is the most conservative of the three coverage-aware
    strategies (conservative, revenue+stability, cash-flow coverage). The naive
    days-of-revenue rule is shown for comparison but never drives the
    recommendation.
    """
    limits = limits_for(prediction, cv_daily_revenue_30d, median_daily_revenue_30d)
    candidates = {
        "conservative": limits["conservative_kes"],
        "revenue_stability": limits["revenue_stability_kes"],
        "cashflow_coverage": limits["cashflow_coverage_kes"],
    }
    binding = min(candidates, key=candidates.get)
    recommended = candidates[binding]
    tier = _risk_tier(cv_daily_revenue_30d)
    daily = float(prediction) / 30.0
    reasoning = (
        f"Recommended amount is the most conservative of the coverage-aware "
        f"strategies ({binding}). Based on predicted daily revenue of KSh "
        f"{daily:,.0f} (30-day forecast KSh {float(prediction):,.0f}) with CV "
        f"{cv_daily_revenue_30d:.2f} (risk tier: {tier})."
    )
    return {
        "recommended_available_kes": round(recommended, 2),
        "binding_strategy": binding,
        "risk_tier": tier,
        "limits": limits,
        "reasoning": reasoning,
    }


def model_info() -> dict[str, Any]:
    """A JSON-safe view of the artifact metadata (the estimator is excluded)."""
    bundle = get_bundle()
    info = {
        key: value
        for key, value in bundle.items()
        if key not in {"estimator", "imputation_medians"}
    }
    info["model_label"] = _model_label()
    return info
