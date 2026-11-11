"""Business reliability scoring for Q-Cash ERA.

The score is a **weighted, rules-based** composite (not a second Random Forest
classifier). It combines configurable component scores so it stays explainable:

    S = sum( w_i * s_i )

where ``w_i`` are configurable weights and ``s_i`` are normalized component
scores in [0, 100]. Component weights and a few normalization reference points
are configurable via the policy document; the normalization scheme itself is
explicit and documented per component.

Important properties:

* the score is **not** a calibrated probability of default;
* a new business is **not** penalised for having no prior Q-Cash repayment
  history — that component is marked unavailable and the remaining weights are
  renormalised, which keeps "unavailable" distinct from "poor";
* every component records its raw value, normalized score, weight and reason
  code so the result is fully explainable.

All weights and thresholds are provisional pending empirical calibration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .policy import Policy


@dataclass
class ScoreComponent:
    name: str
    raw_value: float
    normalized_score: float
    weight: float
    available: bool
    flag: str
    reason_code: str
    description: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "raw_value": round(self.raw_value, 6),
            "normalized_score": round(self.normalized_score, 2),
            "weight": self.weight,
            "available": self.available,
            "flag": self.flag,
            "reason_code": self.reason_code,
            "description": self.description,
        }


@dataclass
class ReliabilityResult:
    score: float
    band: str
    components: list[ScoreComponent]
    reason_codes: list[str]
    unavailable_components: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": round(self.score, 2),
            "band": self.band,
            "components": [c.to_dict() for c in self.components],
            "reason_codes": list(self.reason_codes),
            "unavailable_components": list(self.unavailable_components),
        }


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _flag(value: float, good: float, poor: float) -> str:
    if value >= good:
        return "good"
    if value < poor:
        return "poor"
    return "neutral"


def score_component(
    name: str,
    raw: float,
    score01: float,
    good: float,
    poor: float,
    good_code: str,
    poor_code: str,
    description: str,
) -> ScoreComponent:
    norm = 100.0 * _clip01(score01)
    flag = _flag(norm, good, poor)
    code = good_code if flag == "good" else poor_code if flag == "poor" else ""
    return ScoreComponent(
        name=name,
        raw_value=raw,
        normalized_score=norm,
        weight=0.0,
        available=True,
        flag=flag,
        reason_code=code,
        description=description,
    )


def _repayment_component(repayment: dict[str, Any] | None) -> ScoreComponent:
    if not repayment or int(repayment.get("advances_count", 0) or 0) <= 0:
        return ScoreComponent(
            name="repayment_history",
            raw_value=0.0,
            normalized_score=0.0,
            weight=0.0,
            available=False,
            flag="neutral",
            reason_code="NO_PRIOR_REPAYMENT_HISTORY",
            description="No previous Q-Cash repayment history; treated as neutral, not penalised.",
        )
    advances = int(repayment.get("advances_count", 0))
    on_time = int(repayment.get("on_time_count", 0))
    late = int(repayment.get("late_count", 0))
    default = int(repayment.get("default_count", 0))
    on_time_rate = on_time / advances if advances else 0.0
    raw = on_time_rate
    norm = 100.0 * _clip01(on_time_rate)
    flag = "good" if on_time_rate >= 0.8 else "poor" if on_time_rate < 0.5 else "neutral"
    code = (
        "GOOD_REPAYMENT_HISTORY" if flag == "good"
        else "POOR_REPAYMENT_HISTORY" if flag == "poor" else ""
    )
    return ScoreComponent(
        name="repayment_history",
        raw_value=raw,
        normalized_score=norm,
        weight=0.0,
        available=True,
        flag=flag,
        reason_code=code,
        description=f"{on_time}/{advances} on-time, {late} late, {default} default.",
    )


class ReliabilityScorer:
    """Compute the weighted reliability score from transaction analytics."""

    def __init__(self, policy: Policy) -> None:
        self.policy = policy

    def score(self, analytics: dict[str, float], repayment: dict[str, Any] | None) -> ReliabilityResult:
        sc = self.policy.scoring
        active_rate = float(analytics.get("active_day_rate_30d", 0.0))
        cv = float(analytics.get("cv_daily_revenue_30d", 0.0))
        freq = float(analytics.get("avg_tx_per_active_day_30d", 0.0))
        active_days = float(analytics.get("active_days_30d", 0.0))
        trend = float(analytics.get("revenue_trend_30_vs_90", 0.0))
        hhi = float(analytics.get("transaction_concentration_30d", 1.0))

        components = [
            score_component(
                "revenue_consistency", active_rate, active_rate, 70, 40,
                "CONSISTENT_REVENUE", "IRREGULAR_REVENUE",
                "Share of days with eligible revenue in the last 30 days.",
            ),
            score_component(
                "revenue_volatility", cv, 1.0 - cv / sc.volatility_cv_ceiling, 70, 40,
                "ACCEPTABLE_VOLATILITY", "HIGH_VOLATILITY",
                "Inverse of the coefficient of variation of daily eligible revenue.",
            ),
            score_component(
                "transaction_frequency", freq, freq / sc.frequency_ref_per_active_day, 60, 30,
                "HEALTHY_TRANSACTION_FREQUENCY", "LOW_TRANSACTION_FREQUENCY",
                "Average eligible transactions per active day.",
            ),
            score_component(
                "active_days", active_days, active_days / 30.0, 70, 40,
                "SUFFICIENT_ACTIVE_DAYS", "INSUFFICIENT_ACTIVE_DAYS",
                "Number of active business days in the last 30 days.",
            ),
            score_component(
                "revenue_trend", trend, 0.5 + 0.5 * (trend / sc.trend_ref), 60, 40,
                "GROWING_REVENUE", "DECLINING_REVENUE",
                "Recent revenue trend relative to the longer-term average.",
            ),
            score_component(
                "transaction_concentration", hhi, 1.0 - hhi, 60, 30,
                "DIVERSIFIED_REVENUE", "CONCENTRATED_REVENUE",
                "Inverse of the Herfindahl concentration of eligible transactions.",
            ),
            _repayment_component(repayment),
        ]

        weights = sc.weights
        available = [c for c in components if c.available]
        for component in available:
            component.weight = weights.get(component.name, 0.0)
        total_weight = sum(c.weight for c in available)
        if total_weight <= 0:
            equal = 1.0 / len(available) if available else 0.0
            for component in available:
                component.weight = equal
            total_weight = sum(c.weight for c in available)

        score = 0.0
        for component in available:
            score += (component.weight / total_weight) * component.normalized_score

        history_days = float(analytics.get("history_days", 0.0))
        reason_codes = [c.reason_code for c in components if c.reason_code]
        if history_days >= self.policy.minimum_history_days:
            reason_codes.append("SUFFICIENT_TRANSACTION_HISTORY")

        unavailable = [c.name for c in components if not c.available]
        band = self.policy.score_band_for_score(score)
        return ReliabilityResult(
            score=score,
            band=band,
            components=components,
            reason_codes=reason_codes,
            unavailable_components=unavailable,
        )
