"""Hybrid repayment recommendation for Q-Cash ERA.

The recommendation balances MSME working-capital protection against
financing-partner recovery predictability using a fixed agreed recovery
percentage with limited, rule-governed adjustments:

1. a percentage-based recovery from eligible incoming business payments;
2. a maximum repayment period;
3. a pre-agreed fallback arrangement if a balance remains outstanding at the
   end of the window.

The recovery percentage is business-specific and configurable. It starts from a
fixed agreed base rate and is adjusted only by narrow, documented rules when a
change in business performance justifies it. Python recommends the terms; it
does **not** collect or execute repayments — Laravel determines which payments
are eligible for recovery and performs the deductions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .policy import Policy


@dataclass
class RepaymentRecommendation:
    recovery_percentage: float
    maximum_repayment_days: int
    fallback_description: str
    reason_codes: list[str]
    risk_flags: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "recovery_percentage": round(self.recovery_percentage, 2),
            "maximum_repayment_days": self.maximum_repayment_days,
            "fallback_description": self.fallback_description,
            "reason_codes": list(self.reason_codes),
            "risk_flags": list(self.risk_flags),
        }


class RepaymentRecommender:
    """Recommend recovery percentage and repayment period."""

    def __init__(self, policy: Policy) -> None:
        self.policy = policy

    def recommend(self, analytics: dict[str, float]) -> RepaymentRecommendation:
        cfg = self.policy.repayment
        cv = float(analytics.get("cv_daily_revenue_30d", 0.0))
        trend = float(analytics.get("revenue_trend_30_vs_90", 0.0))

        percentage = cfg.base_recovery_percentage
        reason_codes = ["FIXED_AGREED_RATE"]
        risk_flags: list[str] = []

        if cv > cfg.volatility_adjustment_threshold:
            percentage += cfg.volatility_adjustment_pp
            reason_codes.append("VOLATILITY_ADJUSTMENT")
            risk_flags.append("HIGH_REVENUE_VOLATILITY")

        if trend < cfg.decline_adjustment_threshold:
            percentage += cfg.decline_adjustment_pp
            reason_codes.append("DECLINE_ADJUSTMENT")
            risk_flags.append("DECLINING_REVENUE")

        percentage = max(cfg.min_recovery_percentage, min(cfg.max_recovery_percentage, percentage))

        return RepaymentRecommendation(
            recovery_percentage=percentage,
            maximum_repayment_days=cfg.max_repayment_days,
            fallback_description=cfg.fallback_description,
            reason_codes=reason_codes,
            risk_flags=risk_flags,
        )
