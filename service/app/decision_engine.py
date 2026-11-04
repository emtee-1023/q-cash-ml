"""ERA decision engine for Q-Cash.

The decision engine combines the predicted future revenue, the reliability
score, available transaction history, revenue quality/volatility, outstanding
exposure supplied by Laravel, and the configured policy to produce an
**advance recommendation**. It recommends; it does not execute a lending
decision or disburse funds.

    Recommended Advance = Eligible Predicted Revenue x Recommended Advance Rate

The advance rate is chosen from configurable score bands. The recommendation is
then bounded by minimum/maximum advance limits and the exposure headroom Laravel
reports. A recommendation is never forced when the data is insufficient: the
engine returns one of ``eligible``, ``ineligible``, ``insufficient_data`` or
``manual_review``.

For businesses with insufficient history the engine supports a configurable,
clearly-labelled conservative cold-start assessment only when Laravel supplies
relevant business profile / cohort information; it never fabricates history or
forces a model prediction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .policy import Policy
from .reliability import ReliabilityResult
from .transactions import TransactionAnalysis


@dataclass
class DecisionResult:
    status: str
    recommended_advance_amount: float
    recommended_advance_rate: float
    reason_codes: list[str]
    warnings: list[str]
    applied_limits: dict[str, float]
    cold_start: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "recommended_advance_amount": round(self.recommended_advance_amount, 2),
            "recommended_advance_rate": self.recommended_advance_rate,
            "reason_codes": list(self.reason_codes),
            "warnings": list(self.warnings),
            "applied_limits": {k: round(v, 2) for k, v in self.applied_limits.items()},
            "cold_start": self.cold_start,
        }


class DecisionEngine:
    """Produce an advance recommendation from assessment inputs."""

    def __init__(self, policy: Policy) -> None:
        self.policy = policy

    def decide(
        self,
        predicted_revenue: float | None,
        reliability: ReliabilityResult,
        analytics: dict[str, float],
        analysis: TransactionAnalysis,
        exposure: dict[str, Any] | None,
        business_profile: dict[str, Any] | None,
    ) -> DecisionResult:
        policy = self.policy
        advance_cfg = policy.advance
        history_days = float(analytics.get("history_days", 0.0))
        n_transactions = float(analytics.get("n_transactions", 0.0))

        reasons: list[str] = []
        warnings: list[str] = list(analysis.warnings)

        exposure = exposure or {}
        outstanding = float(exposure.get("outstanding_principal", 0.0) or 0.0)
        max_exposure = float(exposure.get("max_exposure_kes") or advance_cfg.max_exposure_kes)

        if history_days < policy.minimum_history_days:
            return self._cold_start(reasons, warnings, business_profile, outstanding, max_exposure)

        if n_transactions < policy.minimum_transactions:
            reasons.append("INSUFFICIENT_TRANSACTIONS")
            return DecisionResult(
                status="insufficient_data",
                recommended_advance_amount=0.0,
                recommended_advance_rate=0.0,
                reason_codes=reasons,
                warnings=warnings,
                applied_limits=self._limits(outstanding, max_exposure),
                cold_start=False,
            )

        reasons.append("SUFFICIENT_TRANSACTION_HISTORY")
        for code in reliability.reason_codes:
            if code not in reasons:
                reasons.append(code)

        if analysis.unknown_ratio > policy.max_unknown_ratio:
            reasons.append("HIGH_UNCLASSIFIED_RATIO")
            warnings.append(
                f"{analysis.unknown_ratio:.0%} of transactions could not be classified."
            )

        if analysis.anomaly_ratio > policy.max_anomaly_ratio:
            reasons.append("UNUSUAL_ACTIVITY")
            warnings.append(
                f"{analysis.anomaly_ratio:.0%} of transactions were flagged as unusual."
            )

        if predicted_revenue is None or predicted_revenue <= 0:
            reasons.append("NO_REVENUE_SIGNAL")
            return DecisionResult(
                status="ineligible",
                recommended_advance_amount=0.0,
                recommended_advance_rate=0.0,
                reason_codes=reasons,
                warnings=warnings,
                applied_limits=self._limits(outstanding, max_exposure),
                cold_start=False,
            )

        if reliability.score < policy.scoring.min_eligible_score:
            reasons.append("LOW_RELIABILITY_SCORE")
            return DecisionResult(
                status="ineligible",
                recommended_advance_amount=0.0,
                recommended_advance_rate=0.0,
                reason_codes=reasons,
                warnings=warnings,
                applied_limits=self._limits(outstanding, max_exposure),
                cold_start=False,
            )

        rate = policy.advance_rate_for_score(reliability.score)
        amount = float(predicted_revenue) * rate
        if amount > 0 and amount < advance_cfg.min_advance_kes:
            amount = advance_cfg.min_advance_kes
        amount = min(amount, advance_cfg.max_advance_kes)

        headroom = max(0.0, max_exposure - outstanding)
        if headroom <= 0:
            reasons.append("EXPOSURE_LIMIT_REACHED")
            return DecisionResult(
                status="ineligible",
                recommended_advance_amount=0.0,
                recommended_advance_rate=rate,
                reason_codes=reasons,
                warnings=warnings,
                applied_limits=self._limits(outstanding, max_exposure),
                cold_start=False,
            )
        amount = min(amount, headroom)

        status = "manual_review" if ("HIGH_UNCLASSIFIED_RATIO" in reasons or "UNUSUAL_ACTIVITY" in reasons) else "eligible"
        if status == "manual_review":
            warnings.append("Advance recommendation held for manual review.")

        return DecisionResult(
            status=status,
            recommended_advance_amount=amount,
            recommended_advance_rate=rate,
            reason_codes=reasons,
            warnings=warnings,
            applied_limits=self._limits(outstanding, max_exposure),
            cold_start=False,
        )

    def _cold_start(
        self,
        reasons: list[str],
        warnings: list[str],
        business_profile: dict[str, Any] | None,
        outstanding: float,
        max_exposure: float,
    ) -> DecisionResult:
        cold = self.policy.advance.cold_start
        reasons.append("LIMITED_HISTORY")
        warnings.append(
            "Insufficient transaction history for a personalised revenue prediction."
        )
        if not cold.enabled or not business_profile:
            reasons.append("INSUFFICIENT_HISTORY")
            return DecisionResult(
                status="insufficient_data",
                recommended_advance_amount=0.0,
                recommended_advance_rate=0.0,
                reason_codes=reasons,
                warnings=warnings,
                applied_limits=self._limits(outstanding, max_exposure),
                cold_start=False,
            )
        reasons.append("COLD_START_CONSERVATIVE_ASSESSMENT")
        amount = cold.conservative_advance_kes
        headroom = max(0.0, max_exposure - outstanding)
        amount = min(amount, headroom) if headroom > 0 else 0.0
        if amount <= 0:
            reasons.append("EXPOSURE_LIMIT_REACHED")
            return DecisionResult(
                status="ineligible",
                recommended_advance_amount=0.0,
                recommended_advance_rate=cold.conservative_advance_rate,
                reason_codes=reasons,
                warnings=warnings,
                applied_limits=self._limits(outstanding, max_exposure),
                cold_start=True,
            )
        return DecisionResult(
            status="eligible",
            recommended_advance_amount=amount,
            recommended_advance_rate=cold.conservative_advance_rate,
            reason_codes=reasons,
            warnings=warnings,
            applied_limits=self._limits(outstanding, max_exposure),
            cold_start=True,
        )

    def _limits(self, outstanding: float, max_exposure: float) -> dict[str, float]:
        return {
            "min_advance_kes": self.policy.advance.min_advance_kes,
            "max_advance_kes": self.policy.advance.max_advance_kes,
            "max_exposure_kes": max_exposure,
            "outstanding_principal_kes": outstanding,
            "exposure_headroom_kes": max(0.0, max_exposure - outstanding),
        }
