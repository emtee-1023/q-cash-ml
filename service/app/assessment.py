"""Assessment orchestration for Q-Cash.

This module wires the four stages together into a single assessment:

    transaction intelligence -> revenue prediction -> reliability scoring
                              -> ERA decision + repayment recommendation

It keeps business rules out of the HTTP layer: route handlers only parse the
request and call :meth:`AssessmentService.assess`.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from . import features as feature_engine
from . import model as model_module
from . import transactions as txn
from .config import POLICY_PATH
from .decision_engine import DecisionEngine, DecisionResult
from .policy import Policy, load_policy
from .reliability import ReliabilityResult, ReliabilityScorer
from .repayment import RepaymentRecommender, RepaymentRecommendation
from .transactions import TransactionAnalysis

logger = logging.getLogger("qcash.assessment")


def _naive(ts: pd.Timestamp) -> pd.Timestamp:
    if ts.tzinfo is not None:
        return ts.tz_localize(None)
    return ts


class AssessmentService:
    """Run the full assessment pipeline for a single business."""

    def __init__(self, policy: Policy | None = None) -> None:
        self.policy = policy or load_policy(POLICY_PATH)
        self.scorer = ReliabilityScorer(self.policy)
        self.engine = DecisionEngine(self.policy)
        self.repayment = RepaymentRecommender(self.policy)

    def assess(
        self,
        business_id: str,
        transactions: list[dict[str, Any]],
        assessment_reference: str | None = None,
        assessment_date: datetime | None = None,
        business_profile: dict[str, Any] | None = None,
        repayment_history: dict[str, Any] | None = None,
        exposure: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        now = datetime.now(timezone.utc)
        as_of = self._as_of(transactions, assessment_date, now)

        analysis = txn.analyze_transactions(transactions, self.policy.classification, as_of)
        analytics = analysis.analytics

        reliability = self.scorer.score(analytics, repayment_history)

        predicted_revenue, model_version = self._predict(analysis, analytics, as_of)

        decision = self.engine.decide(
            predicted_revenue=predicted_revenue,
            reliability=reliability,
            analytics=analytics,
            analysis=analysis,
            exposure=exposure,
            business_profile=business_profile,
        )
        repayment = self.repayment.recommend(analytics)

        reference = assessment_reference or self._default_reference(business_id, now)

        self._log(reference, business_id, decision, analysis, reliability)

        return self._response(
            reference=reference,
            business_id=business_id,
            status=decision.status,
            predicted_revenue=predicted_revenue,
            model_version=model_version,
            reliability=reliability,
            decision=decision,
            repayment=repayment,
            analysis=analysis,
            timestamp=now,
            policy_version=self.policy.policy_version,
            forecast_period_days=self.policy.forecast_period_days,
        )

    def _as_of(
        self,
        transactions: list[dict[str, Any]],
        assessment_date: datetime | None,
        now: datetime,
    ) -> pd.Timestamp:
        if assessment_date is not None:
            return _naive(pd.Timestamp(assessment_date))
        dates = [txn._parse_date(t.get("transaction_date")) for t in transactions]
        dates = [d for d in dates if d is not None]
        if dates:
            return max(dates) + pd.Timedelta(days=1)
        return _naive(pd.Timestamp(now))

    def _predict(
        self,
        analysis: TransactionAnalysis,
        analytics: dict[str, float],
        as_of: pd.Timestamp,
    ) -> tuple[float | None, str | None]:
        if analytics.get("history_days", 0.0) < self.policy.minimum_history_days:
            return None, None
        daily = txn.reindex_daily(analysis.eligible_daily, as_of)
        if len(daily) < self.policy.minimum_history_days:
            return None, None
        try:
            feature_row = feature_engine.compute_features(daily)
            prediction = float(model_module.predict_revenue(pd.DataFrame([feature_row]))[0])
            bundle = model_module.get_bundle()
            version = str(bundle.get("model_label", bundle.get("artifact_version", "unknown")))
            return prediction, version
        except Exception as exc:
            logger.warning("Revenue prediction failed: %s", exc)
            return None, None

    @staticmethod
    def _default_reference(business_id: str, now: datetime) -> str:
        return f"{business_id}-{now.strftime('%Y%m%d%H%M%S')}"

    def _log(
        self,
        reference: str,
        business_id: str,
        decision: DecisionResult,
        analysis: TransactionAnalysis,
        reliability: ReliabilityResult,
    ) -> None:
        logger.info(
            "assessment",
            extra={
                "assessment_reference": reference,
                "business_id": business_id,
                "status": decision.status,
                "reliability_score": round(reliability.score, 2),
                "category_counts": analysis.category_counts,
                "anomaly_count": analysis.anomaly_count,
                "unknown_count": analysis.unknown_count,
            },
        )

    @staticmethod
    def _response(
        reference: str,
        business_id: str,
        status: str,
        predicted_revenue: float | None,
        model_version: str | None,
        reliability: ReliabilityResult,
        decision: DecisionResult,
        repayment: RepaymentRecommendation,
        analysis: TransactionAnalysis,
        timestamp: datetime,
        policy_version: str,
        forecast_period_days: int,
    ) -> dict[str, Any]:
        return {
            "assessment_reference": reference,
            "business_id": business_id,
            "assessment_status": status,
            "predicted_revenue": round(predicted_revenue, 2) if predicted_revenue is not None else None,
            "forecast_period_days": forecast_period_days,
            "reliability_score": round(reliability.score, 2),
            "score_breakdown": [c.to_dict() for c in reliability.components],
            "recommended_advance_rate": decision.recommended_advance_rate,
            "recommended_advance_amount": round(decision.recommended_advance_amount, 2),
            "recommended_recovery_rate": round(repayment.recovery_percentage, 2),
            "maximum_repayment_days": repayment.maximum_repayment_days,
            "repayment_fallback": repayment.fallback_description,
            "repayment_risk_flags": repayment.risk_flags,
            "reason_codes": decision.reason_codes,
            "warnings": decision.warnings,
            "model_version": model_version,
            "decision_policy_version": policy_version,
            "assessment_timestamp": timestamp.isoformat(),
            "currency": "KES",
            "transaction_summary": {
                "total_transactions": len(analysis.records),
                "category_counts": analysis.category_counts,
                "anomaly_count": analysis.anomaly_count,
                "unknown_count": analysis.unknown_count,
                "eligible_revenue_kes": round(analysis.eligible_revenue_total, 2),
            },
        }
