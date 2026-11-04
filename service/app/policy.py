"""Q-Cash assessment and decision policy.

The decision engine, reliability scorer and repayment recommender are all
driven by a versioned JSON policy document so that scoring weights, advance
rates, limits and thresholds are configurable rather than hardcoded. Every
assessment records which policy version it used.

All values in the shipped policy are clearly labelled as *provisional* defaults:
they make the system runnable but have not been calibrated against real
repayment outcomes and must not be treated as a production lending policy.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class AnomalyConfig:
    mad_multiplier: float = 5.0
    min_tx_for_anomaly: int = 10


@dataclass
class ClassificationConfig:
    sale_keywords: list[str] = field(default_factory=list)
    refund_keywords: list[str] = field(default_factory=list)
    reversal_keywords: list[str] = field(default_factory=list)
    capital_keywords: list[str] = field(default_factory=list)
    transfer_keywords: list[str] = field(default_factory=list)
    failed_statuses: list[str] = field(default_factory=list)
    pending_statuses: list[str] = field(default_factory=list)
    assume_unknown_as_sale: bool = False
    anomaly: AnomalyConfig = field(default_factory=AnomalyConfig)


@dataclass
class ScoreBand:
    min: float
    label: str


@dataclass
class ScoringConfig:
    weights: dict[str, float] = field(default_factory=dict)
    volatility_cv_ceiling: float = 1.0
    frequency_ref_per_active_day: float = 20.0
    trend_ref: float = 0.5
    min_eligible_score: float = 40.0
    score_bands: list[ScoreBand] = field(default_factory=list)


@dataclass
class RateBand:
    min_score: float
    advance_rate: float


@dataclass
class ColdStartConfig:
    enabled: bool = True
    conservative_advance_kes: float = 5000.0
    conservative_advance_rate: float = 0.10


@dataclass
class AdvanceConfig:
    rate_bands: list[RateBand] = field(default_factory=list)
    min_advance_kes: float = 1000.0
    max_advance_kes: float = 100000.0
    max_exposure_kes: float = 100000.0
    cold_start: ColdStartConfig = field(default_factory=ColdStartConfig)


@dataclass
class RepaymentConfig:
    base_recovery_percentage: float = 10.0
    min_recovery_percentage: float = 8.0
    max_recovery_percentage: float = 20.0
    volatility_adjustment_threshold: float = 0.6
    volatility_adjustment_pp: float = 2.0
    decline_adjustment_threshold: float = -0.2
    decline_adjustment_pp: float = 2.0
    max_repayment_days: int = 30
    fallback_description: str = (
        "Outstanding balance is due under a pre-agreed installment arrangement "
        "at the end of the repayment window."
    )


@dataclass
class Policy:
    policy_version: str
    provisional: bool
    description: str
    currency: str
    forecast_period_days: int
    minimum_history_days: int
    minimum_transactions: int
    max_unknown_ratio: float
    max_anomaly_ratio: float
    classification: ClassificationConfig
    scoring: ScoringConfig
    advance: AdvanceConfig
    repayment: RepaymentConfig

    def advance_rate_for_score(self, score: float) -> float:
        """Return the recommended advance rate for a reliability score.

        The highest band whose ``min_score`` is <= ``score`` wins. A score below
        every band yields 0.0.
        """
        ordered = sorted(self.advance.rate_bands, key=lambda b: b.min_score, reverse=True)
        for band in ordered:
            if score >= band.min_score:
                return band.advance_rate
        return 0.0

    def score_band_for_score(self, score: float) -> str:
        ordered = sorted(self.scoring.score_bands, key=lambda b: b.min, reverse=True)
        for band in ordered:
            if score >= band.min:
                return band.label
        return "very_low"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Policy file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def policy_from_dict(data: dict[str, Any]) -> Policy:
    cls = data.get("classification", {})
    sc = data.get("scoring", {})
    adv = data.get("advance", {})
    rep = data.get("repayment", {})

    classification = ClassificationConfig(
        sale_keywords=cls.get("sale_keywords", []),
        refund_keywords=cls.get("refund_keywords", []),
        reversal_keywords=cls.get("reversal_keywords", []),
        capital_keywords=cls.get("capital_keywords", []),
        transfer_keywords=cls.get("transfer_keywords", []),
        failed_statuses=cls.get("failed_statuses", []),
        pending_statuses=cls.get("pending_statuses", []),
        assume_unknown_as_sale=bool(cls.get("assume_unknown_as_sale", False)),
        anomaly=AnomalyConfig(**cls.get("anomaly", {})),
    )
    scoring = ScoringConfig(
        weights=dict(sc.get("weights", {})),
        volatility_cv_ceiling=float(sc.get("volatility_cv_ceiling", 1.0)),
        frequency_ref_per_active_day=float(sc.get("frequency_ref_per_active_day", 20.0)),
        trend_ref=float(sc.get("trend_ref", 0.5)),
        min_eligible_score=float(sc.get("min_eligible_score", 40.0)),
        score_bands=[ScoreBand(**b) for b in sc.get("score_bands", [])],
    )
    cold_start = ColdStartConfig(**adv.get("cold_start", {}))
    advance = AdvanceConfig(
        rate_bands=[RateBand(**b) for b in adv.get("rate_bands", [])],
        min_advance_kes=float(adv.get("min_advance_kes", 1000.0)),
        max_advance_kes=float(adv.get("max_advance_kes", 100000.0)),
        max_exposure_kes=float(adv.get("max_exposure_kes", 100000.0)),
        cold_start=cold_start,
    )
    repayment = RepaymentConfig(**rep)
    return Policy(
        policy_version=str(data.get("policy_version", "unknown")),
        provisional=bool(data.get("provisional", True)),
        description=str(data.get("description", "")),
        currency=str(data.get("currency", "KES")),
        forecast_period_days=int(data.get("forecast_period_days", 30)),
        minimum_history_days=int(data.get("minimum_history_days", 90)),
        minimum_transactions=int(data.get("minimum_transactions", 30)),
        max_unknown_ratio=float(data.get("max_unknown_ratio", 0.5)),
        max_anomaly_ratio=float(data.get("max_anomaly_ratio", 0.1)),
        classification=classification,
        scoring=scoring,
        advance=advance,
        repayment=repayment,
    )


def load_policy(path: Path) -> Policy:
    """Load and validate a policy document from disk."""
    return policy_from_dict(_load_json(path))
