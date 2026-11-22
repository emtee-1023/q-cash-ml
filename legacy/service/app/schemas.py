"""Request and response bodies for the Q-Cash assessment API."""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Legacy forecast API models (kept for backwards compatibility).
# ---------------------------------------------------------------------------


class DailyRecord(BaseModel):
    transaction_date: date = Field(..., description="Calendar date of the daily record.")
    revenue_kes: float = Field(..., ge=0, description="M-Pesa Till revenue (KSh) for that day.")
    transaction_count: int = Field(..., ge=0, description="Number of Till transactions that day.")


class PredictRequest(BaseModel):
    business_id: str | None = Field(default=None, description="Optional caller-supplied id.")
    daily: list[DailyRecord] = Field(
        ..., min_length=90, description="Trailing daily revenue + transaction counts."
    )


class LimitBreakdown(BaseModel):
    simple_revenue_kes: float
    conservative_kes: float
    revenue_stability_kes: float
    cashflow_coverage_kes: float


class PredictResponse(BaseModel):
    predicted_next_30d_revenue_kes: float
    range_low_kes: float
    range_high_kes: float
    range_basis: str
    limits: LimitBreakdown
    model: str
    target: str


class BatchPredictRequest(BaseModel):
    items: list[PredictRequest] = Field(..., min_length=1)


class BatchPredictResponse(BaseModel):
    predictions: list[PredictResponse]


class DecisionResponse(BaseModel):
    recommended_available_kes: float
    binding_strategy: str
    risk_tier: str
    predicted_next_30d_revenue_kes: float
    range_low_kes: float
    range_high_kes: float
    limits: LimitBreakdown
    reasoning: str
    model: str
    target: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model: str


# ---------------------------------------------------------------------------
# Assessment API models.
# ---------------------------------------------------------------------------


class TransactionRecord(BaseModel):
    """A single raw transaction. All fields are preserved; none are dropped."""

    transaction_id: str | None = Field(default=None, description="Unique transaction identifier.")
    transaction_date: datetime = Field(..., description="Date/time of the transaction.")
    amount: float = Field(..., description="Amount in KES (positive credit; negative = debit).")
    direction: Literal["credit", "debit"] | None = Field(
        default=None, description="Flow direction; defaults from the sign of amount."
    )
    status: str | None = Field(default=None, description="Settlement status, e.g. completed.")
    transaction_type: str | None = Field(
        default=None, description="Channel/type, e.g. till_payment_received."
    )
    category: str | None = Field(default=None, description="Optional caller-supplied category.")
    reference: str | None = Field(default=None, description="Payment reference text.")
    counterparty: str | None = Field(default=None, description="Counterparty identifier or name.")
    customer_id_hash: str | None = Field(default=None, description="Hashed customer identifier.")


class BusinessProfile(BaseModel):
    business_type: str | None = None
    sector: str | None = None
    cohort: str | None = None
    registration_date: date | None = None
    first_transaction_date: date | None = None


class RepaymentHistory(BaseModel):
    advances_count: int = Field(default=0, ge=0)
    on_time_count: int = Field(default=0, ge=0)
    late_count: int = Field(default=0, ge=0)
    default_count: int = Field(default=0, ge=0)


class Exposure(BaseModel):
    outstanding_principal: float = Field(default=0.0, ge=0)
    active_advances: int = Field(default=0, ge=0)
    max_exposure_kes: float | None = Field(default=None, ge=0)


class AssessmentRequest(BaseModel):
    """Input for a full ERA assessment, supplied by Laravel."""

    business_id: str = Field(..., description="Business identifier.")
    assessment_reference: str | None = Field(default=None, description="Caller reference.")
    assessment_date: datetime | None = Field(default=None, description="Assessment cutoff date.")
    transactions: list[TransactionRecord] = Field(
        default_factory=list, description="Historical transaction records."
    )
    business_profile: BusinessProfile | None = Field(default=None)
    repayment_history: RepaymentHistory | None = Field(default=None)
    exposure: Exposure | None = Field(default=None)


class ScoreComponentModel(BaseModel):
    name: str
    raw_value: float
    normalized_score: float
    weight: float
    available: bool
    flag: str
    reason_code: str
    description: str


class TransactionSummaryModel(BaseModel):
    total_transactions: int
    category_counts: dict[str, int]
    anomaly_count: int
    unknown_count: int
    eligible_revenue_kes: float


class AssessmentResponse(BaseModel):
    assessment_reference: str
    business_id: str
    assessment_status: str
    predicted_revenue: float | None
    forecast_period_days: int
    reliability_score: float
    score_breakdown: list[ScoreComponentModel]
    recommended_advance_rate: float
    recommended_advance_amount: float
    recommended_recovery_rate: float
    maximum_repayment_days: int
    repayment_fallback: str
    repayment_risk_flags: list[str]
    reason_codes: list[str]
    warnings: list[str]
    model_version: str | None
    decision_policy_version: str
    assessment_timestamp: datetime
    currency: str
    transaction_summary: TransactionSummaryModel
