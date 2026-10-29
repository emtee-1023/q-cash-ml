"""Request and response bodies for the Q-Cash synthetic forecast API."""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field


class DailyRecord(BaseModel):
    """One day of a business's trailing revenue and transaction activity."""

    transaction_date: date = Field(..., description="Calendar date of the daily record.")
    revenue_kes: float = Field(..., ge=0, description="M-Pesa Till revenue (KSh) for that day.")
    transaction_count: int = Field(..., ge=0, description="Number of Till transactions that day.")


class PredictRequest(BaseModel):
    """The trailing history the served model needs.

    Send at least 90 days of daily revenue and transaction counts. The latest
    date is treated as "today"; the model forecasts the following 30 days.
    """

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
    predicted_next_30d_revenue_kes: float = Field(
        ..., description="Forecast M-Pesa Till revenue (KSh) for the next 30 days."
    )
    range_low_kes: float = Field(..., description="Lower end of the typical error band.")
    range_high_kes: float = Field(..., description="Upper end of the typical error band.")
    range_basis: str = Field(..., description="Plain-language explanation of the band.")
    limits: LimitBreakdown = Field(..., description="Proposed early-access limits by strategy.")
    model: str
    target: str


class BatchPredictRequest(BaseModel):
    items: list[PredictRequest] = Field(..., min_length=1)


class BatchPredictResponse(BaseModel):
    predictions: list[PredictResponse]


class DecisionResponse(BaseModel):
    recommended_available_kes: float = Field(
        ..., description="Single recommended early-access amount (KSh)."
    )
    binding_strategy: str = Field(
        ..., description="Which strategy produced the recommendation."
    )
    risk_tier: str = Field(..., description="low | medium | high, from daily revenue CV.")
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
