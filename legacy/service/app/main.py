"""FastAPI service for Q-Cash revenue forecasting and ERA assessment.

Run locally:
    uvicorn app.main:app --host 0.0.0.0 --port 8000

Endpoints
---------
* ``POST /api/v1/assessments`` — full ERA assessment: transaction intelligence,
  revenue prediction, reliability scoring, advance + repayment recommendation.
* ``POST /predict``, ``POST /predict/batch``, ``POST /decision`` — legacy
  synthetic revenue-forecast endpoints, kept for backwards compatibility.
* ``GET /health``, ``GET /model``, ``GET /policy`` — operational metadata.

The assessment service never queries Laravel's database, disburses funds,
deducts repayments, or calls payment providers. Laravel supplies all inputs and
executes all financial operations.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from . import features as feature_engine
from . import model
from .assessment import AssessmentService
from .config import API_KEY, CORS_ORIGINS, POLICY_PATH
from .logging_conf import setup_logging
from .policy import load_policy
from .schemas import (
    AssessmentRequest,
    AssessmentResponse,
    BatchPredictRequest,
    BatchPredictResponse,
    DailyRecord,
    DecisionResponse,
    HealthResponse,
    LimitBreakdown,
    PredictRequest,
    PredictResponse,
)

setup_logging()

_service: AssessmentService | None = None


def get_service() -> AssessmentService:
    global _service
    if _service is None:
        _service = AssessmentService()
    return _service


def require_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
    """Reject the request when an API key is configured and does not match."""
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-API-Key header.",
        )


def _daily_frame(records: list[DailyRecord]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "transaction_date": [r.transaction_date for r in records],
            "revenue_kes": [r.revenue_kes for r in records],
            "transaction_count": [r.transaction_count for r in records],
        }
    )


def _build_response(features: pd.DataFrame) -> PredictResponse:
    bundle = model.get_bundle()
    prediction = float(model.predict_revenue(features)[0])
    cv = float(features["cv_daily_revenue_30d"].iloc[0])
    median_daily = float(features["median_daily_revenue_30d"].iloc[0])
    error = model._error_band_kes()
    limits = model.limits_for(prediction, cv, median_daily)
    return PredictResponse(
        predicted_next_30d_revenue_kes=round(prediction, 2),
        range_low_kes=round(max(0.0, prediction - error), 2),
        range_high_kes=round(prediction + error, 2),
        range_basis=(
            "Prediction plus or minus the model's average error on held-out "
            f"weeks (KSh {error:,.0f})."
        ),
        limits=LimitBreakdown(**limits),
        model=model._model_label(),
        target=bundle.get("target", "unknown"),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    model.get_bundle()
    get_service()
    yield


app = FastAPI(
    title="Q-Cash ML and Decision API",
    version="3.0.0",
    description=(
        "Transaction intelligence, revenue forecasting, reliability scoring and "
        "ERA advance / repayment recommendations for Kenyan MSMEs. Synthetic "
        "proof-of-concept model; recommendations are not credit decisions."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    try:
        loaded = model.get_bundle() is not None
    except Exception:
        loaded = False
    return HealthResponse(
        status="ok" if loaded else "degraded",
        model_loaded=loaded,
        model=model._model_label() if loaded else "unloaded",
    )


@app.get("/model", tags=["model"], dependencies=[Depends(require_api_key)])
def model_metadata() -> dict[str, Any]:
    return model.model_info()


@app.get("/policy", tags=["model"], dependencies=[Depends(require_api_key)])
def policy_metadata() -> dict[str, Any]:
    policy = load_policy(POLICY_PATH)
    return {
        "policy_version": policy.policy_version,
        "provisional": policy.provisional,
        "currency": policy.currency,
        "forecast_period_days": policy.forecast_period_days,
        "minimum_history_days": policy.minimum_history_days,
    }


@app.post(
    "/api/v1/assessments",
    response_model=AssessmentResponse,
    tags=["assessments"],
    dependencies=[Depends(require_api_key)],
)
def assess(request: AssessmentRequest) -> AssessmentResponse:
    """Run a full ERA assessment for one business."""
    try:
        result = get_service().assess(
            business_id=request.business_id,
            transactions=[t.model_dump() for t in request.transactions],
            assessment_reference=request.assessment_reference,
            assessment_date=request.assessment_date,
            business_profile=request.business_profile.model_dump()
            if request.business_profile
            else None,
            repayment_history=request.repayment_history.model_dump()
            if request.repayment_history
            else None,
            exposure=request.exposure.model_dump() if request.exposure else None,
        )
        return AssessmentResponse(**result)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@app.post(
    "/predict",
    response_model=PredictResponse,
    tags=["forecast"],
    dependencies=[Depends(require_api_key)],
)
def predict(request: PredictRequest) -> PredictResponse:
    try:
        features = feature_engine.compute_features(_daily_frame(request.daily))
        return _build_response(pd.DataFrame([features]))
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@app.post(
    "/predict/batch",
    response_model=BatchPredictResponse,
    tags=["forecast"],
    dependencies=[Depends(require_api_key)],
)
def predict_batch(request: BatchPredictRequest) -> BatchPredictResponse:
    rows = []
    for item in request.items:
        try:
            rows.append(feature_engine.compute_features(_daily_frame(item.daily)))
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
            ) from exc
    features = pd.DataFrame(rows)
    predictions = model.predict_revenue(features)
    responses = []
    for i, prediction in enumerate(predictions):
        cv = float(features["cv_daily_revenue_30d"].iloc[i])
        median_daily = float(features["median_daily_revenue_30d"].iloc[i])
        error = model._error_band_kes()
        limits = model.limits_for(float(prediction), cv, median_daily)
        responses.append(
            PredictResponse(
                predicted_next_30d_revenue_kes=round(float(prediction), 2),
                range_low_kes=round(max(0.0, float(prediction) - error), 2),
                range_high_kes=round(float(prediction) + error, 2),
                range_basis=(
                    "Prediction plus or minus the model's average error on "
                    f"held-out weeks (KSh {error:,.0f})."
                ),
                limits=LimitBreakdown(**limits),
                model=model._model_label(),
                target=model.get_bundle().get("target", "unknown"),
            )
        )
    return BatchPredictResponse(predictions=responses)


@app.post(
    "/decision",
    response_model=DecisionResponse,
    tags=["forecast"],
    dependencies=[Depends(require_api_key)],
)
def decision(request: PredictRequest) -> DecisionResponse:
    try:
        features = feature_engine.compute_features(_daily_frame(request.daily))
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    prediction = float(model.predict_revenue(pd.DataFrame([features]))[0])
    cv = float(features["cv_daily_revenue_30d"])
    median_daily = float(features["median_daily_revenue_30d"])
    error = model._error_band_kes()
    decision = model.decision_for(prediction, cv, median_daily)
    return DecisionResponse(
        recommended_available_kes=decision["recommended_available_kes"],
        binding_strategy=decision["binding_strategy"],
        risk_tier=decision["risk_tier"],
        predicted_next_30d_revenue_kes=round(prediction, 2),
        range_low_kes=round(max(0.0, prediction - error), 2),
        range_high_kes=round(prediction + error, 2),
        limits=LimitBreakdown(**decision["limits"]),
        reasoning=decision["reasoning"],
        model=model._model_label(),
        target=model.get_bundle().get("target", "unknown"),
    )
