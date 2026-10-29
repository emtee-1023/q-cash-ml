"""FastAPI service that serves the Q-Cash synthetic revenue forecast.

Run locally:
    uvicorn app.main:app --host 0.0.0.0 --port 8000

The application sends a business's trailing 90+ days of daily revenue and
transaction counts to ``POST /predict`` and receives a 30-day revenue forecast
plus four proposed early-access limits. See the service README for details.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from . import features as feature_engine
from . import model
from .config import API_KEY, CORS_ORIGINS
from .schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    DailyRecord,
    DecisionResponse,
    HealthResponse,
    LimitBreakdown,
    PredictRequest,
    PredictResponse,
)


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
    yield


app = FastAPI(
    title="Q-Cash Revenue Forecast API",
    version="2.0.0",
    description=(
        "Estimates the next 30 days of M-Pesa Till revenue and proposes early "
        "access limits from a business's trailing daily revenue and transaction "
        "history. Synthetic proof-of-concept model."
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
    """Liveness probe used by Docker and load balancers."""
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
    """Training provenance, metrics and caveats for the served model."""
    return model.model_info()


@app.post(
    "/predict",
    response_model=PredictResponse,
    tags=["forecast"],
    dependencies=[Depends(require_api_key)],
)
def predict(request: PredictRequest) -> PredictResponse:
    """Forecast next-30-day Till revenue and access limits for one business."""
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
    """Forecast for several businesses in one call."""
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
    """Return a single recommended early-access amount with its reasoning."""
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
