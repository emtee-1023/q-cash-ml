"""FastAPI service for the Q-Cash decision engine (qcash_v2).

A thin HTTP wrapper around ``engine.run_assessment`` — all of the business logic
(transaction intelligence, prediction, reliability, decision, terms) lives in
``engine.py``. Money is integer minor units (1 KSh = 100) end to end.

Endpoints
---------
* ``POST /assess``  — full ERA assessment for one business.
* ``GET  /health``  — liveness + model presence.
* ``GET  /model``   — model metadata (no estimator).

Run locally:  cd qcash_v2 && uvicorn api:app --host 0.0.0.0 --port 8000
Docker:       see Dockerfile / docker-compose.yml
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Literal

import pandas as pd
from fastapi import FastAPI, Header, HTTPException, status
from pydantic import BaseModel, Field

import engine

API_KEY = os.getenv("QCASH_API_KEY", "").strip()

app = FastAPI(
    title="Q-Cash Decision Engine",
    version="2.0.0",
    description=(
        "Transaction intelligence, revenue forecasting, reliability scoring and "
        "ERA advance / repayment recommendation for Kenyan MSMEs. "
        "Synthetic proof-of-concept model; recommendations are not credit decisions."
    ),
)


class Transaction(BaseModel):
    ref: str = Field(..., description="Unique transaction reference.")
    occurred_at: datetime = Field(..., description="Transaction timestamp.")
    amount_minor: int = Field(..., description="Amount in minor units (1 KSh = 100).")
    type: Literal["credit", "debit"] = Field(default="credit")
    is_reversal_of: str | None = None
    reference: str | None = None
    payer_hash: str | None = None


class AssessRequest(BaseModel):
    business_id: str
    sector: str
    assessment_as_of: datetime | None = None
    transactions: list[Transaction] = Field(default_factory=list)
    repayment: dict[str, Any] | None = None
    exposure: dict[str, Any] | None = None
    business_profile: dict[str, Any] | None = None


def _require_key(x_api_key: str | None) -> None:
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Invalid or missing X-API-Key header.")


@app.get("/health")
def health() -> dict[str, Any]:
    path = engine.model_path()
    return {"status": "ok", "model_present": path.exists(), "model_path": str(path)}


@app.get("/model")
def model_info() -> dict[str, Any]:
    import joblib

    path = engine.model_path()
    if not path.exists():
        raise HTTPException(status_code=503, detail="Model artifact not found.")
    b = joblib.load(path)
    return {
        "model_type": b.get("model_type"),
        "model_version": b.get("model_version"),
        "cv_error": b.get("cv_error"),
        "error_margin_k": b.get("error_margin_k"),
        "n_features": len(b.get("enh_columns") or b.get("feature_columns") or []),
        "selection_rationale": b.get("selection_rationale"),
    }


@app.post("/assess")
def assess(
    req: AssessRequest,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> dict[str, Any]:
    _require_key(x_api_key)

    records: list[dict[str, Any]] = []
    for t in req.transactions:
        ts = pd.Timestamp(t.occurred_at)
        if ts.tzinfo is not None:
            ts = ts.tz_localize(None)
        records.append({
            "ref": t.ref,
            "occurred_at": ts,
            "amount_minor": int(t.amount_minor),
            "type": t.type,
            "is_reversal_of": t.is_reversal_of,
            "reference": t.reference,
            "payer_hash": t.payer_hash,
        })

    df = pd.DataFrame(records, columns=[
        "ref", "occurred_at", "amount_minor", "type",
        "is_reversal_of", "reference", "payer_hash",
    ])
    df["business_id"] = req.business_id
    df["customer_id"] = None
    df["channel"] = "paybill"

    return engine.run_assessment(
        df,
        req.sector,
        repayment=req.repayment,
        exposure=req.exposure,
        business_profile=req.business_profile,
    )
