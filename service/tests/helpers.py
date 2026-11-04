"""Shared helpers for the Q-Cash service test suite."""

from __future__ import annotations

from typing import Any

from app.config import POLICY_PATH
from app.policy import Policy, load_policy, policy_from_dict


def make_tx(
    amount: float,
    date: str = "2026-09-30T10:00:00",
    *,
    transaction_id: str | None = "T001",
    direction: str | None = "credit",
    status: str | None = "completed",
    transaction_type: str | None = "till_payment_received",
    category: str | None = None,
    reference: str | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "transaction_date": date,
        "amount": amount,
    }
    if transaction_id is not None:
        record["transaction_id"] = transaction_id
    if direction is not None:
        record["direction"] = direction
    if status is not None:
        record["status"] = status
    if transaction_type is not None:
        record["transaction_type"] = transaction_type
    if category is not None:
        record["category"] = category
    if reference is not None:
        record["reference"] = reference
    return record


def default_policy() -> Policy:
    return load_policy(POLICY_PATH)


def policy(**overrides: Any) -> Policy:
    data: dict[str, Any] = {
        "policy_version": "test-v1",
        "provisional": True,
        "currency": "KES",
        "forecast_period_days": 30,
        "minimum_history_days": 90,
        "minimum_transactions": 30,
        "max_unknown_ratio": 0.5,
        "max_anomaly_ratio": 0.1,
    }
    data.update(overrides)
    return policy_from_dict(data)
