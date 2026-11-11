"""Transaction intelligence for the Q-Cash assessment pipeline.

This module validates, classifies and analyses incoming transaction records
*before* they are used for revenue prediction or scoring. It is deliberately
read-only with respect to the raw transactions:

* raw transactions are never deleted or altered;
* unusual transactions are flagged, never silently excluded;
* a large transaction is only flagged as unusual, never automatically
  disqualified — a genuine large customer purchase remains eligible revenue;
* inflows that cannot be classified confidently are marked ``unknown`` rather
  than being assigned a made-up category.

Classification is rules-based and configurable through the policy document.
Only transactions classified as ``business_sale`` contribute to *eligible*
revenue; refunds, reversals, capital injections, internal transfers and other
non-revenue inflows are kept but excluded from the revenue signal so they cannot
artificially inflate a business's predicted revenue.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from .policy import ClassificationConfig


class TransactionCategory(str, Enum):
    BUSINESS_SALE = "business_sale"
    REFUND = "refund"
    REVERSAL = "reversal"
    CAPITAL_INJECTION = "capital_injection"
    INTERNAL_TRANSFER = "internal_transfer"
    OTHER_NON_REVENUE = "other_non_revenue"
    UNKNOWN = "unknown"


REVENUE_CATEGORIES = {TransactionCategory.BUSINESS_SALE}


@dataclass
class ClassifiedTransaction:
    """One transaction after validation and classification.

    ``raw`` preserves the original record untouched; the remaining fields are
    analytical annotations.
    """

    raw: dict[str, Any]
    transaction_id: str | None
    transaction_date: pd.Timestamp
    amount: float
    direction: str
    category: TransactionCategory
    confidence: str
    reason: str
    is_eligible_revenue: bool
    is_unusual: bool = False
    unusual_reason: str | None = None
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "transaction_id": self.transaction_id,
            "transaction_date": self.transaction_date.isoformat(),
            "amount": self.amount,
            "direction": self.direction,
            "category": self.category.value,
            "confidence": self.confidence,
            "reason": self.reason,
            "is_eligible_revenue": self.is_eligible_revenue,
            "is_unusual": self.is_unusual,
            "unusual_reason": self.unusual_reason,
            "issues": list(self.issues),
        }


def _parse_date(value: Any) -> pd.Timestamp | None:
    try:
        ts = pd.to_datetime(value, errors="coerce")
    except Exception:
        return None
    if ts is None or pd.isna(ts):
        return None
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    return ts


def validate_transaction(record: dict[str, Any]) -> list[str]:
    """Return a list of validation problems for a single record."""
    issues: list[str] = []
    amount = record.get("amount")
    if amount is None:
        issues.append("missing_amount")
    else:
        try:
            amount_f = float(amount)
            if not np.isfinite(amount_f):
                issues.append("non_finite_amount")
            elif amount_f == 0:
                issues.append("zero_amount")
            elif amount_f < 0:
                issues.append("negative_amount")
        except (TypeError, ValueError):
            issues.append("non_numeric_amount")

    date_value = record.get("transaction_date")
    if date_value is None:
        issues.append("missing_transaction_date")
    elif _parse_date(date_value) is None:
        issues.append("invalid_transaction_date")

    direction = record.get("direction")
    if direction is not None and direction not in {"credit", "debit"}:
        issues.append("invalid_direction")
    return issues


def _normalize(record: dict[str, Any]) -> dict[str, Any]:
    amount = float(record.get("amount") or 0.0)
    direction = record.get("direction")
    if amount < 0:
        direction = "debit"
    elif direction is None:
        direction = "credit"
    return {
        "transaction_id": record.get("transaction_id"),
        "transaction_date": _parse_date(record.get("transaction_date")),
        "amount": abs(amount),
        "direction": direction,
        "status": (record.get("status") or "completed").strip().lower(),
        "transaction_type": record.get("transaction_type"),
        "category": record.get("category"),
        "reference": record.get("reference"),
        "counterparty": record.get("counterparty"),
        "customer_id_hash": record.get("customer_id_hash"),
    }


def _match(needle: str | None, keywords: list[str]) -> bool:
    if not needle:
        return False
    lowered = str(needle).lower()
    return any(k.lower() in lowered for k in keywords if k)


def classify_one(record: dict[str, Any], config: ClassificationConfig) -> ClassifiedTransaction:
    """Classify a single normalized transaction."""
    tx = _normalize(record)
    issues = validate_transaction(record)
    amount = tx["amount"]
    direction = tx["direction"]
    status = tx["status"]
    transaction_type = tx["transaction_type"]
    category_hint = tx["category"]

    category: TransactionCategory
    confidence = "medium"
    reason = ""

    if direction == "debit":
        category = TransactionCategory.OTHER_NON_REVENUE
        confidence = "high"
        reason = "debit transaction is an outflow, not business revenue"
    elif status in config.failed_statuses:
        category = TransactionCategory.REVERSAL
        confidence = "high"
        reason = f"transaction status {status!r} is not settled"
    elif status in config.pending_statuses:
        category = TransactionCategory.UNKNOWN
        confidence = "low"
        reason = f"transaction status {status!r} is not yet settled"
    elif category_hint is not None and str(category_hint).strip():
        category = _category_from_label(str(category_hint))
        confidence = "high"
        reason = f"caller-supplied category {str(category_hint)!r}"
    elif _match(transaction_type, config.refund_keywords):
        category = TransactionCategory.REFUND
        reason = f"transaction type {transaction_type!r} matches refund keywords"
    elif _match(transaction_type, config.reversal_keywords):
        category = TransactionCategory.REVERSAL
        reason = f"transaction type {transaction_type!r} matches reversal keywords"
    elif _match(transaction_type, config.capital_keywords):
        category = TransactionCategory.CAPITAL_INJECTION
        reason = f"transaction type {transaction_type!r} matches capital-injection keywords"
    elif _match(transaction_type, config.transfer_keywords):
        category = TransactionCategory.INTERNAL_TRANSFER
        reason = f"transaction type {transaction_type!r} matches internal-transfer keywords"
    elif _match(transaction_type, config.sale_keywords):
        category = TransactionCategory.BUSINESS_SALE
        reason = f"transaction type {transaction_type!r} matches sale keywords"
    elif config.assume_unknown_as_sale:
        category = TransactionCategory.BUSINESS_SALE
        confidence = "low"
        reason = "unclassified inflow assumed to be a sale by policy"
    else:
        category = TransactionCategory.UNKNOWN
        confidence = "low"
        reason = "insufficient information to classify inflow confidently"

    eligible = category in REVENUE_CATEGORIES and direction == "credit"
    return ClassifiedTransaction(
        raw=record,
        transaction_id=tx["transaction_id"],
        transaction_date=tx["transaction_date"],
        amount=amount,
        direction=direction,
        category=category,
        confidence=confidence,
        reason=reason,
        is_eligible_revenue=eligible,
        issues=issues,
    )


def _category_from_label(label: str) -> TransactionCategory:
    lowered = label.strip().lower()
    mapping = {
        "sale": TransactionCategory.BUSINESS_SALE,
        "business_sale": TransactionCategory.BUSINESS_SALE,
        "payment": TransactionCategory.BUSINESS_SALE,
        "refund": TransactionCategory.REFUND,
        "reversal": TransactionCategory.REVERSAL,
        "capital": TransactionCategory.CAPITAL_INJECTION,
        "capital_injection": TransactionCategory.CAPITAL_INJECTION,
        "transfer": TransactionCategory.INTERNAL_TRANSFER,
        "internal_transfer": TransactionCategory.INTERNAL_TRANSFER,
    }
    for key, value in mapping.items():
        if key in lowered:
            return value
    return TransactionCategory.UNKNOWN


def detect_anomalies(
    records: list[ClassifiedTransaction], config: ClassificationConfig, as_of_date: pd.Timestamp
) -> None:
    """Flag unusual transactions in place (never excluding them).

    A transaction is flagged, not dropped, when it is a statistical outlier, a
    duplicate, lacks an identifier, or carries a timestamp in the future. Large
    values are only flagged relative to the business's own typical amounts.
    """
    credit_amounts = np.array(
        [r.amount for r in records if r.direction == "credit"],
        dtype=float,
    )
    seen_ids: set[str] = set()
    for record in records:
        record.is_unusual = False
        record.unusual_reason = None
        if record.transaction_id:
            if record.transaction_id in seen_ids:
                record.is_unusual = True
                record.unusual_reason = "duplicate_transaction_id"
            else:
                seen_ids.add(record.transaction_id)
        else:
            record.is_unusual = True
            record.unusual_reason = "missing_transaction_id"
        if record.transaction_date is not None and record.transaction_date > as_of_date:
            record.is_unusual = True
            record.unusual_reason = "future_timestamp"

    anomaly_cfg = config.anomaly
    if len(credit_amounts) < anomaly_cfg.min_tx_for_anomaly:
        return
    median = float(np.median(credit_amounts))
    mad = float(np.median(np.abs(credit_amounts - median)))
    if mad <= 0:
        threshold = median
    else:
        threshold = median + anomaly_cfg.mad_multiplier * mad
    for record in records:
        if record.direction == "credit" and record.amount > threshold:
            record.is_unusual = True
            record.unusual_reason = "unusually_large_amount"


def build_eligible_daily(records: list[ClassifiedTransaction]) -> pd.DataFrame:
    """Aggregate eligible business-sale credits into daily revenue."""
    eligible = [r for r in records if r.is_eligible_revenue]
    if not eligible:
        return pd.DataFrame(columns=["transaction_date", "revenue_kes", "transaction_count"])
    frame = pd.DataFrame(
        {
            "transaction_date": [r.transaction_date.normalize() for r in eligible],
            "amount": [r.amount for r in eligible],
        }
    )
    daily = frame.groupby("transaction_date", as_index=False).agg(
        revenue_kes=("amount", "sum"),
        transaction_count=("amount", "count"),
    )
    return daily.sort_values("transaction_date").reset_index(drop=True)


def _trailing(df: pd.DataFrame, as_of: pd.Timestamp, days: int) -> pd.DataFrame:
    lo = as_of - pd.Timedelta(days=days)
    return df[(df["transaction_date"] >= lo) & (df["transaction_date"] < as_of)]


def _cv(values: pd.Series) -> float:
    mean = values.mean()
    if not np.isfinite(mean) or mean <= 0:
        return 0.0
    return float(values.std() / mean)


def reindex_daily(eligible_daily: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """Fill the eligible daily aggregates over a continuous calendar range.

    Missing days (no eligible revenue) become zero rows so feature engineering
    sees the same zero-revenue days the model was trained on.
    """
    if eligible_daily is None or eligible_daily.empty:
        return pd.DataFrame(columns=["transaction_date", "revenue_kes", "transaction_count"])
    df = eligible_daily.copy()
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    start = df["transaction_date"].min()
    end = as_of - pd.Timedelta(days=1)
    if end < start:
        return df.sort_values("transaction_date").reset_index(drop=True)
    idx = pd.date_range(start, end, freq="D", name="transaction_date")
    return (
        df.set_index("transaction_date")
        .reindex(idx, fill_value=0)
        .reset_index()
    )


def transaction_concentration(records: list[ClassifiedTransaction], as_of: pd.Timestamp, days: int = 30) -> float:
    """Herfindahl-Hirschman index of eligible transaction amounts over a window.

    1.0 means a single transaction accounts for all revenue (maximum
    concentration); a low value means revenue is spread across many
    transactions. Scoring treats diversification (low HHI) as favourable.
    """
    lo = as_of - pd.Timedelta(days=days)
    amounts = [
        r.amount for r in records
        if r.is_eligible_revenue and r.transaction_date is not None and lo <= r.transaction_date < as_of
    ]
    total = float(sum(amounts))
    if total <= 0:
        return 1.0
    shares = np.array(amounts, dtype=float) / total
    return float(np.sum(shares * shares))


def compute_analytics(
    eligible_daily: pd.DataFrame, records: list[ClassifiedTransaction], as_of: pd.Timestamp
) -> dict[str, float]:
    """Compute the scoring/profile analytics from eligible daily aggregates."""
    df = eligible_daily.copy()
    if "transaction_date" not in df.columns or df.empty:
        return _empty_analytics()

    df["transaction_date"] = pd.to_datetime(df["transaction_date"])

    w7 = _trailing(df, as_of, 7)
    w30 = _trailing(df, as_of, 30)
    w90 = _trailing(df, as_of, 90)

    rev30 = w30["revenue_kes"]
    rev7 = float(w7["revenue_kes"].sum())
    rev30_total = float(rev30.sum())
    rev90_total = float(w90["revenue_kes"].sum())
    tx30 = float(w30["transaction_count"].sum())
    active30 = float((w30["transaction_count"] > 0).sum())

    avg30 = rev30_total / 30.0 if rev30_total > 0 else 0.0
    avg90 = rev90_total / 90.0 if rev90_total > 0 else 0.0
    trend = float(avg30 / avg90 - 1.0) if avg90 > 0 else 0.0

    weekend = w30["transaction_date"].dt.weekday >= 5
    weekend_share = float(w30.loc[weekend, "revenue_kes"].sum() / rev30_total) if rev30_total > 0 else 0.0

    return {
        "history_days": float((df["transaction_date"].max() - df["transaction_date"].min()).days + 1)
        if len(df) else 0.0,
        "n_days": float(len(df)),
        "n_transactions": float(len(records)),
        "eligible_revenue_7d": rev7,
        "eligible_revenue_30d": rev30_total,
        "eligible_revenue_90d": rev90_total,
        "active_days_30d": active30,
        "active_day_rate_30d": float(active30 / 30.0),
        "cv_daily_revenue_30d": _cv(rev30),
        "median_daily_revenue_30d": float(rev30.median()) if len(rev30) else 0.0,
        "avg_tx_per_active_day_30d": float(tx30 / active30) if active30 > 0 else 0.0,
        "transaction_count_30d": tx30,
        "revenue_trend_30_vs_90": trend,
        "weekend_revenue_share_30d": weekend_share,
        "transaction_concentration_30d": transaction_concentration(records, as_of, 30),
    }


def _empty_analytics() -> dict[str, float]:
    keys = [
        "history_days", "n_days", "n_transactions", "eligible_revenue_7d",
        "eligible_revenue_30d", "eligible_revenue_90d", "active_days_30d",
        "active_day_rate_30d", "cv_daily_revenue_30d", "median_daily_revenue_30d",
        "avg_tx_per_active_day_30d", "transaction_count_30d",
        "revenue_trend_30_vs_90", "weekend_revenue_share_30d",
        "transaction_concentration_30d",
    ]
    return {k: 0.0 for k in keys}


@dataclass
class TransactionAnalysis:
    """The full output of the transaction-intelligence stage."""

    records: list[ClassifiedTransaction]
    category_counts: dict[str, int]
    anomaly_count: int
    unknown_count: int
    eligible_revenue_total: float
    eligible_transaction_count: int
    eligible_daily: pd.DataFrame
    analytics: dict[str, float]
    warnings: list[str]

    @property
    def unknown_ratio(self) -> float:
        total = len(self.records)
        return self.unknown_count / total if total else 0.0

    @property
    def anomaly_ratio(self) -> float:
        total = len(self.records)
        return self.anomaly_count / total if total else 0.0


def analyze_transactions(
    raw_records: list[dict[str, Any]],
    config: ClassificationConfig,
    as_of_date: pd.Timestamp,
) -> TransactionAnalysis:
    """Run the full validation + classification + anomaly stage."""
    classified = [classify_one(record, config) for record in raw_records]
    detect_anomalies(classified, config, as_of_date)

    category_counts: dict[str, int] = {}
    for record in classified:
        category_counts[record.category.value] = category_counts.get(record.category.value, 0) + 1

    anomaly_count = sum(1 for r in classified if r.is_unusual)
    unknown_count = category_counts.get(TransactionCategory.UNKNOWN.value, 0)
    eligible = [r for r in classified if r.is_eligible_revenue]
    eligible_revenue_total = float(sum(r.amount for r in eligible))

    eligible_daily = build_eligible_daily(classified)
    analytics = compute_analytics(eligible_daily, classified, as_of_date)

    warnings: list[str] = []
    if anomaly_count:
        warnings.append(f"{anomaly_count} transaction(s) flagged as unusual (not excluded).")
    if unknown_count:
        warnings.append(f"{unknown_count} transaction(s) could not be classified and were excluded from eligible revenue.")

    return TransactionAnalysis(
        records=classified,
        category_counts=category_counts,
        anomaly_count=anomaly_count,
        unknown_count=unknown_count,
        eligible_revenue_total=eligible_revenue_total,
        eligible_transaction_count=len(eligible),
        eligible_daily=eligible_daily,
        analytics=analytics,
        warnings=warnings,
    )
