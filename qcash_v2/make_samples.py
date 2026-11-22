"""Generate sample assessment request/response JSON for the Q-Cash engine.

Writes to ``samples/``:

  * ``sample_normal_request.json`` / ``..._response.json``  — a real synthetic
    barbershop around a cut-off.
  * ``sample_micro_1ksh_request.json`` / ``..._response.json`` — a micro vendor
    whose transactions are 1-shilling and small items, to demonstrate that the
    pipeline handles absurd low-value inputs without blowing up.

The request schema is what a caller (Laravel) would send; the response is exactly
what :func:`engine.run_assessment` returns.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from common import DATA_DIR, ROOT
import engine

SAMPLES = ROOT / "samples"

TX_FIELDS = ["ref", "occurred_at", "amount_minor", "type", "is_reversal_of", "reference", "payer_hash"]


def _tx_to_records(sub: pd.DataFrame) -> list[dict]:
    records = []
    for _, row in sub.sort_values("occurred_at").iterrows():
        records.append({
            "ref": str(row["ref"]),
            "occurred_at": pd.Timestamp(row["occurred_at"]).strftime("%Y-%m-%dT%H:%M:%S"),
            "amount_minor": int(row["amount_minor"]),
            "type": str(row["type"]),
            "is_reversal_of": None if pd.isna(row["is_reversal_of"]) else str(row["is_reversal_of"]),
            "reference": None if pd.isna(row["reference"]) else str(row["reference"]),
            "payer_hash": None if pd.isna(row["payer_hash"]) else str(row["payer_hash"]),
        })
    return records


def _request_to_frame(business_id: str, records: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    df["business_id"] = business_id
    df["customer_id"] = None
    df["channel"] = "paybill"
    return df


def _write(name: str, request: dict) -> None:
    business_id = request["business_id"]
    response = engine.run_assessment(
        _request_to_frame(business_id, request["transactions"]), request["sector"]
    )
    (SAMPLES / f"{name}_request.json").write_text(json.dumps(request, indent=2))
    (SAMPLES / f"{name}_response.json").write_text(json.dumps(response, indent=2))
    n_tx = len(request["transactions"])
    print(f"  {name}: {n_tx} transactions -> status {response['status']}"
          + (f", point {response['prediction']['point_minor']/100:,.0f} KSh"
             if "prediction" in response else ""))


def _pick_active_barbershop(tx: pd.DataFrame, businesses: pd.DataFrame, as_of: str) -> str:
    as_of_ts = pd.Timestamp(as_of)
    for biz_id in businesses[businesses["sector"] == "barbershop"]["business_id"]:
        sub = tx[tx["business_id"] == biz_id]
        if sub.empty:
            continue
        if pd.to_datetime(sub["occurred_at"]).max() >= as_of_ts - pd.Timedelta(days=3):
            return biz_id
    return businesses[businesses["sector"] == "barbershop"]["business_id"].iloc[0]


def _micro_1ksh_records() -> list[dict]:
    """A micro greengrocer: 90 days of 1-20 KSh items, mostly 1-shilling."""
    rng = np.random.default_rng(7)
    dates = pd.date_range("2025-09-01", periods=90, freq="D")
    records: list[dict] = []
    i = 0
    for d in dates:
        n = int(rng.integers(1, 5))
        for _ in range(n):
            ksh = int(rng.choice([1, 1, 1, 2, 5, 10, 20]))
            records.append({
                "ref": f"MICRO-{i:05d}",
                "occurred_at": d.strftime("%Y-%m-%dT%H:%M:%S"),
                "amount_minor": ksh * 100,
                "type": "credit",
                "is_reversal_of": None,
                "reference": None,
                "payer_hash": None,
            })
            i += 1
    return records


def main() -> None:
    SAMPLES.mkdir(parents=True, exist_ok=True)
    tx = pd.read_csv(DATA_DIR / "transactions.csv")
    businesses = pd.read_csv(DATA_DIR / "businesses.csv")

    as_of = "2025-11-30"
    biz_id = _pick_active_barbershop(tx, businesses, as_of)
    # 95 days is enough for the 90-day feature windows and keeps the sample small.
    history_start = pd.Timestamp(as_of) - pd.Timedelta(days=95)
    sub = tx[(tx["business_id"] == biz_id)
             & (pd.to_datetime(tx["occurred_at"]) >= history_start)
             & (pd.to_datetime(tx["occurred_at"]) < pd.Timestamp(as_of))]
    _write("sample_normal", {
        "business_id": biz_id,
        "sector": "barbershop",
        "assessment_as_of": pd.Timestamp(as_of).isoformat(),
        "currency": "KES",
        "amount_units": "minor (1 KSh = 100)",
        "transactions": _tx_to_records(sub),
    })

    _write("sample_micro_1ksh", {
        "business_id": "SYN_MICRO_VENDOR",
        "sector": "greengrocer",
        "assessment_as_of": "2025-11-30T00:00:00",
        "currency": "KES",
        "amount_units": "minor (1 KSh = 100)",
        "note": "Micro vendor: mostly 1-shilling items, to test extreme low-value input.",
        "transactions": _micro_1ksh_records(),
    })

    # 14-day starter business: graduated cold-start, small capped advance.
    starter: list[dict] = []
    for d in pd.date_range("2025-11-17", periods=14, freq="D"):
        for _ in range(10):
            starter.append({
                "ref": f"STARTER-{len(starter):05d}",
                "occurred_at": d.strftime("%Y-%m-%dT%H:%M:%S"),
                "amount_minor": 30000,
                "type": "credit",
                "is_reversal_of": None,
                "reference": None,
                "payer_hash": None,
            })
    _write("sample_starter_14d", {
        "business_id": "SYN_STARTER_14D",
        "sector": "barbershop",
        "assessment_as_of": "2025-12-01T00:00:00",
        "currency": "KES",
        "amount_units": "minor (1 KSh = 100)",
        "note": "Only 14 days of history -> graduated cold-start (small starter advance).",
        "transactions": starter,
    })

    # A small informal trader: ~KES 250/day, 14 days -> qualifies for a small loan
    # (min advance lowered to KES 250).
    small: list[dict] = []
    for d in pd.date_range("2025-11-17", periods=14, freq="D"):
        for _ in range(10):
            small.append({
                "ref": f"SMALL-{len(small):05d}",
                "occurred_at": d.strftime("%Y-%m-%dT%H:%M:%S"),
                "amount_minor": 2500,  # KES 25 x 10/day = KES 250/day
                "type": "credit",
                "is_reversal_of": None,
                "reference": None,
                "payer_hash": None,
            })
    _write("sample_small_trader_250perday", {
        "business_id": "SYN_SMALL_250",
        "sector": "greengrocer",
        "assessment_as_of": "2025-12-01T00:00:00",
        "currency": "KES",
        "amount_units": "minor (1 KSh = 100)",
        "note": "Informal trader ~KES 250/day for 14 days -> small starter advance.",
        "transactions": small,
    })

    print(f"Wrote samples to {SAMPLES}")


if __name__ == "__main__":
    main()
