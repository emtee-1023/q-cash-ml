# Q-Cash Decision Engine — Backend API Reference

HTTP reference for the Q-Cash assessment service (`qcash_v2/api.py`, FastAPI).
It wraps `engine.run_assessment`: transaction intelligence → revenue prediction →
reliability score → advance recommendation → repayment terms.

- **Base URL:** `http://<host>:3004` (VPS host port 3004 → container 8000)
- **Service version:** `2.0.0`
- **Live schema:** `GET /openapi.json` and interactive `GET /docs`
- **Nature:** synthetic proof-of-concept. Recommendations are **not** credit
  decisions and do not establish real-world accuracy.

---

## 1. Units and conventions

| Concept | Representation |
|---|---|
| Money | **Integer minor units** — `1 KSh = 100`. **No decimals, no floats.** |
| Rates | Integer **basis points** — `1% = 100 bps`, `100% = 10000 bps` |
| Timestamps | ISO-8601; timezone is stripped to naive UTC on ingest |
| Horizon | 30 calendar days (the prediction window) |

> The whole project operates in minor units to avoid floating-point rounding
> errors. Clients send and receive `*_minor` integers only. To display KSh,
> divide by 100 at the presentation layer; never send fractional amounts.

Field naming: `*_minor` = money in minor units, `*_bps` = basis points,
`*_days` = days, `*_score` = 0–100.

---

## 2. Authentication

Optional shared secret. If the server was started with `QCASH_API_KEY` set, every
`POST /assess` request must include the header:

```
X-API-Key: <the configured key>
```

If no key is configured on the server, the header is not checked. `GET /health`
and `GET /model` are unauthenticated.

---

## 3. Endpoints

| Method | Path | Purpose | Auth |
|---|---|---|---|
| `POST` | `/assess` | Full assessment for one business | optional |
| `GET` | `/health` | Liveness + model presence | no |
| `GET` | `/model` | Model metadata (no estimator) | no |

---

## 4. `POST /assess`

### 4.1 Request body

```json
{
  "business_id": "SYN_001",
  "sector": "barbershop",
  "assessment_as_of": "2025-11-30T00:00:00",
  "transactions": [
    {
      "ref": "T0001",
      "occurred_at": "2025-11-30T10:15:00",
      "amount_minor": 30000,
      "type": "credit",
      "is_reversal_of": null,
      "reference": null,
      "payer_hash": null
    }
  ],
  "repayment": null,
  "exposure": null,
  "business_profile": null
}
```

#### Top-level fields

| Field | Type | Required | Notes |
|---|---|---|---|
| `business_id` | string | **yes** | Caller identifier; echoed nowhere in the response but used for logging |
| `sector` | string | **yes** | One of §8; unknown values fall back to a generic margin (3500 bps) |
| `transactions` | array | **yes** (may be empty) | See §4.2. Empty/too few → `insufficient_data` |
| `assessment_as_of` | datetime / null | no | **Accepted but not used.** The cut-off is the day after the latest `occurred_at`. Do not send transactions dated after the intended cut-off |
| `repayment` | object / null | no | See §4.3; supplies the reliability `repayment` component |
| `exposure` | object / null | no | See §4.4; drives R03/R04 and exposure caps |
| `business_profile` | object / null | no | Accepted, reserved (not used by the current engine) |

Unknown extra fields are ignored.

#### 4.2 Transaction object

| Field | Type | Required | Meaning |
|---|---|---|---|
| `ref` | string | **yes** | Unique reference. Repeated `ref` → classified `duplicate` |
| `occurred_at` | datetime | **yes** | Transaction timestamp |
| `amount_minor` | integer | **yes** | Amount in minor units (positive) |
| `type` | `credit` \| `debit` | no (default `credit`) | Flow direction |
| `is_reversal_of` | string / null | no | `ref` of the original credit this debit reverses |
| `reference` | string / null | no | Free-text reference; keyword-scanned for classification |
| `payer_hash` | string / null | no | Payer identifier; `own-` prefix marks an own transfer |

**Classification semantics** (how these fields decide eligible revenue):

| Input pattern | Predicted category | Counts as revenue |
|---|---|---|
| `type = credit` (default, no keywords) | `revenue_eligible` | yes |
| `type = debit` with `is_reversal_of` set | `non_revenue_reversal` | no |
| `type = debit` without `is_reversal_of` | `non_revenue_refund` | no |
| `type = credit`, `reference` contains `loan`/`grant`/`sacco`/`capital` | `non_revenue_capital` | no |
| `type = credit`, `reference` contains `own_transfer`/`transfer`/`internal` | `non_revenue_own_transfer` | no |
| `type = credit`, `payer_hash` begins `own-` | `non_revenue_own_transfer` | no |
| duplicate `ref` (2nd+ occurrence) | `duplicate` | no |
| credit whose amount is a robust outlier (z > 3.5) | `ambiguous` | no (flagged) |

Outflows (`debit`) should carry **positive** `amount_minor` and `type: "debit"`.

#### 4.3 `repayment` object (optional)

| Field | Type | Meaning |
|---|---|---|
| `settled_count` | integer | Number of settled prior advances |
| `on_time_count` | integer | Number settled on time |

The `repayment` reliability component is used only when `settled_count >= 2`;
otherwise it is marked unavailable and the remaining weights are re-normalised.

#### 4.4 `exposure` object (optional)

| Field | Type | Default | Meaning |
|---|---|---|---|
| `outstanding_minor` | integer | `0` | Business outstanding principal |
| `max_exposure_minor` | integer | policy max | Business exposure cap |
| `owner_outstanding_minor` | integer | `0` | Owner's aggregate outstanding across businesses |
| `overdue_minor` | integer | `0` | Overdue amount; `> 0` ⇒ R03 |
| `has_overdue` | boolean | `false` | Alternate overdue flag ⇒ R03 |

---

## 5. Response

The response shape depends on `status`. Top-level `status` ∈
`eligible | not_eligible | insufficient_data`.

### 5.1 `eligible`

```json
{
  "status": "eligible",
  "model_version": "xgb-ratio-ma90-2026.11.0",
  "history_tier": "full",
  "observed_days": 95,
  "intelligence": {
    "counts": {"revenue_eligible": 657, "non_revenue_refund": 7, "duplicate": 2},
    "value_minor": {"eligible": 20695100},
    "anomalies_flagged": 3
  },
  "prediction": {
    "horizon_days": 30,
    "point_minor": 6683598,
    "lower_bound_minor": 5655837,
    "method": "xgboost_ratio_ma90"
  },
  "reliability": {
    "score": 81.12,
    "band": "A",
    "components": {
      "consistency":  {"score": 96.67, "weight_bps": 2500, "used": true},
      "volatility":   {"score": 44.08, "weight_bps": 2000, "used": true},
      "activity":     {"score": 96.67, "weight_bps": 2000, "used": true},
      "trend":        {"score": 56.99, "weight_bps": 1000, "used": true},
      "concentration":{"score": 99.93, "weight_bps": 1500, "used": true},
      "repayment":    {"score": null,  "weight_bps": 1000, "used": false}
    },
    "weights_renormalised": false
  },
  "recommendation": {
    "status": "eligible",
    "amount_minor": 1315000,
    "advance_rate_bps": 3000,
    "binding_constraint": "affordability",
    "reason_codes": [],
    "fee_minor": 39450,
    "total_repayable_minor": 1354450,
    "affordability_capacity_minor": 3393502
  },
  "terms": {
    "term_days": 30,
    "max_repayment_days": 60,
    "recovery_rate_bps": 1500,
    "daily_cap_bps": 3000,
    "fallback": "extend_once_then_partner_review"
  },
  "reason_codes": []
}
```

### 5.2 `not_eligible`

Same top-level fields as §5.1 (`model_version`, `history_tier`,
`observed_days`, `intelligence`, `prediction`, `reliability`, `terms`), but
`recommendation` is reduced and `status`/`reason_codes` indicate the block:

```json
{
  "status": "not_eligible",
  "recommendation": {
    "status": "not_eligible",
    "amount_minor": 0,
    "advance_rate_bps": 2200,
    "binding_constraint": "affordability",
    "reason_codes": ["R02", "R05"]
  },
  "reason_codes": ["R02", "R05"]
}
```

`amount_minor` is always `0`; there are **no** `fee_minor`,
`total_repayable_minor` or `affordability_capacity_minor` fields.

### 5.3 `insufficient_data`

Returned before prediction/reliability when there isn't enough history:

```json
{
  "status": "insufficient_data",
  "reason_codes": ["R01"],
  "detail": {
    "observed_days": 1,
    "required_days": 14,
    "eligible_tx": 10,
    "required_txn": 10
  }
}
```

Only `status`, `reason_codes`, `detail` are present. There is **no**
`prediction`, `reliability`, `recommendation`, `terms`, `history_tier` or
`model_version`.

---

## 6. Response field reference

| Field | Type | Present when | Meaning |
|---|---|---|---|
| `status` | enum | always | `eligible` / `not_eligible` / `insufficient_data` |
| `model_version` | string | eligible/not_eligible | `xgb-ratio-ma90-2026.11.0` (full) or `run_rate_30d[<tier>]` (short history) |
| `history_tier` | enum | eligible/not_eligible | `starter` / `limited` / `standard` / `full` |
| `observed_days` | integer | eligible/not_eligible | Calendar days of history in the request |
| `intelligence.counts` | object | eligible/not_eligible | Predicted category → count |
| `intelligence.value_minor.eligible` | integer | eligible/not_eligible | Total eligible revenue (minor) |
| `intelligence.anomalies_flagged` | integer | eligible/not_eligible | Count of `ambiguous` transactions |
| `prediction.horizon_days` | integer | eligible/not_eligible | Always `30` |
| `prediction.point_minor` | integer | eligible/not_eligible | Point forecast (minor) |
| `prediction.lower_bound_minor` | integer | eligible/not_eligible | Conservative lower bound used for sizing (minor) |
| `prediction.method` | enum | eligible/not_eligible | `xgboost_ratio_ma90` (full) / `run_rate_30d` (short) |
| `reliability.score` | number | eligible/not_eligible | 0–100 weighted score |
| `reliability.band` | enum | eligible/not_eligible | `A` / `B` / `C` / `D` |
| `reliability.components` | object | eligible/not_eligible | Six components, each `{score, weight_bps, used}`; `score: null` + `used: false` when unavailable |
| `reliability.weights_renormalised` | boolean | eligible/not_eligible | True when a component was dropped and weights re-normalised |
| `recommendation.status` | enum | eligible/not_eligible | Mirrors top-level status |
| `recommendation.amount_minor` | integer | eligible/not_eligible | Final advance (minor); `0` if not eligible |
| `recommendation.advance_rate_bps` | integer | eligible/not_eligible | Rate applied (band rate, or tier rate for short history) |
| `recommendation.binding_constraint` | enum | eligible/not_eligible | Which cap limited the amount (§7.2) |
| `recommendation.reason_codes` | array | eligible/not_eligible | See §7.1 |
| `recommendation.fee_minor` | integer | eligible only | Fee (3% of principal) |
| `recommendation.total_repayable_minor` | integer | eligible only | `amount + fee` |
| `recommendation.affordability_capacity_minor` | integer | eligible only | Monthly affordability capacity before share/fee |
| `terms.term_days` | integer | eligible/not_eligible | Contractual term (30) |
| `terms.max_repayment_days` | integer | eligible/not_eligible | Hard limit (60) |
| `terms.recovery_rate_bps` | integer | eligible/not_eligible | Per-receipt recovery rate by band |
| `terms.daily_cap_bps` | integer | eligible/not_eligible | Daily collection cap (3000 bps of receipts) |
| `terms.fallback` | string | eligible/not_eligible | `extend_once_then_partner_review` |
| `reason_codes` | array | always | Top-level mirror of `recommendation.reason_codes` (or `["R01"]`) |

---

## 7. Enumerations

### 7.1 Reason codes

Returned sorted, and **every applicable code is included**.

| Code | Condition | Status |
|---|---|---|
| `R01` | Insufficient history (`observed_days < 14` or eligible tx in window `< 10`) | `insufficient_data` |
| `R02` | Final amount below the minimum (KES 250) after all caps | `not_eligible` |
| `R03` | Overdue advance outstanding (`exposure.overdue_minor > 0` / `has_overdue`) | `not_eligible` |
| `R04` | Exposure cap reached (business or owner headroom `≤ 0`) | `not_eligible` |
| `R05` | Affordability leaves less than the minimum | `not_eligible` |

`R06` (consent), `R07` (paybill verification), `R08` (partner declined) are
**reserved and not emitted** by the current engine.

### 7.2 `binding_constraint`

| Value | Meaning |
|---|---|
| `rate` | The band/tier advance rate × lower bound was the smallest cap |
| `policy_max` | The maximum advance (global or tier cap) bound the amount |
| `affordability` | The margin-based affordability capacity bound the amount |
| `exposure` | Business exposure headroom bound the amount |
| `owner_exposure` | Owner aggregate exposure headroom bound the amount |

### 7.3 `history_tier` and volume caps

| Tier | Observed days | Forecast | Advance rate | Max advance |
|---|---|---|---|---|
| `starter` | 14–29 | `run_rate_30d` | 800 bps | KES 2,000 |
| `limited` | 30–59 | `run_rate_30d` | 1200 bps | KES 5,000 |
| `standard` | 60–89 | `run_rate_30d` | 1800 bps | KES 10,000 |
| `full` | ≥90 **and** ≥20 eligible tx in last 30 d | `xgboost_ratio_ma90` | band 3000/2200/1500/1000 | KES 25,000 |

If `observed_days ≥ 90` but recent transactions `< 20`, the `full` tier is
downgraded to `standard`. `observed_days < 14` ⇒ R01. Minimum advance: **KES 250**.

### 7.4 Reliability band and advance rate (full tier)

| Band | Score | Advance rate |
|---|---|---|
| `A` | ≥ 75 | 3000 bps |
| `B` | ≥ 60 | 2200 bps |
| `C` | ≥ 45 | 1500 bps |
| `D` | < 45 | 1000 bps |

---

## 8. Sectors (margin factors used in affordability)

| `sector` | Margin factor |
|---|---|
| `barbershop`, `salon` | 6000 bps |
| `food_vendor` | 4500 bps |
| `greengrocer`, `secondhand_clothing` | 4000 bps |
| `general_retail` | 3500 bps |
| `hardware` | 3000 bps |
| `pharmacy` | 2500 bps |

Any other value falls back to 3500 bps.

---

## 9. Other endpoints

### `GET /health`
```json
{"status": "ok", "model_present": true, "model_path": "/app/outputs/model.joblib"}
```

### `GET /model`
```json
{
  "model_type": "xgboost_ratio_ma90",
  "model_version": "xgb-ratio-ma90-2026.11.0",
  "cv_error": 0.1538,
  "error_margin_k": 1.0,
  "n_features": 27,
  "selection_rationale": "XGBoost on the ratio target ..."
}
```

---

## 10. Errors

| HTTP | When | Body |
|---|---|---|
| `401` | `X-API-Key` configured on server but missing/wrong | `{"detail": "Invalid or missing X-API-Key header."}` |
| `422` | Request fails schema validation (missing required field, wrong type) | FastAPI validation detail |
| `503` | `GET /model` when the artifact is missing | `{"detail": "Model artifact not found."}` |
| `500` | `POST /assess` when the model artifact is missing at load time | check `GET /health` → `model_present` |

A successful assessment always returns HTTP `200`, even when `status` is
`not_eligible` or `insufficient_data`.

---

## 11. Examples

Reference payloads are in `samples/` (regenerate with `make_samples.py`):

| File | Scenario | Outcome |
|---|---|---|
| `samples/sample_normal_*` | Barbershop, ≥90 days | `eligible`, `full` tier |
| `samples/sample_starter_14d_*` | 14 days of history | `eligible`, `starter` tier (capped) |
| `samples/sample_small_trader_250perday_*` | ~KES 250/day, 14 days | `eligible`, small advance |
| `samples/sample_micro_1ksh_*` | KES 1 transactions | `not_eligible`, `["R02","R05"]` |
