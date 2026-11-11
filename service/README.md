# Q-Cash ML and Decision API

A Python service that answers two questions over HTTP:

> "Given this business's recent transaction activity, what will it make in the
> next 30 days, and what early revenue access (ERA) advance and repayment terms
> should we recommend?"

It runs the full Q-Cash assessment pipeline:

1. **Transaction intelligence** — validate, classify and flag transactions.
2. **Revenue prediction** — Random Forest forecast of next-30-day eligible revenue.
3. **Reliability scoring** — weighted, rules-based, explainable score.
4. **Decision + repayment** — ERA advance recommendation and hybrid repayment terms.

> **Synthetic model.** The served revenue model is trained on the synthetic MSME
> population in `qcash_synthetic_v1/`, not on real Kenyan MSMEs. It is a proof of
> concept. Predictions and recommendations are **not** credit decisions and do
> not generalise to real Kenyan MSMEs.

---

## 1. Responsibilities and boundaries

The Python service **recommends**; it never executes financial operations:

* It does **not** disburse funds, deduct repayments, or modify Laravel's database.
* It does **not** query Laravel's PostgreSQL database directly — Laravel supplies
  all assessment inputs.
* It does **not** call live payment providers or make lending decisions.

Laravel ingests transactions, initiates assessments, presents/accepts offers, and
executes disbursement and repayment. Python only analyses and recommends.

---

## 2. Architecture

```
service/
  app/
    main.py                     HTTP endpoints (thin; no business rules)
    assessment.py               orchestrates the four pipeline stages
    transactions.py             validation, classification, anomaly flagging, analytics
    features.py                 90-day model feature engineering (leakage-safe)
    model.py                    loads the trained artifact, predicts revenue
    reliability.py              weighted rules-based reliability scoring
    decision_engine.py          ERA advance recommendation + eligibility statuses
    repayment.py                hybrid repayment recommendation
    policy.py                   typed policy loader (config, weights, thresholds)
    schemas.py                  request/response models
    config.py                   environment-variable settings
    logging_conf.py             structured JSON logging
  policies/
    era_policy_v1.json          versioned, provisional policy document
  tests/                        unit tests (stdlib unittest)
  Dockerfile
  docker-compose.yml
  requirements.txt
  sample_assessment_request.json
```

### Pipeline

```
transactions ──> transaction intelligence ──> eligible daily aggregates
                                                   │
                                                   ├──> revenue prediction (RF)
                                                   │
                                                   └──> reliability scoring
                                                          │
                 ┌───────────────────────────────────────┘
                 ▼
        ERA decision engine ──> advance + repayment recommendation
```

---

## 3. Transaction intelligence

Raw transactions are validated, classified and flagged — never deleted or
altered. Categories:

| Category | Eligible revenue? |
| --- | --- |
| `business_sale` | yes |
| `refund` | no |
| `reversal` | no |
| `capital_injection` | no |
| `internal_transfer` | no |
| `other_non_revenue` | no |
| `unknown` | no |

Rules (all configurable in `policies/era_policy_v1.json`):

* Classification is driven by `status`, `direction`, caller-supplied `category`,
  and `transaction_type` keyword matching.
* Insufficient information -> `unknown` (never a made-up category).
* Unusual transactions (outliers, duplicates, missing IDs, future timestamps) are
  **flagged**, not excluded.
* A large genuine sale remains eligible; a large unclassified inflow is flagged
  and excluded from eligible revenue.

## 4. Revenue prediction

A **Random Forest Regressor** predicts next-30-day eligible revenue from 90 days
of eligible daily aggregates. Feature engineering (`features.py`) is separate
from inference (`model.py`); the trained artifact is loaded once and cached, and
every prediction records its model version.

The model cannot be trusted for a business with fewer than
`minimum_history_days` (default 90) of history; those cases fall through to the
cold-start policy instead of forcing a prediction.

## 5. Reliability scoring

A weighted, rules-based score (not a second Random Forest):

```
S = sum( w_i * s_i )
```

Each component `s_i` is normalised to `[0, 100]` and recorded with its raw value,
weight, flag and reason code. Weights are configurable; the score is **not** a
probability of default. A business with no prior Q-Cash repayment history has
that component marked *unavailable* (weights renormalised) so it is not penalised.

## 6. ERA decision engine

```
Recommended Advance = Eligible Predicted Revenue x Recommended Advance Rate
```

The rate comes from configurable score bands and is bounded by minimum/maximum
advance limits and the exposure headroom Laravel reports. Statuses:

* `eligible`
* `ineligible` (low score, no revenue signal, exposure limit reached)
* `insufficient_data` (too little history / too few transactions)
* `manual_review` (too many unclassified or unusual transactions)

Cold-start businesses (insufficient history) get a conservative, clearly-labelled
assessment only when Laravel supplies business profile/cohort information.

## 7. Hybrid repayment recommendation

* Percentage-based recovery from eligible incoming business payments.
* A maximum repayment period.
* A pre-agreed fallback arrangement if a balance remains at the end of the window.

The recovery percentage is business-specific, starts from a fixed agreed base
rate, and is adjusted only by narrow, documented rules (high volatility or
declining revenue). Python recommends terms; Laravel executes recovery.

---

## 8. Run locally (no Docker)

From the `service/` directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

uvicorn app.main:app --reload --port 8000
```

Then open <http://localhost:8000/docs>.

The model artifact and policy resolve automatically (see `config.py`); set
`MODEL_PATH` / `QCASH_POLICY_PATH` to override.

## 9. Run with Docker

```bash
cd service && docker compose up --build
```

The service is on <http://localhost:8000>.

---

## 10. API reference

All endpoints (except `/health`) require `X-API-Key` when `QCASH_API_KEY` is set.

### `POST /api/v1/assessments`

Full ERA assessment. Request:

```bash
curl -X POST http://localhost:8000/api/v1/assessments \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-key" \
  --data-binary @sample_assessment_request.json
```

The request contains `business_id`, optional `assessment_reference` /
`assessment_date`, a list of `transactions`, and optional `business_profile`,
`repayment_history` and `exposure`. See `schemas.py` and
`sample_assessment_request.json` for the exact shape.

The response contains the assessment status, predicted revenue, reliability score
(with component breakdown), recommended advance rate/amount, recommended recovery
rate and maximum repayment days, reason codes, warnings, model version, policy
version and a transaction summary. Amounts are in **KES**.

### `GET /health`, `GET /model`, `GET /policy`

Liveness, model provenance, and the active (provisional) policy version.

### Legacy forecast endpoints

`POST /predict`, `POST /predict/batch`, `POST /decision` remain for backwards
compatibility with the synthetic revenue-forecast service.

---

## 11. Training, validation and evaluation

Reproducible pipelines live at the repository root and in `qcash_synthetic_v1/`:

```bash
# synthetic population (proof of concept)
.venv/bin/python qcash_synthetic_v1/scripts/generate_transactions.py --n-per-archetype 8 --seed 42
.venv/bin/python qcash_synthetic_v1/scripts/build_supervised_dataset.py
.venv/bin/python qcash_synthetic_v1/scripts/run_experiment.py
```

The experiment uses a chronological train/validation/holdout split (never
shuffled) with an embargo gap so overlapping 30-day targets never leak, and
reports MAE, RMSE, R² and sMAPE. The real (single-business) barbershop pipeline
is `train_qcash.py`; it deliberately reports that no ML configuration beat the
persistence baseline, so the served model is the synthetic Random Forest.

---

## 12. Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `MODEL_PATH` | auto | Trained model bundle (`.joblib`). |
| `QCASH_POLICY_PATH` | `policies/era_policy_v1.json` | Decision policy document. |
| `QCASH_API_KEY` | empty (auth off) | If set, all requests need `X-API-Key`. |
| `QCASH_CORS_ORIGINS` | `*` | Allowed browser origins. |
| `QCASH_LOG_LEVEL` | `INFO` | Logging level. |

---

## 13. Assumptions, provisional policies and open decisions

* **Currency is KES**; all amounts are KSh.
* The input is **transaction records**. Prepared daily aggregates are not yet a
  first-class input path (Laravel can send raw transactions it already ingests).
* **Classification rules are provisional.** They rely on `transaction_type` /
  `category` / `status` fields; where a channel's codes are unknown, inflows are
  marked `unknown` rather than guessed.
* **Scoring weights, advance rates, limits and thresholds are provisional** and
  documented in `policies/era_policy_v1.json`; they have not been calibrated
  against real repayment outcomes and must not be used as a production policy.
* **The revenue model is synthetic** and does not generalise to all Kenyan MSMEs.
* The hybrid repayment fallback is descriptive only; its legal/operational terms
  are owned by Laravel and the financing partner.

---

## 14. Tests

```bash
cd service
.venv/bin/python -m unittest discover -s tests
```
