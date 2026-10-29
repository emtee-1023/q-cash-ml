# Q-Cash Revenue Forecast Service

A small web service that answers one question over HTTP:

> "Given this business's recent M-Pesa Till activity, how much will it make in
> the next 30 days, and how much early access can we reasonably offer?"

The application sends a business's trailing **daily revenue and transaction
counts** (at least 90 days), and this service returns a 30-day revenue forecast
plus four proposed early-access limits. It runs in its own Docker container.

> **Synthetic model.** The served model is trained on the synthetic MSME
> population in `qcash_synthetic_v1/`, not on the single real barbershop. It is
> a proof of concept: predictions and limits are not credit decisions and do not
> generalise to real Kenyan MSMEs.

---

## 1. What the model does (in plain language)

The model (a random forest on transaction-behaviour features) forecasts the next
30 days of M-Pesa Till revenue from 90 days of daily history. From that forecast
it derives four transparent early-access limits. Because these are short-term
bridge loans, limits are sized in **days of revenue**, not a percentage of a
month, and are capped at KSh 15,000.

| Strategy | Rule |
| --- | --- |
| `simple_revenue` | 5 days of predicted daily revenue |
| `conservative` | 5 days of *typical* (median) daily revenue |
| `revenue_stability` | 5 days, scaled down for volatile businesses |
| `cashflow_coverage` | 5 days × repayment rate (50%) |

The forecast also comes with a plus/minus band: the model's average error on
held-out weeks (about KSh 12,298 at the time of training).

The training population spans both barbershop-scale businesses and micro MSEs
(mama mboga / boda boda scale), so the model can score both.

---

## 2. Files

```
service/
  app/
    main.py        HTTP endpoints
    model.py       loads the artifact, predicts, computes limits
    features.py    computes 90-day features from daily history
    schemas.py     request/response shapes
    config.py      environment-variable settings
  Dockerfile
  docker-compose.yml
  requirements.txt
  .env.example
```

The trained model lives in `../qcash_synthetic_v1/models/service_model_bundle.joblib`
and is copied into the image at build time.

---

## 3. Run it locally (no Docker)

From this `service/` directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export MODEL_PATH=../qcash_synthetic_v1/models/service_model_bundle.joblib
uvicorn app.main:app --reload --port 8000
```

Then open <http://localhost:8000/docs> for an interactive page.

---

## 4. Run it with Docker (recommended)

From the `ml/` directory (the build context must include the model):

```bash
cp service/.env.example service/.env   # optional: set an API key
cd service && docker compose up --build
```

The service is on <http://localhost:8000>.

---

## 5. API reference

### `GET /health`
Liveness probe. Returns `{"status":"ok","model_loaded":true,"model":"..."}`.

### `GET /model`
Training provenance, feature list and caveats. Protected by the API key when
one is configured.

### `POST /predict`
Forecast one business. Send at least 90 days of daily history.

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-key" \
  -d '{
        "business_id": "SYN_008",
        "daily": [
          {"transaction_date": "2026-09-01", "revenue_kes": 5000, "transaction_count": 10},
          {"transaction_date": "2026-09-02", "revenue_kes": 4200, "transaction_count": 8}
        ]
      }'
```

The last date in `daily` is treated as "today"; the model forecasts the next 30
days. Fewer than 90 days returns HTTP 422.

```json
{
  "predicted_next_30d_revenue_kes": 145927.08,
  "range_low_kes": 133628.90,
  "range_high_kes": 158225.27,
  "range_basis": "Prediction plus or minus the model's average error on held-out weeks (KSh 12,298).",
  "limits": {
    "simple_revenue_kes": 15000.00,
    "conservative_kes": 15000.00,
    "revenue_stability_kes": 15000.00,
    "cashflow_coverage_kes": 12160.59
  },
  "model": "random_forest/D_revenue_behavioural_seasonal",
  "target": "M-Pesa Till revenue (Credit) in the 30 days after the latest input day"
}
```

### `POST /predict/batch`
Same as `/predict`, but accepts `{"items": [ ... ]}` and returns several
forecasts at once.

### `POST /decision`
Returns a **single recommended early-access amount** with its reasoning. The
recommendation is the most conservative of the three coverage-aware strategies
(conservative, revenue+stability, cash-flow coverage); the naive days-of-revenue
rule is shown for comparison but never drives the recommendation.

```bash
curl -X POST http://localhost:8000/decision \
  -H "Content-Type: application/json" \
  --data-binary @sample_request.json
```

```json
{
  "recommended_available_kes": 12160.59,
  "binding_strategy": "cashflow_coverage",
  "risk_tier": "low",
  "predicted_next_30d_revenue_kes": 145927.08,
  "range_low_kes": 133628.90,
  "range_high_kes": 158225.27,
  "limits": { "simple_revenue_kes": 15000.00, "conservative_kes": 15000.00,
              "revenue_stability_kes": 15000.00, "cashflow_coverage_kes": 12160.59 },
  "reasoning": "Recommended amount is the most conservative of the coverage-aware strategies (cashflow_coverage). Based on predicted daily revenue of KSh 4,864 (30-day forecast KSh 145,927) with CV 0.24 (risk tier: low).",
  "model": "random_forest/D_revenue_behavioural_seasonal",
  "target": "M-Pesa Till revenue (Credit) in the 30 days after the latest input day"
}
```

---

## 6. Calling it from Laravel

```php
use Illuminate\Support\Facades\Http;

$response = Http::withHeaders(['X-API-Key' => config('services.qcash_ml.key')])
    ->post(config('services.qcash_ml.url').'/predict', [
        'business_id' => $businessId,
        'daily' => $dailyRevenueAndTxCounts, // [{transaction_date, revenue_kes, transaction_count}, ...]
    ])
    ->throw();

$forecast  = $response->json('predicted_next_30d_revenue_kes');
$limit     = $response->json('limits.revenue_stability_kes');
```

Add to `config/services.php`:

```php
'qcash_ml' => [
    'url' => env('QCASH_ML_URL', 'http://qcash-ml:8000'),
    'key' => env('QCASH_ML_KEY'),
],
```

---

## 7. Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `MODEL_PATH` | `/app/artifacts/service_model_bundle.joblib` | Where the `.joblib` file lives. |
| `QCASH_API_KEY` | empty (auth off) | If set, every request must send `X-API-Key: <value>`. **Set this in production.** |
| `QCASH_CORS_ORIGINS` | `*` | Comma-separated allowed browser origins. |

---

## 8. Updating the model

1. Regenerate data and retrain:
   ```bash
   .venv/bin/python qcash_synthetic_v1/scripts/generate_transactions.py
   .venv/bin/python qcash_synthetic_v1/scripts/build_supervised_dataset.py
   .venv/bin/python qcash_synthetic_v1/scripts/run_experiment.py
   ```
   This overwrites `qcash_synthetic_v1/models/service_model_bundle.joblib`.
2. Rebuild the image: `cd service && docker compose up --build`.
