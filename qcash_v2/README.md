# Q-Cash v2 — rebuilt dataset, model and decision engine

A from-scratch rebuild of the Q-Cash Early Revenue Access pipeline, following
`../ML-RETRAINING-PROMPT.md` and `../Q-CASH-DECISION-ENGINE-GUIDE.md`. This
directory is now the **single** Q-Cash ML + decision system; the old FastAPI
`service/` app has been retired to `../legacy/service/`.

The headline result: raw-level tree models (Random Forest, XGBoost) **overfit
business identity**, so the production forecast **anchors on a 90-day moving
average** and an XGBoost model predicts the *ratio* `target / (rev_sum_90/3)`
with momentum features (the "moving average as a feature" pattern). It ties the
moving average on unseen businesses, beats it on the temporal and tail tests, and
is robust to **1-shilling transactions** (integer minor units everywhere,
minimum generated transaction 1 KSh = 100 minor units).

> Everything here is synthetic. Numbers demonstrate pipeline correctness, not
> accuracy on real MSMEs.

## Reproduce everything

```bash
# from the ml/ repo root, using the project venv
.venv/bin/python qcash_v2/run_all.py
```

Or step by step:

```bash
cd qcash_v2
../.venv/bin/python generator.py      # data/*.csv
../.venv/bin/python features.py       # data/supervised.csv
../.venv/bin/python train.py          # outputs/model.joblib + metrics.json
../.venv/bin/python simulate.py       # outputs/simulation.csv
../.venv/bin/python report.py         # reports/*.md + plots/*.png
../.venv/bin/python make_samples.py   # samples/*.json
../.venv/bin/python tests/test_pipeline.py
```

## Layout

```
qcash_v2/
  config.yaml         all parameters (money in minor units, rates in bps)
  generator.py        deterministic synthetic DGP (latent AR(1) shock, min 1 KSh)
  features.py         leakage-safe feature engineering + target
  model_features.py   shared feature frames (base + MA/momentum), used by train & engine
  splits.py           business-grouped + temporal splits, leakage tests
  train.py            baselines + linear + RF + XGBoost, walk-forward, overfit controls
  engine.py           THE decision engine: transaction intelligence + reliability + decision + terms
  simulate.py         decision-engine repayment simulation
  make_samples.py     sample request/response JSON for the engine
  report.py           generates reports/ and plots/
  run_all.py          orchestrator
  tests/test_pipeline.py
  data/  outputs/  reports/  plots/  samples/
```

## Decision engine (`engine.py`)

`engine.run_assessment(transactions, sector, ...)` is the single entry point. It
runs the four guide stages in integer minor units:

1. **Transaction intelligence** — classify eligible revenue; strip reversals,
   refunds, own transfers, capital injections, duplicates and anomalies.
2. **Feature construction** — `compute_feature_row` + `model_features.enhanced_frame`
   (identical to training).
3. **Revenue prediction** — the production ratio-anchored XGBoost
   (`outputs/model.joblib`), `point = ma_90 × mean_ratio`, with a lower bound
   `point × (1 − k·cv_error)`.
4. **Reliability + decision + terms** — six-component weighted score, advance
   rate by band, affordability / exposure caps, and a `binding_constraint`,
   exactly as specified in the decision-engine guide.

### Minimum history and graduated cold-start

A business can be assessed from **14 days** of trading (KopoKopo-style), but the
advance is graduated by history length so short-history businesses get a **smaller,
more conservative** offer:

| observed days | tier | forecast | advance rate | max advance |
|---|---|---|---|---|
| < 14 | refused (`R01`) | — | — | — |
| 14–29 | starter | run-rate | 800 bps | KES 2,000 |
| 30–59 | limited | run-rate | 1200 bps | KES 5,000 |
| 60–89 | standard | run-rate | 1800 bps | KES 10,000 |
| ≥90 + ≥20 tx/30d | full | ratio-XGB on `ma_90` | band 3000/2200/1500/1000 | KES 25,000 |

The full 90-day-anchored model is only used at the `full` tier (below 90 days it
would be out-of-distribution). See `reports/COLD_START.md` for the accuracy-vs-
history evidence and the per-tier worked example.

## Samples

`samples/` holds request/response pairs produced by the actual engine:

* `sample_normal_*` — a synthetic barbershop around a cut-off (eligible offer).
* `sample_micro_1ksh_*` — a micro vendor doing **1-shilling** items; the engine
  forecasts a tiny amount and correctly declines (`R02`, below minimum), proving
  it handles absurd low-value inputs without blowing up.

Regenerate with `../.venv/bin/python make_samples.py`.

## Key reports

* `reports/RETRAINING_SUMMARY.md` — two-page summary for a supervisor
* `reports/EVALUATION.md` — metrics, overfitting findings, honest caveats
* `reports/COLD_START.md` — 14-day minimum, graduated advance tiers, accuracy-vs-history
* `reports/FEATURE_EXPERIMENTS.md` — features tested, what was adopted/rejected and why
* `reports/AUDIT.md` — what was wrong before and what changed
* `reports/SIMULATION.md` — repayment policy simulation
* `reports/VALIDATION.md` — real vs synthetic comparison + oracle ceiling
* `reports/SPLITS.md` — partition manifest + leakage tests
* `reports/DGP.md` — data-generating process and assumptions
* `reports/FEATURES.md` — feature definitions

## Dependencies

`../requirements.txt` plus `pyyaml` and `xgboost` (installed into the project venv).
