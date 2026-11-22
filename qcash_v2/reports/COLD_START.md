# Graduated cold-start (14-day minimum)

_Product decision: allow assessment after **14 days** of trading (KopoKopo-style)
instead of 60, but give short-history businesses a **smaller, heavily haircut
advance**. This is a provisional policy, not a validated credit rule._

## Why the advance must be smaller

A run-rate forecast (`mean daily eligible revenue × 30`) becomes less accurate as
history shrinks. Pooled over all synthetic businesses and weekly cut-offs
(`experiment_short_history.py`):

| history | median abs % error | within ±25% | R² |
|---|---|---|---|
| 7 days | 21% | 57% | 0.75 |
| **14 days** | **18%** | **66%** | **0.83** |
| 21 days | 16% | 71% | 0.85 |
| 30 days | 15% | 75% | 0.88 |
| 60 days | 13% | 80% | 0.89 |
| 90 days | 13% | 81% | 0.90 |

At 14 days you are roughly "two-thirds within ±25%", versus ~80% at 90 days. So a
14-day business can be served, but the forecast error is materially higher and
the advance must be smaller and more conservative.

## Tiers (config.yaml → `engine.history_tiers`)

| tier | observed days | forecast | uncertainty haircut | advance rate | max advance |
|---|---|---|---|---|---|
| starter | 14–29 | run-rate (14d mean × 30) | 45% | 800 bps (8%) | KES 2,000 |
| limited | 30–59 | run-rate | 35% | 1200 bps (12%) | KES 5,000 |
| standard | 60–89 | run-rate | 25% | 1800 bps (18%) | KES 10,000 |
| full | ≥90 (and ≥20 tx/30d) | ratio-XGB on ma_90 | model cv_error | band-based 3000/2200/1500/1000 | KES 25,000 |

Rules:
- Absolute floor to be assessed at all: **14 calendar days** and **≥10 eligible
  transactions** over the observed window; otherwise `insufficient_data` (R01).
- Every tier still passes through the same affordability, exposure and
  owner-exposure caps; `min_advance` (KES 500) still applies. The tier cap and
  rate usually bind first for short history.
- The reliability score is still computed and returned (it is an indicator, not
  a probability), but for short-history tiers it does **not** raise the advance
  rate — the tier rate/cap governs.

## Worked example (barbershop, ~KES 3,000/day)

| history | tier | point (KSh) | lower bound | advance | binding |
|---|---|---|---|---|---|
| 10 d | refused | — | — | — | R01 |
| 14 d | starter | 90,000 | 49,500 | **2,000** | policy_max |
| 30 d | limited | 90,000 | 58,500 | 5,000 | policy_max |
| 60 d | standard | 90,000 | 67,500 | 10,000 | policy_max |
| 90 d | full | 89,092 | 75,392 | 17,550 | affordability |

## Minimum advance lowered to KES 250

The minimum advance is now **KES 250** (`min_advance_minor: 25000`), down from
KES 500, to include informal traders who turn over little. Under the starter tier
(8% of a 55%-haircut lower bound) the daily revenue needed to clear a given
minimum is `minimum ÷ 1.32`:

| minimum advance | daily revenue needed |
|---|---|
| KES 500 (old) | ~KES 379/day |
| **KES 250 (now)** | **~KES 189/day** |
| KES 100 (option) | ~KES 76/day |

14-day examples (`sample_small_trader_250perday`):

| daily revenue | point | lower bound | advance | outcome |
|---|---|---|---|---|
| KES 10/day (1-KES items) | 300 | 165 | 0 | declined (R02, `rate`) |
| KES 100/day | 3,000 | 1,650 | 0 | declined (R02, `rate`) |
| **KES 250/day** | **7,500** | **4,125** | **300** | **eligible** |
| KES 1,000/day | 30,000 | 16,500 | 1,300 | eligible |

So a trader making ~KES 250/day now receives a **KES 300** starter advance, while
Kes-10/day and KES-100/day businesses are still correctly declined — they cannot
service even a KES 250 loan.

## Caveats

- The run-rate forecast and tier parameters are **policy assumptions**, not
  calibrated against realised repayment. Short-history lending is inherently
  higher-risk; the KES 2,000 starter cap is deliberately small.
- The 90-day-anchored production model is **only** used at the `full` tier;
  applying it below 90 days would be out-of-distribution.
- **Small-ticket economics:** at a KES 250-300 advance the 3% fee is only ~KES 8-9.
  Fixed disbursement/collection costs and the higher relative default risk on
  tiny tickets may make such loans uneconomic; this is a policy/inclusion choice
  to be validated against real short-history repayment, not a proven unit cost.
  If a lower floor is desired (e.g. KES 100 → ~KES 76/day traders), change
  `min_advance_minor` in `config.yaml` and re-run the simulation.

