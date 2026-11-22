# Retraining summary

Each cell below is MAE in KSh (and as % of mean monthly revenue); lower is
better. Synthetic data, seed 20261122.

## What was wrong

1. Money was stored as KSh floats, not integer minor units (engine guide I3).
2. The synthetic generator leaked the answer (transactions rescaled to a
   pre-drawn revenue target, no unobserved noise).
3. Splits were not grouped by business, so the model memorised identity.
4. No robustness to 1-shilling transactions (the real data already contains
   KSh 1); no oracle ceiling, permutation test, or leakage tests.

## What was built

A deterministic synthetic generator (8 sectors, latent AR(1) shock, digital
share, non-revenue events, minimum 1 KSh), leakage-safe features, a
business-grouped + temporal split protocol with leakage tests, a walk-forward
tuning pipeline, and a minor-unit decision engine with a repayment simulation.

## What the numbers say

| model | temporal | unseen businesses | tail |
|---|---|---|---|
| seasonal naive | 28,204 (19%) | 34,067 (19%) | 46,013 (44%) |
| moving average 90d | 24,133 (16%) | 32,272 (18%) | 31,243 (30%) |
| linear regression | 25,414 (17%) | 35,491 (19%) | 28,512 (27%) |
| random forest (level) | 28,905 (19%) | 44,640 (24%) | 34,469 (33%) |
| random forest (growth target) | 27,882 (18%) | 38,305 (21%) | 41,479 (40%) |
| xgboost (level) | 30,394 (20%) | 46,042 (25%) | 30,158 (29%) |
| xgboost (growth target) | 27,757 (18%) | 37,379 (20%) | 39,463 (38%) |
| **xgboost (ratio / MA anchor) — PRODUCTION** | 23,896 (16%) | 32,807 (18%) | 29,535 (28%) |
| oracle (floor) | 18,026 (12%) | 20,928 (11%) | 27,149 (26%) |

* **Tree ensembles overfit business identity on the raw level target**: Random
  Forest is 44,640 (24%) and XGBoost is worse than the naive baseline
  (34,067 (19%)) on unseen businesses.
* **Anchoring the tree on the 90-day moving average fixes it.** The production
  model, XGBoost predicting the ratio `target / (rev_sum_90/3)` with momentum
  features, reaches 23,896 (16%) on temporal and 32,807 (18%) on unseen
  businesses — matching the moving average and beating it on the tail. It stays
  robust to 1-shilling micro businesses.
* The **linear model** remains the most conservative on the tail (shocked)
  businesses.
* Shuffle-target permutation R² ≈ 0.002 (no leakage).
* 90% lower bound covers ~89% on normal cases but only ~50% on tail cases.

## What the numbers do not say

* All results are on **synthetic** data; they do not estimate accuracy on real
  MSMEs.
* Only **one** real business (Belgard Barbers) was available for calibration;
  the "three pilot businesses" in the plan were not present in the repo.
* There are no realised defaults — only simulated shortfalls under a controlled
  collection rule.
* The reliability score, advance rates and margin factors are **policy
  assumptions**, not calibrated findings.

## What to do next

1. Obtain the three real pilot businesses (two barbershops + pharmacy) and run
   the real-vs-synthetic comparison and out-of-distribution sanity check.
2. Replace the point forecast's fixed error margin with a quantile regressor (or
   conformal) so the lower bound is honest on tail cases too.
3. Calibrate `margin_factor_bps`, band weights and recovery rates against
   observed repayment outcomes rather than assumption.
