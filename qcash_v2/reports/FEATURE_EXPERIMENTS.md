# Feature-engineering experiments (what we tried, what worked, what did not)

Two feature-engineering experiments were run against the production
ratio-target XGBoost. Both are reproducible via the scripts named below. Only
the first produced a model change; the second was rejected on evidence.

All numbers are MAE in KSh on the three test sets: `temporal` / `unseen
businesses` / `tail`. Synthetic data, seed 20261122.

## Experiment 1 — the 90-day moving average as a feature (ACCEPTED)

Script: `experiment_ma_feature.py`.

| approach | temporal | unseen biz | tail |
|---|---|---|---|
| moving average (baseline) | 24,133 | 32,272 | 31,243 |
| XGBoost, raw level target | 30,394 | 46,042 | 30,158 |
| XGBoost, +ma/momentum as plain features | 28,738 | 45,535 | 30,946 |
| **XGBoost, ratio target = target / (rev_sum_90/3)** | **23,636** | **32,685** | **29,607** |

Adding the moving average as a plain column changes nothing (for trees it is a
monotonic rescaling of `rev_sum_90`). Using it as the **anchor** — predicting
the *ratio* and multiplying back — fixes the business-identity overfit and
becomes the production model.

## Experiment 2 — five proposed features (REJECTED)

Script: `experiment_new_features.py`. Candidate features: `days_since_last_tx`,
`log_ma_90`, `target_payday_days`, `activity_momentum`, `weekend_rev_share_30`.

Per-feature ablation on the baseline data:

| config | temporal | unseen biz | tail |
|---|---|---|---|
| baseline | 23,516 | 32,583 | 29,682 |
| + days_since_last_tx | 23,717 | 32,653 | 29,169 |
| + log_ma_90 | 23,623 | 32,600 | 29,317 |
| + target_payday_days | 23,604 | 32,590 | 29,307 |
| + activity_momentum | 23,652 | 32,605 | 29,241 |
| + weekend_rev_share_30 | 23,753 | 32,369 | 29,471 |
| + ALL | 23,564 | 32,185 | 29,572 |

None improved the temporal test; the largest unseen-business gain (+ALL, ~1.2%)
is inside seed noise (baseline 32,605 +/- 402 vs +ALL 32,230 +/- 224).

### Why they do not help

| feature | unique values | corr. with MA deviation | reason |
|---|---|---|---|
| days_since_last_tx | 4 | -0.001 | nearly all businesses trade daily; redundant with `rev_sum_7` |
| activity_momentum | 39 | +0.001 | `active_days_7 ~ 7` always (before upgrade) |
| log_ma_90 | 10,730 | +0.039 | monotone transform of `rev_sum_90`, already a feature |
| target_payday_days | 4 | -0.004 | no payday dynamic in the DGP; and a 30-day window averages a monthly cycle out |
| weekend_rev_share_30 | 10,813 | -0.037 | weak but real; within noise |

### Data-generating-process upgrade test

To give the features a fair chance we upgraded the generator with a realistic
end-of-month **payday cycle** and **closure spells** (trading gaps), then
re-ran the ablation (`config.yaml: generator.payday_boost / closure_*`).

| DGP variant | baseline temporal / unseen | +ALL temporal / unseen |
|---|---|---|
| payday + closures (harsh) | 29,798 / 36,240 | 30,092 / 36,395 |
| payday + closures (mild) | 27,774 / 38,118 | 27,768 / 38,168 |

The upgrade gave `days_since_last_tx` (4 -> 17 unique values) and
`activity_momentum` (39 -> 105) real variance, but the features **still did not
improve the model**, and the closures added ~16-19% to MAE as unforecastable
error. The features are genuinely redundant with the existing revenue-window
features, and the future is not forecastable from them.

**Decision:** keep the production model unchanged and leave payday/closure
**disabled** in `config.yaml` (the code remains as configurable options, off by
default). The validated generator reproduces the baseline exactly (1,545,074
transactions). We do not add features the data cannot reward — that is the
overfitting line we set out not to cross.
