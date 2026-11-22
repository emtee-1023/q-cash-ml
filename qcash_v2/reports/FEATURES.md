# Feature definitions

Every feature is computed from eligible Paybill transactions strictly **before**
the cut-off. Money features are integer minor units. `sector` is a categorical
(one-hot); `business_id` is never a feature.

| Feature | Window | Definition |
|---|---|---|
| `rev_sum_7/30/90` | 7/30/90 d | sum of eligible `amount_minor` |
| `txn_count_30` | 30 d | count of eligible transactions |
| `freq_per_active_week` | 30 d | eligible tx ÷ active weeks (guard ≥ 1) |
| `active_days_30` | 30 d | days with ≥ 1 eligible receipt |
| `adr_pct` | 30 d | active days ÷ 30 × 100 |
| `mean_txn_minor` | 30 d | mean eligible amount |
| `cv_daily_rev_30` | 30 d | σ daily revenue ÷ mean daily revenue |
| `trend_pct` | 14 d | (mean last 14 − mean prior 14) ÷ prior × 100, clamped ±100 |
| `hhi_30` | 30 d | Σ pᵢ² over transactions |
| `top1_share_30` | 30 d | largest single transaction ÷ total |
| `tenure_days` | all | days since first eligible receipt |
| `sector_*` | — | one-hot sector |

**Target** `target_minor` = sum of eligible Paybill revenue in the 30 days after
the cut-off. Zero-revenue days are included in the daily series.
