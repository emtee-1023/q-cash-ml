# Split protocol

Businesses are split first (a business appears in exactly one business
partition), then by time. A 35-day embargo gaps train and test to avoid 30-day
target overlap.

| Partition | rows | businesses | date range |
|---|---|---|---|
| train | 4951 | 90 | 2024-03-31 → 2025-04-20 |
| validation | 664 | 83 | 2025-06-01 → 2025-07-20 |
| test_temporal | 1162 | 83 | 2025-08-31 → 2025-11-30 |
| test_business | 490 | 35 | 2025-08-31 → 2025-11-30 |
| test_tail | 114 | 62 | 2025-08-31 → 2025-11-16 |

**Leakage tests** (must all pass):

| Test | Result |
|---|---|
| business identity not a feature | PASS |
| train/test business + date overlap | PASS |
| business grouped (disjoint) | PASS |
| no month in both train and test_temporal | PASS |

Walk-forward (expanding window) folds select hyperparameters; the test sets are
opened once, at the end.
