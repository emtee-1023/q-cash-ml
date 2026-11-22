# Legacy archive

This directory holds superseded code and data, preserved for history (the
retraining brief requires that prior work be archived, not deleted).

## `legacy/service/` — retired FastAPI decision service

**Status:** retired. **Superseded by:** `qcash_v2/engine.py`.

The original `service/` app was a standalone FastAPI wrapper (transaction
intelligence → revenue forecast → reliability → ERA recommendation) built around
the **old** synthetic Random-Forest model in `qcash_synthetic_v1/`. It used KSh
floats and a different feature pipeline.

The rebuild in `qcash_v2/` supersedes it:

| Concern | old `legacy/service/` | current `qcash_v2/` |
|---|---|---|
| Money | KSh floats | integer minor units (1 KSh = 100) |
| Forecast model | Random Forest (level target) | XGBoost, ratio target anchored on the 90-day moving average |
| Feature pipeline | `service/app/features.py` | `qcash_v2/model_features.py` + `engine.py` |
| Decision engine | `service/app/decision_engine.py` | `qcash_v2/engine.py` |
| Samples | `sample_*.json` here | `qcash_v2/samples/` |

Nothing in `qcash_v2/` imports from this directory. It is kept only as a record
of the earlier implementation. Do not wire it back in without an explicit
decision to re-adopt the KSh-float design.

Its `.venv/` was deleted (regenerable) to keep the archive small.
