# Q-Cash Synthetic MSME Experiment (Proof of Concept)

A controlled, reproducible synthetic-data experiment that investigates whether
transaction-behaviour features can support a **defensible early revenue access
limit** for MSMEs — and how such a limit behaves under stress.

This directory is **separate** from the real Belgard Barbers dataset and its
existing results (in `../dataset_v1` and `../outputs`). Nothing here modifies or
mixes with the real data.

> **Important:** everything here is a simulation. The parameters are simulation
> assumptions (not measured Kenyan MSME statistics), and the repayment outcomes
> are controlled simulation rules (not observed borrower behaviour). Results do
> **not** generalise to real Kenyan MSMEs.

---

## Layout

```
qcash_synthetic_v1/
  data/
    synthetic_transactions.csv        transaction-level records
    synthetic_daily_metrics.csv       per-business daily aggregates
    synthetic_supervised_weekly.csv   leakage-safe ML feature dataset
    synthetic_metadata.json           generation provenance + assumptions
  scripts/
    generate_transactions.py          synthetic population generator
    build_supervised_dataset.py       feature engineering + multi-target
    run_experiment.py                 forecasting, limits, stress tests
  models/
    selected_forecast_model.joblib    fitted model chosen by validation
  outputs/
    forecast_metrics.csv              all model x feature-set metrics
    feature_ablation.csv              ablation results (holdout)
    feature_importance.csv            permutation importance
    stress_test_results.csv           per-observation stress outcomes
    stress_summary.csv                aggregated sustainability
    business_limit_analysis.csv       per-business limit/sustainability table
    run_summary.json                  experiment summary
    plot_*.png                        figures
  reports/
    experiment_report.md              concise generated report
```

## Reproducing the experiment

```bash
# from the ml/ directory, using the project venv
.venv/bin/python qcash_synthetic_v1/scripts/generate_transactions.py \
    --n-per-archetype 8 --seed 42

.venv/bin/python qcash_synthetic_v1/scripts/build_supervised_dataset.py

.venv/bin/python qcash_synthetic_v1/scripts/run_experiment.py
```

A single fixed master seed (`42`) drives every random draw; each business gets a
derived seed so the population is regenerated identically.

## Business archetypes

| Key | Label | Behaviour |
| --- | --- | --- |
| `A_stable_barbershop` | Stable barbershop | consistent revenue, moderate frequency, low volatility |
| `B_volatile_retail` | Highly volatile retail | irregular frequency, large daily swings, high CV |
| `C_growing` | Growing business | increasing revenue + transaction frequency |
| `D_declining` | Declining business | decreasing revenue + frequency |
| `E_weekly_seasonal` | Strong weekly seasonality | strong day-of-week pattern (weekend-heavy) |
| `F_monthly_seasonal` | Strong seasonal business | recurring high/low months |
| `G_intermittent` | Intermittent / irregular | many low/no-revenue days, occasional spikes |
| `H_high_volume` | Stable high-volume | high transaction count, predictable revenue |
| `I_micro_vendor` | Micro vendor (mama mboga) | small frequent low-value sales, market days, many low days |
| `J_micro_service` | Micro service (boda boda / fundi) | few larger jobs, irregular days, occasional spikes |

## Generation model (assumptions)

- Daily revenue = log-normal multiplicative noise around a base level, with
  per-archetype coefficient of variation.
- Explicit zero/low-revenue and unusually-high-revenue days are Bernoulli events.
- Growth/decline is geometric day-over-day.
- Weekly and monthly seasonality are multiplicative profiles.
- Transaction counts are Poisson with mean `revenue / average_transaction_value`;
  per-transaction amounts are log-normal and rescaled so each day's transactions
  sum to that day's revenue.

## What the experiment answers

1. **Does behavioural information improve prediction beyond recent revenue?**
   (feature ablation A → D)
2. **Do days-of-revenue limits set sustainable amounts for short-term bridge
   loans?** (four limit strategies, soft-capped)
3. **How do limits behave under stress?** (five scenarios)
4. **What separates a sustainable business from an unsustainable one?**

Limits are sized to **days of predicted daily revenue** (base 5 days, repayment
rate 50%, soft cap KSh 15,000) rather than a percentage of monthly revenue, so
they stay small and short-term-appropriate for micro MSEs.

See `reports/experiment_report.md` and `outputs/` for the results.
