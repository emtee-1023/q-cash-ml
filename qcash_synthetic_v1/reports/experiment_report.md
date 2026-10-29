# Q-Cash Synthetic MSME Experiment Report

_Generated 2026-10-01T22:20:30.603339+00:00 — synthetic data only._

> This report describes a proof-of-concept simulation. All data are synthetic. Repayment outcomes are controlled simulation rules, not observed borrower behaviour, and nothing here establishes generalisation to real Kenyan MSMEs.

## 1. Why synthetic data?

The observed dataset is a single barbershop, on which simple recent-revenue baselines already beat the ML models. To investigate whether transaction-behaviour features can support a *defensible* early-revenue-access limit — and how such a system behaves under stress — we generated a controlled population of synthetic businesses with distinct, known cash-flow behaviours.

## 2. Synthetic population

- 80 businesses across 8 archetypes: stable barbershop, highly volatile retail, growing, declining, weekly-seasonal, monthly-seasonal, intermittent, and stable high-volume.
- Daily revenue follows a log-normal multiplicative noise model with per-archetype coefficient of variation, explicit zero/low and high-revenue day probabilities, geometric growth/decline, and weekly/monthly seasonality profiles.
- Transaction counts are Poisson with mean revenue / average transaction value; per-transaction amounts are log-normal and rescaled so each day's transactions sum to that day's revenue.
- Fixed seed 42; fully reproducible. Assumptions are simulation choices, not measured Kenyan MSME statistics.

## 3. Features and targets

Historical features are computed strictly from days before `as_of_date` (revenue/tx-count rolling windows 7–90 days, active days, daily-revenue mean/median/std/cv, zero-revenue-day rate, weekend revenue share, average transaction value, transaction frequency, trend, growth/decline). Targets are computed strictly from the 30 days after `as_of_date`: next-30-day revenue, minimum future daily revenue, 10th percentile future daily revenue, and future daily-revenue volatility.

## 4. Validation

Chronological train/validation/holdout split (never shuffled), with a 35-day embargo gap between every split to reduce overlap between 30-day targets.

## 5. Forecasting results

Model selection used the validation split. Selected model: **random_forest / B_revenue_tx** (holdout MAE KSh 13,408). The best configuration found on holdout was random_forest/D_revenue_behavioural_seasonal (MAE KSh 12,298), compared with the best baseline at KSh 12,918.

Holdout MAE (KSh) by model and feature set:

```
model                           hist_gradient_boosting   linear  naive_30d  naive_60d  random_forest  sma_90d  weighted_avg
feature_set                                                                                                                
A_revenue_only                                 14260.0  12964.0        NaN        NaN        13691.0      NaN           NaN
B_revenue_tx                                   12723.0  12955.0        NaN        NaN        13408.0      NaN           NaN
C_revenue_behavioural                          13082.0  12691.0        NaN        NaN        12434.0      NaN           NaN
D_revenue_behavioural_seasonal                 12499.0  12721.0        NaN        NaN        12298.0      NaN           NaN
baseline                                           NaN      NaN    14799.0    13554.0            NaN  12918.0       13853.0
```

Top permutation-importance features (feature set D):

```
                      feature  permutation_importance_mean
     median_daily_revenue_30d                     0.109434
                  revenue_90d                     0.102241
    transaction_frequency_30d                     0.053222
                  revenue_30d                     0.033540
                  revenue_60d                     0.022249
        avg_daily_revenue_30d                     0.020827
                  revenue_14d                     0.003997
average_transaction_value_30d                     0.002571
                   revenue_7d                     0.001894
       revenue_trend_30_vs_90                     0.000912
```

## 6. Limit-setting strategies

- **A_simple_revenue**: 5 days of predicted daily revenue (soft cap 15,000)
- **B_conservative**: 5 days of median (typical) daily revenue
- **C_revenue_stability**: 5 days scaled by stability factor (volatile -> fewer days)
- **D_cashflow_coverage**: 5 days * repayment rate (50%)

## 7. Stress testing

Mean sustainable rate by archetype and scenario (fraction of limits the business could repay under the simulation rule):

```
scenario             growth  high_volatility  normal  revenue_shock  seasonal_downturn
archetype                                                                             
A_stable_barbershop     1.0              1.0     1.0           1.00               1.00
B_volatile_retail       1.0              1.0     1.0           1.00               1.00
C_growing               1.0              1.0     1.0           1.00               1.00
D_declining             1.0              1.0     1.0           1.00               1.00
E_weekly_seasonal       1.0              1.0     1.0           1.00               1.00
F_monthly_seasonal      1.0              1.0     1.0           1.00               1.00
G_intermittent          1.0              1.0     1.0           0.95               0.99
H_high_volume           1.0              1.0     1.0           1.00               1.00
I_micro_vendor          1.0              1.0     1.0           0.99               1.00
J_micro_service         1.0              1.0     1.0           0.98               1.00
```

## 8. Business-level summary (sample)

```
business_id           archetype  predicted_30d_revenue_kes  actual_30d_revenue_kes  revenue_volatility_cv  trend_growth_rate  transaction_frequency_30d  zero_revenue_day_rate  limit_A_simple_revenue  limit_B_conservative  limit_C_revenue_stability  limit_D_cashflow_coverage  overall_sustainable_rate
    SYN_000 A_stable_barbershop              140432.479837           140625.939231               0.231991          -0.500730                  11.766304               0.023077                 15000.0          15000.000000               14954.772226               11702.706653                       1.0
    SYN_001 A_stable_barbershop              125140.717224           124032.910769               0.178608          -0.497631                  10.524359               0.000000                 15000.0          15000.000000               15000.000000               10428.393102                       1.0
    SYN_002 A_stable_barbershop              121991.958511           117754.051538               0.181850          -0.498818                   9.929134               0.005128                 15000.0          15000.000000               14977.124353               10165.996543                       1.0
    SYN_003 A_stable_barbershop              127573.551488           128160.233846               0.194001          -0.498527                  11.119805               0.005128                 15000.0          15000.000000               15000.000000               10631.129291                       1.0
    SYN_004 A_stable_barbershop              102463.264222           100117.095385               0.221641          -0.499241                   8.454111               0.016667                 15000.0          15000.000000               13289.043119                8538.605352                       1.0
    SYN_005 A_stable_barbershop               94020.992933            95071.408462               0.156207          -0.499605                   7.691026               0.000000                 15000.0          15000.000000               13222.121159                7835.082744                       1.0
    SYN_006 A_stable_barbershop              136236.385691           134550.989615               0.231397          -0.501058                  11.549681               0.023077                 15000.0          15000.000000               14978.113318               11353.032141                       1.0
    SYN_007 A_stable_barbershop              123496.227139           122180.948077               0.229814          -0.498488                  10.154560               0.012821                 15000.0          15000.000000               14750.747163               10291.352262                       1.0
    SYN_008   B_volatile_retail              188775.853437           191529.543077               0.878624          -0.514632                   7.966023               0.085897                 15000.0          15000.000000               11202.638259               14555.281489                       1.0
    SYN_009   B_volatile_retail              171339.341583           171029.799615               0.888572          -0.498697                   6.268015               0.041026                 15000.0          15000.000000               10212.362825               14258.358995                       1.0
    SYN_010   B_volatile_retail              171031.253750           163468.786154               0.952564          -0.502963                   6.231954               0.035897                 15000.0          14915.223077               10000.772814               14231.318541                       1.0
    SYN_011   B_volatile_retail              160712.084020           157846.975769               1.124605          -0.503840                   5.972922               0.037179                 15000.0          14488.425962                9653.060730               13392.034614                       1.0
    SYN_012   B_volatile_retail              173146.583532           173090.461538               0.733247          -0.500327                   6.603679               0.065385                 15000.0          15000.000000               10487.314251               14428.376286                       1.0
    SYN_013   B_volatile_retail              154695.964975           148835.868462               0.892622          -0.506839                   5.670847               0.055128                 15000.0          14784.574038                9407.086530               12891.330415                       1.0
    SYN_014   B_volatile_retail              198715.163298           227829.573077               1.002099          -0.481121                   8.820778               0.026923                 15000.0          15000.000000               11691.543493               14830.765899                       1.0
    SYN_015   B_volatile_retail              173605.392768           182125.464615               0.941498          -0.500066                   7.287014               0.070513                 15000.0          15000.000000               10126.981245               14392.816771                       1.0
```

## 9. Limitations

- Synthetic data tests the *architecture and behaviour* of the framework, not real-world predictive accuracy.
- The repayment rule (a fixed fraction of realised revenue) is a controlled simulation assumption, not calibrated borrower behaviour.
- No conclusion about real Kenyan MSMEs should be drawn from these numbers.
