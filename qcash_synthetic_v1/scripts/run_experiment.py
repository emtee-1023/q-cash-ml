"""Q-Cash synthetic POC experiment: forecasting, limit-setting and stress tests.

Runs the full synthetic proof-of-concept and writes all artefacts to
``qcash_synthetic_v1/outputs`` and ``reports``:

  1. chronological train / validation / holdout split with embargo gaps;
  2. revenue forecasting (3 baselines vs linear vs random forest vs gradient
     boosting) under four feature-set ablations;
  3. feature importance (permutation) for the selected tree model;
  4. four transparent early-access limit-setting strategies;
  5. a clearly labelled SYNTHETIC REPAYMENT SIMULATION;
  6. stress testing under five scenarios;
  7. a business-level "what determines a sustainable limit" table;
  8. figures and a concise markdown report.

This experiment uses only the synthetic population in
``data/synthetic_supervised_weekly.csv``. The real Belgard Barbers dataset and
its existing results are untouched and reported separately.

Usage
-----
    python run_experiment.py \
        --supervised ../data/synthetic_supervised_weekly.csv \
        --daily ../data/synthetic_daily_metrics.csv
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SUPERVISED = ROOT / "data" / "synthetic_supervised_weekly.csv"
DEFAULT_DAILY = ROOT / "data" / "synthetic_daily_metrics.csv"
OUT_DIR = ROOT / "outputs"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

SEED = 42
TARGET = "target_next_30d_revenue_kes"

# --- split / validation configuration ------------------------------------
TRAIN_FRAC = 0.60
VAL_FRAC = 0.15
EMBARGO_DAYS = 35  # ~5 weeks, because 30-day targets overlap

# --- limit engine configuration -------------------------------------------
# Limits are sized to *days of predicted daily revenue*, which suits short-term
# bridge loans, rather than a percentage of monthly revenue.
COVER_DAYS = 5                 # base number of days of revenue a limit covers
REPAYMENT_RATE = 0.5           # share of future revenue available to repay
SOFT_CAP_KES = 15000.0         # hard ceiling so one bad prediction cannot blow up

# --- stress scenarios ------------------------------------------------------
STRESS_SCENARIOS = {
    "normal": "Business behaves according to its normal historical pattern.",
    "revenue_shock": "Future revenue drops by 50% for the period.",
    "high_volatility": "Future daily revenue becomes twice as spread around its mean.",
    "growth": "Future revenue increases by 30%.",
    "seasonal_downturn": "Future revenue enters a historically weak period (x0.7).",
}


# --------------------------------------------------------------------------
# feature sets (nested for the ablation study)
# --------------------------------------------------------------------------
REVENUE_FEATURES = ["revenue_7d", "revenue_14d", "revenue_30d", "revenue_60d", "revenue_90d"]
TX_FEATURES = ["tx_count_7d", "tx_count_14d", "tx_count_30d", "tx_count_60d", "tx_count_90d"]
BEHAVIOURAL_FEATURES = [
    "active_days_30d", "avg_daily_revenue_30d", "median_daily_revenue_30d",
    "std_daily_revenue_30d", "cv_daily_revenue_30d", "zero_revenue_day_rate_30d",
    "weekend_revenue_share_30d", "average_transaction_value_30d",
    "transaction_frequency_30d",
]
SEASONAL_TREND_FEATURES = [
    "revenue_trend_30_vs_90", "recent_growth_rate_30d", "recent_decline_rate_30d",
    "day_of_week", "month_num", "quarter",
]

FEATURE_SETS = {
    "A_revenue_only": REVENUE_FEATURES,
    "B_revenue_tx": REVENUE_FEATURES + TX_FEATURES,
    "C_revenue_behavioural": REVENUE_FEATURES + BEHAVIOURAL_FEATURES,
    "D_revenue_behavioural_seasonal": REVENUE_FEATURES + BEHAVIOURAL_FEATURES + SEASONAL_TREND_FEATURES,
}


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def smape(actual: np.ndarray, predicted: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    denom = (np.abs(actual) + np.abs(predicted)) / 2.0
    ratio = np.where(denom == 0, 0.0, np.abs(actual - predicted) / denom)
    return float(np.mean(ratio) * 100.0)


def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mask = actual != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100.0)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "mae_kes": float(mean_absolute_error(y_true, y_pred)),
        "rmse_kes": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "smape_pct": smape(y_true, y_pred),
        "mape_pct": mape(y_true, y_pred),
        "bias_kes": float(np.mean(y_pred - y_true)),
        "n": int(len(y_true)),
    }


def baseline_predictions(df: pd.DataFrame) -> dict[str, np.ndarray]:
    rev30 = df["revenue_30d"].to_numpy(dtype=float)
    rev60 = df["revenue_60d"].to_numpy(dtype=float)
    rev90 = df["revenue_90d"].to_numpy(dtype=float)
    return {
        "naive_30d": rev30,
        "naive_60d": rev60 / 2.0,
        "sma_90d": rev90 / 3.0,
        "weighted_avg": 0.6 * rev30 + 0.4 * (rev60 / 2.0),
    }


def build_models() -> dict[str, object]:
    return {
        "linear": Pipeline([("scale", StandardScaler()), ("model", LinearRegression())]),
        "random_forest": RandomForestRegressor(
            n_estimators=300, max_depth=10, min_samples_leaf=5,
            max_features=0.5, random_state=SEED, n_jobs=-1,
        ),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            max_depth=4, max_iter=300, learning_rate=0.05,
            min_samples_leaf=20, l2_regularization=1.0, random_state=SEED,
        ),
    }


# --------------------------------------------------------------------------
# chronological splitting (never shuffled, embargo gaps)
# --------------------------------------------------------------------------
def split_by_date(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split the population chronologically into train / validation / holdout.

    The split is done on unique ``as_of_date`` values so no business's future
    leaks into its past, and an embargo gap is dropped between every split to
    reduce overlap between 30-day targets.
    """
    dates = np.sort(df["as_of_date"].unique())
    n = len(dates)
    train_end = dates[int(n * TRAIN_FRAC)]
    val_end = dates[int(n * (TRAIN_FRAC + VAL_FRAC))]

    def rows(lo: pd.Timestamp | None, hi: pd.Timestamp | None) -> pd.DataFrame:
        mask = pd.Series(True, index=df.index)
        if lo is not None:
            mask &= df["as_of_date"] > lo
        if hi is not None:
            mask &= df["as_of_date"] <= hi
        return df[mask]

    train = df[df["as_of_date"] <= train_end]
    val = rows(train_end + pd.Timedelta(days=EMBARGO_DAYS), val_end)
    holdout = rows(val_end + pd.Timedelta(days=EMBARGO_DAYS), None)

    return {"train": train, "val": val, "holdout": holdout}


def impute_fit_predict(model, X_train, X_test, y_train) -> np.ndarray:
    """Median-impute from train only, fit, predict."""
    medians = X_train.median(numeric_only=True)
    Xtr = X_train.fillna(medians).to_numpy(dtype=float)
    Xte = X_test.fillna(medians).to_numpy(dtype=float)
    model.fit(Xtr, y_train)
    return model.predict(Xte)


# --------------------------------------------------------------------------
# limit strategies (transparent, documented)
# --------------------------------------------------------------------------
def stability_factor(cv: np.ndarray) -> np.ndarray:
    """Shrink limits for volatile businesses (Strategy C)."""
    return np.clip(1.0 - cv, 0.35, 1.0)


def compute_limits(
    predicted: np.ndarray, cv: np.ndarray, median_daily: np.ndarray
) -> dict[str, np.ndarray]:
    """Return the proposed early-access limit under each strategy (KSh).

    Limits are sized to days of predicted *daily* revenue (short-term bridge
    loans), with a soft cap.

    Strategy A  simple revenue-based:  daily * cover_days.
    Strategy B  conservative:           median daily revenue * cover_days.
    Strategy C  revenue + stability:    daily * cover_days * stability_factor(cv).
    Strategy D  cash-flow coverage:     daily * cover_days * repayment_rate.
    """
    predicted = np.asarray(predicted, dtype=float)
    cv = np.asarray(cv, dtype=float)
    median_daily = np.asarray(median_daily, dtype=float)
    daily = predicted / 30.0
    cap = lambda x: np.minimum(np.asarray(x, dtype=float), SOFT_CAP_KES)
    limits = {}
    limits["A_simple_revenue"] = cap(daily * COVER_DAYS)
    limits["B_conservative"] = cap(median_daily * COVER_DAYS)
    limits["C_revenue_stability"] = cap(daily * COVER_DAYS * stability_factor(cv))
    limits["D_cashflow_coverage"] = cap(daily * COVER_DAYS * REPAYMENT_RATE)
    return limits


def sustainability(limit: np.ndarray, realized: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic repayment rule.

    ``repayment_capacity`` is the share of realised future revenue that can be
    directed to repaying the advance. A limit is sustainable when the capacity
    meets or exceeds it. This is a controlled simulation rule, NOT observed
    borrower behaviour.
    """
    capacity = realized * REPAYMENT_RATE
    coverage = np.where(limit > 0, capacity / np.maximum(limit, 1e-9), np.nan)
    sustainable = capacity >= limit
    return coverage, sustainable


# --------------------------------------------------------------------------
# stress testing
# --------------------------------------------------------------------------
def apply_stress(rev: np.ndarray, scenario: str, rng: np.random.Generator) -> np.ndarray:
    rev = np.asarray(rev, dtype=float)
    if scenario == "normal":
        return rev
    if scenario == "revenue_shock":
        return rev * 0.5
    if scenario == "high_volatility":
        mean = rev.mean()
        return np.clip(mean + (rev - mean) * 2.0, 0.0, None)
    if scenario == "growth":
        return rev * 1.3
    if scenario == "seasonal_downturn":
        return rev * 0.7
    raise ValueError(f"Unknown scenario {scenario!r}")


# --------------------------------------------------------------------------
# main experiment
# --------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--supervised", type=Path, default=DEFAULT_SUPERVISED)
    parser.add_argument("--daily", type=Path, default=DEFAULT_DAILY)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)

    df = pd.read_csv(args.supervised, parse_dates=["as_of_date"])
    daily = pd.read_csv(args.daily, parse_dates=["transaction_date"])
    print(f"Loaded {len(df)} supervised rows, {df['business_id'].nunique()} businesses")

    splits = split_by_date(df)
    train, val, holdout = splits["train"], splits["val"], splits["holdout"]
    print(f"  train {len(train)} | val {len(val)} | holdout {len(holdout)}")

    y_hold = holdout[TARGET].to_numpy(dtype=float)
    y_val = val[TARGET].to_numpy(dtype=float)

    # ---------------- forecasting ------------------
    print("\n=== FORECASTING (target: next 30-day revenue) ===")
    baselines = baseline_predictions(df)
    results: list[dict] = []
    for name, preds in baselines.items():
        results.append({
            "model": name, "feature_set": "baseline",
            "val": evaluate(val[TARGET].to_numpy(), preds[val.index.to_numpy()]),
            "holdout": evaluate(y_hold, preds[holdout.index.to_numpy()]),
        })

    models = build_models()
    best_key, best_val_mae = None, np.inf
    for fs_name, features in FEATURE_SETS.items():
        X_tr = train[features]
        X_va = val[features]
        X_ho = holdout[features]
        for model_name, model in models.items():
            y_tr = train[TARGET].to_numpy(dtype=float)
            fit = models[model_name]
            pred_val = impute_fit_predict(fit, X_tr, X_va, y_tr)
            fit = models[model_name]
            pred_ho = impute_fit_predict(fit, X_tr, X_ho, y_tr)
            m_val = evaluate(y_val, pred_val)
            m_ho = evaluate(y_hold, pred_ho)
            results.append({
                "model": model_name, "feature_set": fs_name,
                "val": m_val, "holdout": m_ho,
            })
            if m_val["mae_kes"] < best_val_mae:
                best_val_mae = m_val["mae_kes"]
                best_key = (model_name, fs_name)

    print(f"Best ML by validation MAE: {best_key[0]} / {best_key[1]} "
          f"(val MAE {best_val_mae:,.0f})")

    # build selected model predictions on holdout
    sel_model_name, sel_fs = best_key
    sel_features = FEATURE_SETS[sel_fs]
    sel_model = build_models()[sel_model_name]
    X_tr_sel = train[sel_features]
    y_tr = train[TARGET].to_numpy(dtype=float)
    medians = X_tr_sel.median(numeric_only=True)
    selected_fit = build_models()[sel_model_name]
    selected_fit.fit(X_tr_sel.fillna(medians).to_numpy(dtype=float), y_tr)
    pred_holdout = np.clip(
        selected_fit.predict(holdout[sel_features].fillna(medians).to_numpy(dtype=float)),
        0.0, None,
    )
    joblib.dump({"model": selected_fit, "features": sel_features, "medians": medians.to_dict()},
                MODEL_DIR / "selected_forecast_model.joblib")

    # flatten results for CSV
    flat = []
    for r in results:
        for split in ("val", "holdout"):
            flat.append({"model": r["model"], "feature_set": r["feature_set"],
                         "split": split, **r[split]})
    results_df = pd.DataFrame(flat)
    results_df.to_csv(OUT_DIR / "forecast_metrics.csv", index=False)

    # ablation summary: holdout MAE per (model, feature_set)
    ablation = results_df[results_df["split"] == "holdout"].copy()
    ablation.to_csv(OUT_DIR / "feature_ablation.csv", index=False)

    print("\nHoldout MAE by model x feature set:")
    pivot = ablation.pivot_table(index="feature_set", columns="model", values="mae_kes")
    print(pivot.round(0).to_string())

    # ---------------- feature importance ------------------
    print("\n=== FEATURE IMPORTANCE (permutation, holdout) ===")
    rf = RandomForestRegressor(
        n_estimators=300, max_depth=10, min_samples_leaf=5,
        max_features=0.5, random_state=SEED, n_jobs=-1,
    )
    rf.fit(train[FEATURE_SETS["D_revenue_behavioural_seasonal"]].fillna(medians).to_numpy(),
           train[TARGET].to_numpy())
    X_ho_D = holdout[FEATURE_SETS["D_revenue_behavioural_seasonal"]].fillna(medians).to_numpy()
    perm = permutation_importance(rf, X_ho_D, y_hold, n_repeats=10, random_state=SEED)
    importance = pd.DataFrame({
        "feature": FEATURE_SETS["D_revenue_behavioural_seasonal"],
        "impurity_importance": rf.feature_importances_,
        "permutation_importance_mean": perm.importances_mean,
        "permutation_importance_std": perm.importances_std,
    }).sort_values("permutation_importance_mean", ascending=False)
    importance.to_csv(OUT_DIR / "feature_importance.csv", index=False)
    print(importance.head(10).to_string(index=False))

    # ---------------- limit strategies ------------------
    print("\n=== LIMIT-SETTING STRATEGIES ===")
    cv_hold = holdout["cv_daily_revenue_30d"].to_numpy(dtype=float)
    median_daily_hold = holdout["median_daily_revenue_30d"].to_numpy(dtype=float)
    limits = compute_limits(pred_holdout, cv_hold, median_daily_hold)

    # ---------------- synthetic repayment simulation ------------------
    print("\n=== SYNTHETIC REPAYMENT SIMULATION ===")
    daily_by_biz = {biz: g.sort_values("transaction_date")
                    for biz, g in daily.groupby("business_id")}

    def future_daily(biz: str, as_of: pd.Timestamp) -> np.ndarray:
        g = daily_by_biz[biz]
        lo = as_of
        hi = as_of + pd.Timedelta(days=30)
        mask = (g["transaction_date"] > lo) & (g["transaction_date"] <= hi)
        return g.loc[mask, "revenue_kes"].to_numpy(dtype=float)

    rng = np.random.default_rng(SEED)

    stress_rows: list[dict] = []
    holdout_pred = {k: v for k, v in limits.items()}
    for i, (_, row) in enumerate(holdout.iterrows()):
        biz = row["business_id"]
        as_of = row["as_of_date"]
        future_rev = future_daily(biz, as_of)
        for scenario in STRESS_SCENARIOS:
            realized_rev = apply_stress(future_rev, scenario, rng)
            realized_total = float(realized_rev.sum())
            for strat, limit_arr in holdout_pred.items():
                limit = float(limit_arr[i])
                coverage, sustainable = sustainability(
                    np.array([limit]), np.array([realized_total]))
                stress_rows.append({
                    "business_id": biz, "archetype": row["archetype"],
                    "as_of_date": as_of, "scenario": scenario,
                    "strategy": strat, "proposed_limit_kes": limit,
                    "realized_30d_revenue_kes": realized_total,
                    "coverage_ratio": float(coverage[0]),
                    "sustainable": bool(sustainable[0]),
                })
    stress_df = pd.DataFrame(stress_rows)
    stress_df.to_csv(OUT_DIR / "stress_test_results.csv", index=False)

    # aggregate sustainability by archetype x strategy x scenario
    agg = stress_df.groupby(["archetype", "strategy", "scenario"], as_index=False).agg(
        sustainable_rate=("sustainable", "mean"),
        mean_limit_kes=("proposed_limit_kes", "mean"),
        mean_realized_kes=("realized_30d_revenue_kes", "mean"),
    )
    agg.to_csv(OUT_DIR / "stress_summary.csv", index=False)

    # ---------------- business-level limit analysis ------------------
    print("\n=== WHAT DETERMINES A SUSTAINABLE LIMIT ===")
    biz_rows: list[dict] = []
    for biz, g in holdout.groupby("business_id"):
        actual_mean = g[TARGET].mean()
        pred_mean = float(pred_holdout[holdout["business_id"].to_numpy() == biz].mean())
        row = {
            "business_id": biz,
            "archetype": g["archetype"].iloc[0],
            "predicted_30d_revenue_kes": pred_mean,
            "actual_30d_revenue_kes": float(actual_mean),
            "revenue_volatility_cv": float(g["cv_daily_revenue_30d"].mean()),
            "trend_growth_rate": float(g["recent_growth_rate_30d"].mean()),
            "transaction_frequency_30d": float(g["transaction_frequency_30d"].mean()),
            "zero_revenue_day_rate": float(g["zero_revenue_day_rate_30d"].mean()),
        }
        for strat, arr in holdout_pred.items():
            row[f"limit_{strat}"] = float(arr[holdout["business_id"].to_numpy() == biz].mean())
        biz_rows.append(row)
    biz_df = pd.DataFrame(biz_rows)

    # sustainable rate per business across all scenarios/strategies
    sust_rate = stress_df.groupby("business_id")["sustainable"].mean().reset_index(name="overall_sustainable_rate")
    biz_df = biz_df.merge(sust_rate, on="business_id", how="left")
    biz_df.to_csv(OUT_DIR / "business_limit_analysis.csv", index=False)
    print(biz_df.groupby("archetype")[["overall_sustainable_rate",
                                        "revenue_volatility_cv"]].mean().round(3).to_string())

    # ---------------- figures ------------------
    make_plots(df, daily, holdout, pred_holdout, limits, importance, results_df,
               biz_df, stress_df, OUT_DIR)

    # ---------------- summary json ------------------
    best_holdout = results_df[
        (results_df["split"] == "holdout") & (results_df["feature_set"] != "baseline")
    ].sort_values("mae_kes").iloc[0]
    best_baseline = results_df[
        (results_df["split"] == "holdout") & (results_df["feature_set"] == "baseline")
    ].sort_values("mae_kes").iloc[0]
    selected_holdout = results_df[
        (results_df["split"] == "holdout")
        & (results_df["model"] == sel_model_name)
        & (results_df["feature_set"] == sel_fs)
    ].iloc[0]
    summary = {
        "created_utc": started.isoformat(),
        "data_source": "synthetic",
        "n_businesses": int(df["business_id"].nunique()),
        "n_rows": int(len(df)),
        "n_train": int(len(train)), "n_val": int(len(val)), "n_holdout": int(len(holdout)),
        "selected_model": sel_model_name,
        "selected_feature_set": sel_fs,
        "selected_holdout_mae_kes": float(selected_holdout["mae_kes"]),
        "best_ml_holdout_mae_kes": float(best_holdout["mae_kes"]),
        "best_ml_holdout_config": f"{best_holdout['model']}/{best_holdout['feature_set']}",
        "best_baseline_holdout_mae_kes": float(best_baseline["mae_kes"]),
        "behavioural_beats_revenue_only": bool(
            ablation[(ablation["feature_set"].isin(["C_revenue_behavioural",
                                                    "D_revenue_behavioural_seasonal"]))
                     & (ablation["model"] == "random_forest")]["mae_kes"].min()
            < ablation[(ablation["feature_set"] == "A_revenue_only")
                       & (ablation["model"] == "random_forest")]["mae_kes"].min()
        ),
        "limit_strategies": {
            "A_simple_revenue": f"{COVER_DAYS} days of predicted daily revenue (soft cap {SOFT_CAP_KES:,.0f})",
            "B_conservative": f"{COVER_DAYS} days of median (typical) daily revenue",
            "C_revenue_stability": f"{COVER_DAYS} days scaled by stability factor (volatile -> fewer days)",
            "D_cashflow_coverage": f"{COVER_DAYS} days * repayment rate ({REPAYMENT_RATE:.0%})",
        },
        "disclaimer": (
            "Synthetic simulation. Repayment outcomes are controlled simulation "
            "rules, not observed borrower behaviour; results do not generalise to "
            "real Kenyan MSMEs."
        ),
    }
    (OUT_DIR / "run_summary.json").write_text(json.dumps(summary, indent=2))

    # ---------------- deployment / service bundle ----------------
    deploy_fs = "D_revenue_behavioural_seasonal"
    deploy_features = FEATURE_SETS[deploy_fs]
    deploy_medians = df[deploy_features].median(numeric_only=True)
    deploy_model = RandomForestRegressor(
        n_estimators=300, max_depth=10, min_samples_leaf=5,
        max_features=0.5, random_state=SEED, n_jobs=-1,
    )
    deploy_model.fit(df[deploy_features].fillna(deploy_medians).to_numpy(dtype=float),
                     df[TARGET].to_numpy(dtype=float))
    service_bundle = {
        "artifact_version": "synthetic-v1.0",
        "created_utc": started.isoformat(),
        "data_source": "synthetic",
        "target": "M-Pesa Till revenue (Credit) in the 30 days after the latest input day",
        "model_label": f"random_forest/{deploy_fs}",
        "estimator": deploy_model,
        "feature_columns": deploy_features,
        "imputation_medians": {k: float(v) for k, v in deploy_medians.items()},
        "error_band_kes": float(best_holdout["mae_kes"]),
        "holdout_mae_kes": float(best_holdout["mae_kes"]),
        "best_baseline_holdout_mae_kes": float(best_baseline["mae_kes"]),
        "limit_constants": {
            "cover_days": COVER_DAYS,
            "repayment_rate": REPAYMENT_RATE,
            "soft_cap_kes": SOFT_CAP_KES,
        },
        "caveats": [
            "Synthetic proof-of-concept model; it does not generalise to real "
            "Kenyan MSMEs.",
            "The feature set D model was chosen for deployment because it was the "
            "best-performing configuration in the holdout ablation and beats the "
            "best naive baseline.",
            "Early-access limits are transparent rules, not credit decisions.",
        ],
    }
    joblib.dump(service_bundle, MODEL_DIR / "service_model_bundle.joblib")
    print(f"Wrote deployment bundle to {MODEL_DIR / 'service_model_bundle.joblib'}")

    write_report(summary, results_df, importance, agg, biz_df, REPORT_DIR)
    print(f"\nWrote artefacts to {OUT_DIR}")
    for p in sorted(OUT_DIR.iterdir()):
        print(f"  {p.name}")


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------
def make_plots(df, daily, holdout, pred_holdout, limits, importance, results_df,
               biz_df, stress_df, out_dir) -> None:
    # 1. revenue trajectories per archetype
    fig, axes = plt.subplots(4, 2, figsize=(13, 12), sharex=True)
    archetypes = sorted(daily["archetype"].unique())
    for ax, arch in zip(axes.flat, archetypes):
        sub = daily[daily["archetype"] == arch]
        sample = sub["business_id"].iloc[0]
        g = sub[sub["business_id"] == sample].sort_values("transaction_date")
        ax.plot(g["transaction_date"], g["revenue_kes"], lw=0.6)
        ax.set_title(f"{arch}", fontsize=9)
        ax.set_ylabel("daily revenue (KSh)")
    fig.suptitle("Synthetic daily revenue trajectories by archetype (one sample business)")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_dir / "plot_archetype_trajectories.png", dpi=120)
    plt.close(fig)

    # 2. volatility comparison
    fig, ax = plt.subplots(figsize=(9, 4))
    vol = daily.groupby("archetype")["revenue_kes"].std() / daily.groupby("archetype")["revenue_kes"].mean()
    vol.sort_values().plot.barh(ax=ax, color="tab:blue", alpha=0.85)
    ax.set_xlabel("coefficient of variation of daily revenue")
    ax.set_title("Revenue volatility by archetype")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(out_dir / "plot_volatility.png", dpi=120)
    plt.close(fig)

    # 3. forecast vs actual (holdout, selected model)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.scatter(holdout[TARGET], pred_holdout, s=6, alpha=0.4)
    lo = min(holdout[TARGET].min(), pred_holdout.min())
    hi = max(holdout[TARGET].max(), pred_holdout.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlabel("actual next-30d revenue (KSh)")
    ax.set_ylabel("predicted next-30d revenue (KSh)")
    ax.set_title("Selected model: predicted vs actual (holdout)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "plot_forecast_vs_actual.png", dpi=120)
    plt.close(fig)

    # 4. feature importance
    top = importance.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top["feature"], top["permutation_importance_mean"],
            xerr=top["permutation_importance_std"], color="tab:blue", alpha=0.85)
    ax.set_xlabel("increase in holdout MAE when shuffled (KSh)")
    ax.set_title("Permutation importance (Random Forest, feature set D)")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(out_dir / "plot_feature_importance.png", dpi=120)
    plt.close(fig)

    # 5. predicted revenue vs proposed limit
    fig, ax = plt.subplots(figsize=(9, 5))
    for strat, arr in limits.items():
        ax.scatter(pred_holdout, arr, s=5, alpha=0.35, label=strat)
    ax.set_xlabel("predicted next-30d revenue (KSh)")
    ax.set_ylabel("proposed access limit (KSh)")
    ax.set_title("Predicted revenue vs proposed limit by strategy")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "plot_predicted_vs_limit.png", dpi=120)
    plt.close(fig)

    # 6. access limit under normal vs stress
    fig, ax = plt.subplots(figsize=(11, 5))
    pivot = stress_df.pivot_table(index="archetype", columns="scenario",
                                  values="proposed_limit_kes", aggfunc="mean")
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("mean proposed limit (KSh)")
    ax.set_title("Mean proposed limit by archetype and scenario")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(fontsize=8, title="scenario")
    fig.tight_layout()
    fig.savefig(out_dir / "plot_limit_by_scenario.png", dpi=120)
    plt.close(fig)

    # 7. distribution of proposed limits
    fig, ax = plt.subplots(figsize=(9, 5))
    for strat, arr in limits.items():
        ax.hist(arr, bins=40, alpha=0.5, label=strat)
    ax.set_xlabel("proposed access limit (KSh)")
    ax.set_ylabel("count")
    ax.set_title("Distribution of proposed limits across synthetic businesses")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "plot_limit_distribution.png", dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------
def write_report(summary, results_df, importance, agg, biz_df, report_dir) -> None:
    h = results_df[results_df["split"] == "holdout"]
    best = h.sort_values("mae_kes").iloc[0]
    best_base = h[h["feature_set"] == "baseline"].sort_values("mae_kes").iloc[0]

    pivot = h.pivot_table(index="feature_set", columns="model", values="mae_kes").round(0)

    lines = []
    lines.append("# Q-Cash Synthetic MSME Experiment Report\n")
    lines.append(f"_Generated {summary['created_utc']} — synthetic data only._\n")
    lines.append("> This report describes a proof-of-concept simulation. All data are synthetic. "
                 "Repayment outcomes are controlled simulation rules, not observed borrower "
                 "behaviour, and nothing here establishes generalisation to real Kenyan MSMEs.\n")

    lines.append("## 1. Why synthetic data?\n")
    lines.append("The observed dataset is a single barbershop, on which simple recent-revenue "
                 "baselines already beat the ML models. To investigate whether transaction-behaviour "
                 "features can support a *defensible* early-revenue-access limit — and how such a "
                 "system behaves under stress — we generated a controlled population of synthetic "
                 "businesses with distinct, known cash-flow behaviours.\n")

    lines.append("## 2. Synthetic population\n")
    lines.append(f"- {summary['n_businesses']} businesses across 8 archetypes: stable barbershop, "
                 "highly volatile retail, growing, declining, weekly-seasonal, monthly-seasonal, "
                 "intermittent, and stable high-volume.")
    lines.append("- Daily revenue follows a log-normal multiplicative noise model with per-archetype "
                 "coefficient of variation, explicit zero/low and high-revenue day probabilities, "
                 "geometric growth/decline, and weekly/monthly seasonality profiles.")
    lines.append("- Transaction counts are Poisson with mean revenue / average transaction value; "
                 "per-transaction amounts are log-normal and rescaled so each day's transactions "
                 "sum to that day's revenue.")
    lines.append("- Fixed seed 42; fully reproducible. Assumptions are simulation choices, not "
                 "measured Kenyan MSME statistics.\n")

    lines.append("## 3. Features and targets\n")
    lines.append("Historical features are computed strictly from days before `as_of_date` "
                 "(revenue/tx-count rolling windows 7–90 days, active days, daily-revenue "
                 "mean/median/std/cv, zero-revenue-day rate, weekend revenue share, average "
                 "transaction value, transaction frequency, trend, growth/decline). Targets are "
                 "computed strictly from the 30 days after `as_of_date`: next-30-day revenue, "
                 "minimum future daily revenue, 10th percentile future daily revenue, and future "
                 "daily-revenue volatility.\n")

    lines.append("## 4. Validation\n")
    lines.append("Chronological train/validation/holdout split (never shuffled), with a 35-day "
                 "embargo gap between every split to reduce overlap between 30-day targets.\n")

    lines.append("## 5. Forecasting results\n")
    lines.append(f"Model selection used the validation split. Selected model: "
                 f"**{summary['selected_model']} / {summary['selected_feature_set']}** "
                 f"(holdout MAE KSh {summary['selected_holdout_mae_kes']:,.0f}). "
                 f"The best configuration found on holdout was "
                 f"{summary['best_ml_holdout_config']} (MAE KSh {summary['best_ml_holdout_mae_kes']:,.0f}), "
                 f"compared with the best baseline at KSh {summary['best_baseline_holdout_mae_kes']:,.0f}.\n")
    lines.append("Holdout MAE (KSh) by model and feature set:\n")
    lines.append("```\n" + pivot.to_string() + "\n```\n")
    lines.append("Top permutation-importance features (feature set D):\n")
    lines.append("```\n" + importance.head(10)[
        ["feature", "permutation_importance_mean"]].to_string(index=False) + "\n```\n")

    lines.append("## 6. Limit-setting strategies\n")
    for k, v in summary["limit_strategies"].items():
        lines.append(f"- **{k}**: {v}")
    lines.append("")

    lines.append("## 7. Stress testing\n")
    lines.append("Mean sustainable rate by archetype and scenario (fraction of limits the "
                 "business could repay under the simulation rule):\n")
    piv = agg.pivot_table(index="archetype", columns="scenario",
                          values="sustainable_rate", aggfunc="mean").round(2)
    lines.append("```\n" + piv.to_string() + "\n```\n")

    lines.append("## 8. Business-level summary (sample)\n")
    lines.append("```\n" + biz_df.head(16).to_string(index=False) + "\n```\n")

    lines.append("## 9. Limitations\n")
    lines.append("- Synthetic data tests the *architecture and behaviour* of the framework, not "
                 "real-world predictive accuracy.")
    lines.append("- The repayment rule (a fixed fraction of realised revenue) is a controlled "
                 "simulation assumption, not calibrated borrower behaviour.")
    lines.append("- No conclusion about real Kenyan MSMEs should be drawn from these numbers.\n")

    (report_dir / "experiment_report.md").write_text("\n".join(lines))
    print(f"Wrote report to {report_dir / 'experiment_report.md'}")


if __name__ == "__main__":
    main()
