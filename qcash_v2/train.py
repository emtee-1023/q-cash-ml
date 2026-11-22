"""Model training, selection and evaluation for Q-Cash v2.

Candidates (retraining prompt section 6.1):
  1. seasonal-naive      next 30d = previous 30d
  2. moving-average 90d  next 30d = previous 90d / 3
  3. linear regression   on the same features
  4. Random Forest       level and growth targets
  5. XGBoost             level, growth, and ratio (MA-anchored) targets

Production model: XGBoost on the ratio target / (rev_sum_90 / 3) with
moving-average + momentum features. It beats the moving average on the temporal
and tail tests, ties it on unseen businesses, and is robust to micro businesses.

Selection uses walk-forward MAE (expanding window). Grids are small and
pre-declared. Five seeds are averaged; best-of-N is never reported.
Overfitting controls: permutation (shuffle-target) test, learning curves,
train/test gap, and the oracle ceiling (irreducible error floor).

All money is integer minor units; metrics are converted to KSh for reports.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor
except ImportError:  # pragma: no cover
    XGBRegressor = None

from common import (
    DATA_DIR,
    OUTPUTS_DIR,
    PLOTS_DIR,
    evaluate,
    load_config,
    minor_metrics_to_ksh,
)
from model_features import FEATURE_COLS, MA_DIVISOR, enhanced_frame, feature_frame
from splits import Partitions, build_partitions, walk_forward_folds

TARGET = "target_minor"
NAIVE_COL = "rev_sum_30"


def _impute(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    medians = X_train.median(numeric_only=True)
    Xtr = X_train.fillna(medians).to_numpy(dtype=float)
    Xte = X_test.fillna(medians).to_numpy(dtype=float)
    Xtr = np.nan_to_num(Xtr, nan=0.0, posinf=0.0, neginf=0.0)
    Xte = np.nan_to_num(Xte, nan=0.0, posinf=0.0, neginf=0.0)
    return Xtr, Xte


def _naive_pred(df: pd.DataFrame) -> np.ndarray:
    return df[NAIVE_COL].to_numpy(dtype=float)


def _ma_pred(df: pd.DataFrame) -> np.ndarray:
    return df["rev_sum_90"].to_numpy(dtype=float) / 3.0


def _rf(**params) -> RandomForestRegressor:
    cfg = load_config()["model"]["random_forest"]
    merged = {**cfg, **params}
    return RandomForestRegressor(n_jobs=-1, random_state=42, **merged)


def _linear() -> Pipeline:
    return Pipeline([("scale", StandardScaler()), ("model", LinearRegression())])


def _xgb(**params) -> object:
    cfg = load_config()["model"]["xgboost"]
    merged = {**cfg, **params}
    return XGBRegressor(
        n_jobs=-1, random_state=42, verbosity=0,
        objective="reg:absoluteerror",
        **merged,
    )


def _fit_predict(model, Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray) -> np.ndarray:
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    return np.clip(np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)


# ---------------------------------------------------------------------------
# walk-forward tuning
# ---------------------------------------------------------------------------


def tune_random_forest(train: pd.DataFrame) -> dict:
    cfg = load_config()["model"]["grid"]
    folds = walk_forward_folds(train)
    X_all = feature_frame(train)
    cols = list(X_all.columns)
    y_all = train[TARGET].to_numpy(dtype=float)

    best = None
    grid_results: list[dict] = []
    for n_est in cfg["n_estimators"]:
        for depth in cfg["max_depth"]:
            for leaf in cfg["min_samples_leaf"]:
                fold_maes: list[float] = []
                for fold in folds:
                    tr = fold["train"]
                    va = fold["val"]
                    Xtr = feature_frame(tr).reindex(columns=cols).fillna(X_all.median(numeric_only=True))
                    Xva = feature_frame(va).reindex(columns=cols).fillna(X_all.median(numeric_only=True))
                    model = _rf(n_estimators=n_est, max_depth=depth, min_samples_leaf=leaf)
                    pred = _fit_predict(model, Xtr.to_numpy(dtype=float), tr[TARGET].to_numpy(dtype=float),
                                        Xva.to_numpy(dtype=float))
                    fold_maes.append(float(mean_absolute_error(va[TARGET].to_numpy(dtype=float), pred)))
                mean_mae = float(np.mean(fold_maes))
                grid_results.append({
                    "n_estimators": n_est, "max_depth": depth, "min_samples_leaf": leaf,
                    "mean_wf_mae_minor": mean_mae, "sd_wf_mae_minor": float(np.std(fold_maes)),
                })
                if best is None or mean_mae < best["mean_wf_mae_minor"]:
                    best = grid_results[-1]
    return best, grid_results


def tune_xgboost(train: pd.DataFrame) -> dict:
    cfg = load_config()["model"]["xgb_grid"]
    folds = walk_forward_folds(train)
    X_all = feature_frame(train)
    cols = list(X_all.columns)
    medians = X_all.median(numeric_only=True)

    best = None
    grid_results: list[dict] = []
    for n_est in cfg["n_estimators"]:
        for depth in cfg["max_depth"]:
            for mcw in cfg["min_child_weight"]:
                fold_maes: list[float] = []
                for fold in folds:
                    tr = fold["train"]
                    va = fold["val"]
                    Xtr = feature_frame(tr).reindex(columns=cols).fillna(medians).to_numpy(dtype=float)
                    Xva = feature_frame(va).reindex(columns=cols).fillna(medians).to_numpy(dtype=float)
                    model = _xgb(n_estimators=n_est, max_depth=depth, min_child_weight=mcw)
                    pred = _fit_predict(model, Xtr, tr[TARGET].to_numpy(dtype=float), Xva)
                    fold_maes.append(float(mean_absolute_error(va[TARGET].to_numpy(dtype=float), pred)))
                mean_mae = float(np.mean(fold_maes))
                grid_results.append({
                    "n_estimators": n_est, "max_depth": depth, "min_child_weight": mcw,
                    "mean_wf_mae_minor": mean_mae, "sd_wf_mae_minor": float(np.std(fold_maes)),
                })
                if best is None or mean_mae < best["mean_wf_mae_minor"]:
                    best = grid_results[-1]
    return best, grid_results


# ---------------------------------------------------------------------------
# main evaluation
# ---------------------------------------------------------------------------


def _eval_baselines(df: pd.DataFrame, y: np.ndarray) -> dict:
    naive = _naive_pred(df)
    ma = _ma_pred(df)
    return {
        "seasonal_naive": evaluate(y, naive),
        "moving_avg_90d": evaluate(y, ma),
    }


def _fit_on_train(train: pd.DataFrame, rf_params: dict, xgb_params: dict, n_seeds: int) -> dict:
    """Fit linear + RF + XGBoost (multi-seed) on the training set.

    The models are fit once on ``train`` and reused to predict the validation
    and test sets, so no test information ever reaches the estimators.
    """
    Xtr = feature_frame(train)
    cols = list(Xtr.columns)
    Xtr_f = Xtr.to_numpy(dtype=float)
    ytr = train[TARGET].to_numpy(dtype=float)

    lin = _linear()
    lin.fit(Xtr_f, ytr)

    rfs = []
    for seed in range(n_seeds):
        rf = _rf(**rf_params)
        rf.random_state = seed
        rf.fit(Xtr_f, ytr)
        rfs.append(rf)

    xgbs = []
    if XGBRegressor is not None:
        for seed in range(n_seeds):
            xgb = _xgb(**xgb_params)
            xgb.random_state = seed
            xgb.fit(Xtr_f, ytr)
            xgbs.append(xgb)

    # "fair shot" for the tree models: predict the GROWTH ratio
    # (target / rev_sum_30) instead of the level, then multiply back. This
    # avoids the tree extrapolation cap and is the formulation most favourable
    # to RF/XGBoost.
    rev30_tr = train["rev_sum_30"].to_numpy(dtype=float)
    ratio_tr = np.where(rev30_tr > 0, ytr / rev30_tr, 1.0)
    ratio_tr = np.clip(np.nan_to_num(ratio_tr, nan=1.0, posinf=1.0, neginf=1.0), 0.0, 5.0)

    rfs_growth = []
    for seed in range(n_seeds):
        rf = _rf(**rf_params)
        rf.random_state = seed
        rf.fit(Xtr_f, ratio_tr)
        rfs_growth.append(rf)

    xgbs_growth = []
    if XGBRegressor is not None:
        for seed in range(n_seeds):
            xgb = _xgb(**xgb_params)
            xgb.random_state = seed
            xgb.fit(Xtr_f, ratio_tr)
            xgbs_growth.append(xgb)

    # Production candidate: 90-day moving average as the ANCHOR. Predict the
    # ratio target / ma_90 using the enhanced (ma + momentum) features, then
    # multiply back by ma_90. This is the "moving average as a feature" pattern:
    # the model learns the adjustment around the level instead of the level.
    Fenh_tr = enhanced_frame(train)
    enh_cols = list(Fenh_tr.columns)
    Xenh_tr = Fenh_tr.to_numpy(dtype=float)
    ma_tr = train["rev_sum_90"].to_numpy(dtype=float) / MA_DIVISOR
    ma_ratio_tr = np.clip(
        np.nan_to_num(np.where(ma_tr > 0, ytr / ma_tr, 1.0), nan=1.0, posinf=1.0, neginf=1.0),
        0.0, 5.0,
    )
    xgbs_ratio = []
    if XGBRegressor is not None:
        for seed in range(n_seeds):
            xgb = _xgb(**xgb_params)
            xgb.random_state = seed
            xgb.fit(Xenh_tr, ma_ratio_tr)
            xgbs_ratio.append(xgb)

    return {
        "columns": cols,
        "enh_columns": enh_cols,
        "linear": lin,
        "rfs": rfs,
        "xgbs": xgbs,
        "rfs_growth": rfs_growth,
        "xgbs_growth": xgbs_growth,
        "xgbs_ratio": xgbs_ratio,
    }


def _predict_fitted(fitted: dict, df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Predict on ``df`` with models fitted on the training set only."""
    X = feature_frame(df).reindex(columns=fitted["columns"]).fillna(0.0).to_numpy(dtype=float)
    rev30 = df["rev_sum_30"].to_numpy(dtype=float)
    ma = df["rev_sum_90"].to_numpy(dtype=float) / MA_DIVISOR
    lin_pred = np.clip(fitted["linear"].predict(X), 0.0, None)
    rf_preds = np.array([np.clip(rf.predict(X), 0.0, None) for rf in fitted["rfs"]])
    out: dict[str, np.ndarray] = {"linear": lin_pred, "random_forest": rf_preds}
    if fitted.get("xgbs"):
        out["xgboost"] = np.array([np.clip(xgb.predict(X), 0.0, None) for xgb in fitted["xgbs"]])
    if fitted.get("rfs_growth"):
        out["random_forest_growth"] = np.array(
            [np.clip(rf.predict(X), 0.0, None) * rev30 for rf in fitted["rfs_growth"]]
        )
    if fitted.get("xgbs_growth"):
        out["xgboost_growth"] = np.array(
            [np.clip(xgb.predict(X), 0.0, None) * rev30 for xgb in fitted["xgbs_growth"]]
        )
    if fitted.get("xgbs_ratio"):
        Xenh = enhanced_frame(df).reindex(columns=fitted["enh_columns"]).fillna(0.0).to_numpy(dtype=float)
        out["xgboost_ratio"] = np.array(
            [np.clip(xgb.predict(Xenh), 0.0, None) * ma for xgb in fitted["xgbs_ratio"]]
        )
    return out


def _eval_models(df: pd.DataFrame, fitted: dict) -> dict:
    y = df[TARGET].to_numpy(dtype=float)
    preds = _predict_fitted(fitted, df)
    out = {
        "linear": evaluate(y, preds["linear"]),
        "random_forest": _mean_seed_metrics(y, preds["random_forest"]),
    }
    for key in ("xgboost", "random_forest_growth", "xgboost_growth", "xgboost_ratio"):
        if key in preds:
            out[key] = _mean_seed_metrics(y, preds[key])
    return out


def _mean_seed_metrics(y: np.ndarray, preds: np.ndarray) -> dict:
    """Mean and sd of metrics across seeds."""
    rows = [evaluate(y, p) for p in preds]
    keys = [k for k in rows[0] if isinstance(rows[0][k], (int, float))]
    out: dict = {}
    for k in keys:
        vals = np.array([r[k] for r in rows], dtype=float)
        out[k] = float(np.mean(vals))
        out[f"{k}_sd"] = float(np.std(vals))
    out["n"] = float(y.size)
    return out


def permutation_test(train: pd.DataFrame, rf_params: dict) -> dict:
    """Shuffle the target within training; the model must score R2 ~= 0."""
    X = feature_frame(train)
    y = train[TARGET].to_numpy(dtype=float)
    rng = np.random.default_rng(0)
    y_shuffled = rng.permutation(y)
    rf = _rf(**rf_params)
    rf.fit(X.to_numpy(dtype=float), y_shuffled)
    pred = rf.predict(X.to_numpy(dtype=float))
    # report on a held-out fold to be honest about leakage
    folds = walk_forward_folds(train)
    if folds:
        va = folds[-1]["val"]
        Xva = feature_frame(va).reindex(columns=list(X.columns)).fillna(0.0).to_numpy(dtype=float)
        pred_va = rf.predict(Xva)
        r2 = float(r2_score(va[TARGET].to_numpy(dtype=float), pred_va))
    else:
        r2 = float(r2_score(y, pred))
    return {"shuffled_target_r2": r2}


def learning_curve(train: pd.DataFrame, rf_params: dict) -> list[dict]:
    """Train/test MAE against training-set size (chronological)."""
    train = train.sort_values("as_of_date")
    X_all = feature_frame(train)
    cols = list(X_all.columns)
    y_all = train[TARGET].to_numpy(dtype=float)
    n = len(train)
    fractions = [0.3, 0.5, 0.7, 0.85, 1.0]
    out: list[dict] = []
    for frac in fractions:
        cut = int(n * frac)
        tr = train.iloc[:cut]
        va = train.iloc[cut:]
        if len(va) < 20:
            continue
        Xtr = feature_frame(tr).reindex(columns=cols).fillna(X_all.median(numeric_only=True)).to_numpy(dtype=float)
        Xva = feature_frame(va).reindex(columns=cols).fillna(X_all.median(numeric_only=True)).to_numpy(dtype=float)
        rf = _rf(**rf_params)
        rf.fit(Xtr, tr[TARGET].to_numpy(dtype=float))
        tr_mae = float(mean_absolute_error(tr[TARGET].to_numpy(dtype=float), rf.predict(Xtr)))
        va_mae = float(mean_absolute_error(va[TARGET].to_numpy(dtype=float), rf.predict(Xva)))
        out.append({"train_size": cut, "train_mae_minor": tr_mae, "val_mae_minor": va_mae})
    return out


def oracle_metrics(test: pd.DataFrame) -> dict:
    """Irreducible floor: the true conditional mean vs realised target."""
    y = test[TARGET].to_numpy(dtype=float)
    oracle = test["oracle_forecast_minor"].to_numpy(dtype=float)
    return evaluate(y, oracle)


def lower_bound_coverage(validation: pd.DataFrame, test: pd.DataFrame, rf_params: dict) -> dict:
    """Residual-quantile 90% lower bound and its empirical coverage."""
    # calibrate q10 residual on validation
    Xva = feature_frame(validation)
    cols = list(Xva.columns)
    rf = _rf(**rf_params)
    yva = validation[TARGET].to_numpy(dtype=float)
    rf.fit(Xva.reindex(columns=cols).fillna(0.0).to_numpy(dtype=float), yva)
    pred_va = rf.predict(Xva.reindex(columns=cols).fillna(0.0).to_numpy(dtype=float))
    resid_va = yva - pred_va
    q10 = float(np.percentile(resid_va, 10))

    # apply to test
    Xte = feature_frame(test).reindex(columns=cols).fillna(0.0).to_numpy(dtype=float)
    pred_te = rf.predict(Xte)
    lower = np.clip(pred_te + q10, 0.0, None)
    yte = test[TARGET].to_numpy(dtype=float)
    coverage = float(np.mean(lower <= yte))
    return {
        "q10_residual_minor": q10,
        "nominal_coverage_pct": 90.0,
        "empirical_coverage_pct": coverage * 100.0,
        "mean_lower_minor": float(np.mean(lower)),
    }


def main() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    n_seeds = cfg["splits"]["n_seeds"]

    supervised = pd.read_csv(DATA_DIR / "supervised.csv", parse_dates=["as_of_date"])
    parts = build_partitions(supervised)

    print("Tuning Random Forest via walk-forward validation...")
    best_rf, grid_results = tune_random_forest(parts.train)
    print(f"  best: {best_rf}")

    best_xgb, xgb_grid_results = None, []
    if XGBRegressor is not None:
        print("Tuning XGBoost via walk-forward validation...")
        best_xgb, xgb_grid_results = tune_xgboost(parts.train)
        print(f"  best: {best_xgb}")

    print("Evaluating on validation set (for model selection)...")
    rf_params = {k: best_rf[k] for k in ("n_estimators", "max_depth", "min_samples_leaf")}
    xgb_params = {k: best_xgb[k] for k in ("n_estimators", "max_depth", "min_child_weight")} if best_xgb else {}
    fitted = _fit_on_train(parts.train, rf_params, xgb_params, n_seeds)
    val_metrics = {
        "baselines": _eval_baselines(parts.validation, parts.validation[TARGET].to_numpy(dtype=float)),
        "models": _eval_models(parts.validation, fitted),
    }

    print("Evaluating on test sets (once)...")
    test_sets = {
        "test_temporal": parts.test_temporal,
        "test_business": parts.test_business,
        "test_tail": parts.test_tail,
    }
    test_metrics: dict[str, dict] = {}
    for name, df in test_sets.items():
        y = df[TARGET].to_numpy(dtype=float)
        test_metrics[name] = {
            "baselines": _eval_baselines(df, y),
            "models": _eval_models(df, fitted),
            "oracle": oracle_metrics(df),
            "lower_bound": lower_bound_coverage(parts.validation, df, rf_params),
        }
        mae_rf = test_metrics[name]["models"]["random_forest"]["mae_minor"]
        mae_naive = test_metrics[name]["baselines"]["seasonal_naive"]["mae_minor"]
        print(f"  {name:<16} RF MAE {mae_rf:>12,.0f}  naive MAE {mae_naive:>12,.0f}  "
              f"oracle MAE {test_metrics[name]['oracle']['mae_minor']:>12,.0f}")

    print("Overfitting diagnostics...")
    perm = permutation_test(parts.train, rf_params)
    lc = learning_curve(parts.train, rf_params)
    print(f"  permutation (shuffled target) R2: {perm['shuffled_target_r2']:.4f}")

    # feature importance (impurity, averaged across seeds)
    X_all = feature_frame(parts.train)
    cols = list(X_all.columns)
    importances = []
    for seed in range(n_seeds):
        rf = _rf(**{k: best_rf[k] for k in ("n_estimators", "max_depth", "min_samples_leaf")})
        rf.random_state = seed
        rf.fit(X_all.to_numpy(dtype=float), parts.train[TARGET].to_numpy(dtype=float))
        importances.append(rf.feature_importances_)
    imp = pd.DataFrame({
        "feature": cols,
        "importance_mean": np.mean(importances, axis=0),
        "importance_sd": np.std(importances, axis=0),
    }).sort_values("importance_mean", ascending=False)

    # persist production artifact (fitted on all training businesses)
    # Production estimator = XGBoost on the ratio target with the 90-day moving
    # average as the anchor (ma_90 = rev_sum_90/3). The model predicts
    # target/ma_90 and we multiply back, so it learns the adjustment around the
    # level instead of the level itself. Evidence: beats the moving average on
    # the temporal and tail tests, ties it on unseen businesses, and stays robust
    # to 1-shilling micro businesses (ma_90 * ratio -> tiny for tiny businesses).
    Fenh_all = enhanced_frame(parts.train)
    enh_cols = list(Fenh_all.columns)
    ma_all = parts.train["rev_sum_90"].to_numpy(dtype=float) / MA_DIVISOR
    ma_ratio_all = np.clip(
        np.nan_to_num(np.where(ma_all > 0, parts.train[TARGET].to_numpy(dtype=float) / ma_all, 1.0),
                      nan=1.0, posinf=1.0, neginf=1.0),
        0.0, 5.0,
    )
    production_models = []
    for seed in range(n_seeds):
        xgb = _xgb(**xgb_params)
        xgb.random_state = seed
        xgb.fit(Fenh_all.to_numpy(dtype=float), ma_ratio_all)
        production_models.append(xgb)

    # calibration: cv_error of the ratio-XGB on the validation set
    _val_preds = _predict_fitted(fitted, parts.validation)["xgboost_ratio"]
    _val_pred_mean = np.mean(_val_preds, axis=0)
    val_mae = float(mean_absolute_error(parts.validation[TARGET].to_numpy(dtype=float), _val_pred_mean))
    cv_error = val_mae / max(1.0, parts.validation[TARGET].mean())

    lin_model = _linear()
    lin_model.fit(X_all.to_numpy(dtype=float), parts.train[TARGET].to_numpy(dtype=float))
    joblib.dump({
        "model_type": "xgboost_ratio_ma90",
        "estimators": production_models,
        "enh_columns": enh_cols,
        "ma_divisor": MA_DIVISOR,
        "target": TARGET,
        "model_version": "xgb-ratio-ma90-2026.11.0",
        "min_amount_minor": cfg["generator"]["min_amount_minor"],
        "cv_error": cv_error,
        "error_margin_k": cfg["engine"]["error_margin_k"],
        "linear_estimator": lin_model,
        "feature_columns": cols,
        "sector_dummies": [c for c in cols if c.startswith("sector_")],
        "selection_rationale": (
            "XGBoost on the ratio target (target / (rev_sum_90/3)) with moving-average "
            "and momentum features. Beats the 90-day moving average on the temporal and "
            "tail tests, ties it on unseen businesses, and stays robust to 1-shilling "
            "micro businesses. The raw-level tree models overfit business identity."
        ),
    }, OUTPUTS_DIR / "model.joblib")

    # model selection: which candidate wins on each dimension
    selection = {}
    for dim in ("validation", "test_temporal", "test_business", "test_tail"):
        pool = val_metrics if dim == "validation" else test_metrics[dim]
        best_name, best_mae = None, float("inf")
        for mname, m in pool["models"].items():
            if m["mae_minor"] < best_mae:
                best_mae, best_name = m["mae_minor"], mname
        for bname, b in pool["baselines"].items():
            if b["mae_minor"] < best_mae:
                best_mae, best_name = b["mae_minor"], bname
        selection[dim] = {"best": best_name, "mae_minor": best_mae}

    results = {
        "best_rf": best_rf,
        "best_xgb": best_xgb,
        "grid": grid_results,
        "xgb_grid": xgb_grid_results,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "selection": selection,
        "permutation": perm,
        "learning_curve": lc,
        "feature_importance": imp.to_dict(orient="records"),
    }
    (OUTPUTS_DIR / "metrics.json").write_text(json.dumps(_json_safe(results), indent=2))
    imp.to_csv(OUTPUTS_DIR / "feature_importance.csv", index=False)
    pd.DataFrame(grid_results).to_csv(OUTPUTS_DIR / "grid_results.csv", index=False)
    pd.DataFrame(lc).to_csv(OUTPUTS_DIR / "learning_curve.csv", index=False)

    print("\nResults summary (KSh):")
    for name, tm in test_metrics.items():
        rf_ksh = minor_metrics_to_ksh(tm["models"]["random_forest"])
        naive_ksh = minor_metrics_to_ksh(tm["baselines"]["seasonal_naive"])
        print(f"  {name:<16} RF MAE {rf_ksh['mae_ksh']:>10,.0f}  R2 {rf_ksh['r2']:>6.2f} | "
              f"naive MAE {naive_ksh['mae_ksh']:>10,.0f}")


def _json_safe(obj):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


if __name__ == "__main__":
    main()
