"""Experiment: use the 90-day moving average as an engineered feature / anchor.

Question: can we make the ML models better by engineering the 90-day moving
average into the inputs, as real credit/earnings systems do?

Three ways this is usually attempted:

  A. Add `ma_90` as a plain extra feature (level target).
     -> for trees this is a monotonic rescaling of rev_sum_90, so no change.

  B. Add scale-free momentum features (rev7/run-rate, rev30/rev90) on top.
     -> gives the model the AR(1) mean-reversion signal the raw level hides.

  C. Hybrid / residual: ma_90 is the baseline, the model learns the KSh
     residual (target - ma_90) with an absolute-error loss, then we add the
     baseline back. This is the "boost from the moving-average offset" pattern.

Everything is evaluated on the same business/temporal/tail holdouts, only using
information available before the cut-off.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from common import DATA_DIR, load_config
from splits import build_partitions
from train import FEATURE_COLS, TARGET, feature_frame

MA_DIVISOR = 3.0
SEEDS = 3


def enhanced_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Base features + the 90-day moving average and scale-free momentum."""
    base = feature_frame(df)
    ma = df["rev_sum_90"].to_numpy(dtype=float) / MA_DIVISOR
    eps = 1.0
    out = base.copy()
    out["ma_90"] = ma
    out["mom_week"] = df["rev_sum_7"].to_numpy(dtype=float) / (df["rev_sum_30"].to_numpy(dtype=float) * 7.0 / 30.0 + eps)
    out["mom_month"] = df["rev_sum_30"].to_numpy(dtype=float) / (ma + eps)
    out["mom_week_q"] = df["rev_sum_7"].to_numpy(dtype=float) / (ma * 7.0 / 30.0 + eps)
    out["month"] = df["as_of_date"].dt.month.astype(float).to_numpy()
    out["quarter"] = df["as_of_date"].dt.quarter.astype(float).to_numpy()
    return out


def _xgb(seed: int) -> XGBRegressor:
    return XGBRegressor(
        n_estimators=400, max_depth=4, learning_rate=0.05, min_child_weight=20,
        subsample=0.8, colsample_bytree=0.7, reg_lambda=1.0,
        objective="reg:absoluteerror", n_jobs=-1, random_state=seed, verbosity=0,
    )


def _rf(seed: int) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=400, max_depth=6, min_samples_leaf=20, max_features=0.5,
        n_jobs=-1, random_state=seed,
    )


def _ridge() -> Pipeline:
    return Pipeline([("scale", StandardScaler()), ("m", Ridge(alpha=1.0))])


def _fit_predict_level(model, Xtr, ytr, Xte) -> np.ndarray:
    model.fit(Xtr, ytr)
    return np.clip(model.predict(Xte), 0.0, None)


def main() -> None:
    supervised = pd.read_csv(DATA_DIR / "supervised.csv", parse_dates=["as_of_date"])
    parts = build_partitions(supervised)
    train = parts.train

    # feature matrices
    Ftr_base = feature_frame(train)
    Ftr_enh = enhanced_frame(train)
    base_cols = list(Ftr_base.columns)
    enh_cols = list(Ftr_enh.columns)

    ma_tr = train["rev_sum_90"].to_numpy(dtype=float) / MA_DIVISOR
    ytr = train[TARGET].to_numpy(dtype=float)
    resid_tr = ytr - ma_tr
    ratio_tr = np.clip(np.where(ma_tr > 0, ytr / ma_tr, 1.0), 0.0, 5.0)

    def X(df, cols, enhanced: bool) -> np.ndarray:
        f = enhanced_frame(df) if enhanced else feature_frame(df)
        return f.reindex(columns=cols).fillna(0.0).to_numpy(dtype=float)

    # pre-build test matrices per partition
    def test_frame(df):
        return df

    Xtr_base = X(train, base_cols, False)
    Xtr_enh = X(train, enh_cols, True)

    methods = {}

    def add(name, preds_by_split):
        methods[name] = preds_by_split

    # evaluate everywhere
    results = {}

    def evaluate_all(name, predict_fn):
        row = {}
        for dim in ("test_temporal", "test_business", "test_tail"):
            df = getattr(parts, dim)
            y = df[TARGET].to_numpy(dtype=float)
            pred = np.clip(np.nan_to_num(predict_fn(df), nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)
            mean_actual = float(np.mean(np.abs(y)))
            row[dim] = {
                "mae": mean_absolute_error(y, pred),
                "pct": mean_absolute_error(y, pred) / mean_actual * 100,
                "r2": r2_score(y, pred),
            }
        results[name] = row

    # 1. moving average baseline
    evaluate_all("moving average (baseline)", lambda df: df["rev_sum_90"].to_numpy(dtype=float) / MA_DIVISOR)

    # 2. linear / RF / XGB on level, base features (current production candidates)
    ridge = _ridge().fit(Xtr_base, ytr)
    evaluate_all("linear (base features)", lambda df: ridge.predict(X(df, base_cols, False)))

    def multi_seed_level(model_fn, enhanced: bool):
        def predict(df):
            Xte = X(df, enh_cols if enhanced else base_cols, enhanced)
            preds = []
            for s in range(SEEDS):
                m = model_fn(s).fit(Xtr_enh if enhanced else Xtr_base, ytr)
                preds.append(np.clip(m.predict(Xte), 0.0, None))
            return np.mean(preds, axis=0)
        return predict

    evaluate_all("random forest (level, base)", multi_seed_level(_rf, False))
    evaluate_all("xgboost (level, base)", multi_seed_level(_xgb, False))

    # 3. EXPERIMENT A: add ma_90 + momentum as plain features, level target
    evaluate_all("xgboost (level, +ma/momentum features)", multi_seed_level(_xgb, True))
    evaluate_all("random forest (level, +ma/momentum features)", multi_seed_level(_rf, True))

    # 4. EXPERIMENT B: ratio target = target / ma_90, enhanced features
    def multi_seed_ratio(model_fn):
        def predict(df):
            Xte = X(df, enh_cols, True)
            ma_te = df["rev_sum_90"].to_numpy(dtype=float) / MA_DIVISOR
            preds = []
            for s in range(SEEDS):
                m = model_fn(s).fit(Xtr_enh, ratio_tr)
                preds.append(np.clip(m.predict(Xte), 0.0, None) * ma_te)
            return np.mean(preds, axis=0)
        return predict

    evaluate_all("xgboost (ratio target, +ma features)", multi_seed_ratio(_xgb))
    evaluate_all("random forest (ratio target, +ma features)", multi_seed_ratio(_rf))

    # 5. EXPERIMENT C: hybrid residual — ma_90 baseline + learned KSh residual
    def multi_seed_resid(model_fn):
        def predict(df):
            Xte = X(df, enh_cols, True)
            ma_te = df["rev_sum_90"].to_numpy(dtype=float) / MA_DIVISOR
            preds = []
            for s in range(SEEDS):
                m = model_fn(s).fit(Xtr_enh, resid_tr)
                preds.append(ma_te + m.predict(Xte))
            return np.mean(preds, axis=0)
        return predict

    evaluate_all("HYBRID: ma + xgboost residual", multi_seed_resid(_xgb))
    evaluate_all("HYBRID: ma + rf residual", multi_seed_resid(_rf))

    # 6. oracle reference
    evaluate_all("oracle (floor)", lambda df: df["oracle_forecast_minor"].to_numpy(dtype=float))

    # print table
    print(f"{'model':<45} {'temporal':>16} {'unseen biz':>16} {'tail':>16}")
    print("-" * 96)
    for name, row in results.items():
        cells = []
        for dim in ("test_temporal", "test_business", "test_tail"):
            cells.append(f"{row[dim]['mae']/100:>8,.0f} ({row[dim]['pct']:>3.0f}%)")
        print(f"{name:<45} " + " ".join(f"{c:>16}" for c in cells))

    # winner per dimension
    print("\nBest model per test set (MAE):")
    for dim in ("test_temporal", "test_business", "test_tail"):
        best = min(results.items(), key=lambda kv: kv[1][dim]["mae"] if kv[0] != "oracle (floor)" else 1e18)
        print(f"  {dim:<16} -> {best[0]}  ({best[1][dim]['mae']/100:,.0f} KSh, {best[1][dim]['pct']:.0f}%)")


if __name__ == "__main__":
    main()
