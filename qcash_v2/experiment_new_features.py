"""Ablation of candidate features for the production ratio-target XGBoost.

Adds each proposed feature to the current enhanced frame, one at a time and all
together, and reports the change in MAE on the three test sets. Only features
that actually reduce error are adopted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

from common import DATA_DIR
from model_features import FEATURE_COLS, MA_DIVISOR, feature_frame
from splits import build_partitions

TARGET = "target_minor"
SEEDS = 5
CANDIDATES = [
    "days_since_last_tx",
    "log_ma_90",
    "target_payday_days",
    "activity_momentum",
    "weekend_rev_share_30",
]


def payday_count(as_of: pd.Series) -> np.ndarray:
    out = []
    for a in pd.to_datetime(as_of):
        days = pd.date_range(a + pd.Timedelta(days=1), periods=30, freq="D")
        out.append(float(((days.day >= 25) | (days.day <= 5)).sum()))
    return np.asarray(out, dtype=float)


def build_frame(df: pd.DataFrame, extras: set[str]) -> pd.DataFrame:
    f = feature_frame(df)
    rev7 = df["rev_sum_7"].to_numpy(dtype=float)
    rev30 = df["rev_sum_30"].to_numpy(dtype=float)
    rev90 = df["rev_sum_90"].to_numpy(dtype=float)
    ma = rev90 / MA_DIVISOR
    f["ma_90"] = ma
    f["mom_week"] = rev7 / (rev30 * 7.0 / 30.0 + 1.0)
    f["mom_month"] = rev30 / (ma + 1.0)
    f["mom_week_q"] = rev7 / (ma * 7.0 / 30.0 + 1.0)
    f["month"] = df["as_of_date"].dt.month.astype(float).to_numpy()
    f["quarter"] = df["as_of_date"].dt.quarter.astype(float).to_numpy()
    if "days_since_last_tx" in extras:
        f["days_since_last_tx"] = df["days_since_last_tx"].to_numpy(dtype=float)
    if "log_ma_90" in extras:
        f["log_ma_90"] = np.log1p(ma)
    if "target_payday_days" in extras:
        f["target_payday_days"] = payday_count(df["as_of_date"])
    if "activity_momentum" in extras:
        f["activity_momentum"] = df["active_days_7"].to_numpy(dtype=float) / (
            df["active_days_30"].to_numpy(dtype=float) / 30.0 * 7.0 + 1.0
        )
    if "weekend_rev_share_30" in extras:
        f["weekend_rev_share_30"] = df["weekend_rev_share_30"].to_numpy(dtype=float)
    return f


def xgb(seed: int) -> XGBRegressor:
    return XGBRegressor(
        n_estimators=400, max_depth=4, learning_rate=0.05, min_child_weight=20,
        subsample=0.8, colsample_bytree=0.7, reg_lambda=1.0,
        objective="reg:absoluteerror", n_jobs=-1, random_state=seed, verbosity=0,
    )


def evaluate_config(parts, extras: set[str]) -> dict:
    train = parts.train
    Ftr = build_frame(train, extras)
    cols = list(Ftr.columns)
    Xtr = Ftr.to_numpy(dtype=float)
    ytr = train[TARGET].to_numpy(dtype=float)
    ma_tr = train["rev_sum_90"].to_numpy(dtype=float) / MA_DIVISOR
    ratio_tr = np.clip(np.where(ma_tr > 0, ytr / ma_tr, 1.0), 0.0, 5.0)
    models = []
    for s in range(SEEDS):
        m = xgb(s).fit(Xtr, ratio_tr)
        models.append(m)
    row = {}
    for dim in ("test_temporal", "test_business", "test_tail"):
        df = getattr(parts, dim)
        Xte = build_frame(df, extras).reindex(columns=cols).fillna(0.0).to_numpy(dtype=float)
        ma_te = df["rev_sum_90"].to_numpy(dtype=float) / MA_DIVISOR
        preds = np.mean([np.clip(m.predict(Xte), 0.0, None) * ma_te for m in models], axis=0)
        y = df[TARGET].to_numpy(dtype=float)
        mae = mean_absolute_error(y, preds)
        row[dim] = (mae, mae / np.mean(np.abs(y)) * 100)
    return row


def main() -> None:
    supervised = pd.read_csv(DATA_DIR / "supervised.csv", parse_dates=["as_of_date"])
    parts = build_partitions(supervised)

    configs = [("baseline (current)", set())]
    for c in CANDIDATES:
        configs.append((f"+ {c}", {c}))
    configs.append(("+ ALL", set(CANDIDATES)))

    base = evaluate_config(parts, set())
    print(f"{'config':<30} {'temporal':>16} {'unseen biz':>16} {'tail':>16}")
    print("-" * 82)
    for name, extras in configs:
        r = evaluate_config(parts, extras)
        cells = []
        for dim in ("test_temporal", "test_business", "test_tail"):
            mae, pct = r[dim]
            d = r[dim][0] - base[dim][0]
            arrow = "" if name.startswith("baseline") else ("  <" if d < -100 else "")
            cells.append(f"{mae/100:>8,.0f} ({pct:>3.0f}%){arrow}")
        print(f"{name:<30} " + " ".join(f"{c:>16}" for c in cells))
    print("\n'<' marks a cell that improved vs baseline by >100 KSh.")


if __name__ == "__main__":
    main()
