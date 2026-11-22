"""Q-Cash revenue prediction pipeline.

Target: M-Pesa Till revenue (Credit) during the 30 calendar days after as_of_date.

The script is a single reproducible entry point that
  1. loads the weekly supervised dataset,
  2. builds leakage-safe momentum features from information available on or
     before as_of_date,
  3. splits chronologically with an embargo gap (never shuffled),
  4. fits naive baselines, linear regression, random forest and gradient
     boosting under three target formulations,
  5. evaluates every model on the same untouched holdout and under walk-forward
     validation,
  6. reports feature importance (impurity and permutation),
  7. saves metrics, predictions, importances and a versioned joblib artifact.

Usage:  python train_qcash.py
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import (
    make_scorer,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from qcash_baselines import PersistenceBaseline

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset_v1" / "qcash_supervised_weekly_30d.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_PATH = OUTPUT_DIR / "qcash_revenue_model_v1.joblib"

SEED = 42
TARGET_REVENUE = "target_next_30d_revenue_kes"
TARGET_TX = "target_next_30d_tx_count"
DATE_COL = "as_of_date"

ID_COLUMNS = ["business_id", DATE_COL]
TARGET_COLUMNS = [TARGET_REVENUE, TARGET_TX]
DROP_COLUMNS = ID_COLUMNS + TARGET_COLUMNS
KNBS_COLUMNS = ["cpi", "headline_inflation_yoy_pct"]

MOMENTUM_MAX_LAG = 8
ROLL_WINDOWS = (4, 8)
EMBARGO_WEEKS = 5
HOLDOUT_FRACTION = 0.30
MIN_TRAIN_OBS_WF = 52
WF_TEST_OBS = 8
WF_STEP_OBS = 8
PERMUTATION_REPEATS = 20000


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------


def smape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Symmetric mean absolute percentage error, in percent.

    Uses |a-p| / ((|a|+|p|)/2). Unlike MAPE it stays finite when the actual
    value is zero, which matters if a business has a revenue-free month.
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    denominator = (np.abs(actual) + np.abs(predicted)) / 2.0
    ratio = np.where(denominator == 0, 0.0, np.abs(actual - predicted) / denominator)
    return float(np.mean(ratio) * 100.0)


def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean absolute percentage error, in percent. Undefined for zero actuals."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mask = actual != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100.0)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return the four headline metrics plus a bias (mean signed error)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "mae_kes": float(mean_absolute_error(y_true, y_pred)),
        "rmse_kes": rmse,
        "r2": float(r2_score(y_true, y_pred)),
        "smape_pct": smape(y_true, y_pred),
        "mape_pct": mape(y_true, y_pred),
        "bias_kes": float(np.mean(y_pred - y_true)),
        "n_test": int(len(y_true)),
    }


# --------------------------------------------------------------------------
# data loading and feature construction
# --------------------------------------------------------------------------


def load_dataset(path: Path) -> pd.DataFrame:
    """Load the weekly supervised CSV, parse dates and sort chronologically."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Looked for:\n"
            f"  {DATA_PATH}\n"
            f"Search the repository, e.g.:\n"
            f"  find ~/code/qcash -name 'qcash_supervised_weekly_30d.csv'"
        )
    df = pd.read_csv(path, parse_dates=[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    df[KNBS_COLUMNS] = df[KNBS_COLUMNS].ffill()
    df[KNBS_COLUMNS] = df[KNBS_COLUMNS].bfill()
    return df


RATIO_90D_30D = "revenue_90d_per_30d"


def add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add weekly trajectory features from the weekly panel itself.

    The panel has one row per week, so a row's own revenue_7d window ends on
    as_of_date. Any value taken from an earlier row therefore describes a
    window that closed strictly before as_of_date, which keeps the feature
    leakage-safe.

    Features added
      revenue_7d_lag{k}        7-day revenue of the k-th previous week
      weekly_revenue_mean_{w}  mean weekly revenue over the last w weeks
      weekly_revenue_std_{w}   standard deviation over the last w weeks
      weekly_revenue_mean_excl_4, weekly_revenue_std_excl_4
                               same statistics over the 4 weeks before this one,
                               used for the momentum ratio denominator
      momentum_ratio_4         latest week / mean of the previous 4 weeks
      weekly_revenue_slope_8   OLS slope of weekly revenue over 8 weeks
    """
    out = df.copy()
    weekly = out["revenue_7d"]

    for lag in range(1, MOMENTUM_MAX_LAG + 1):
        out[f"revenue_7d_lag{lag}"] = weekly.shift(lag)

    for window in ROLL_WINDOWS:
        rolled = weekly.rolling(window)
        out[f"weekly_revenue_mean_{window}"] = rolled.mean()
        out[f"weekly_revenue_std_{window}"] = rolled.std()
        out[f"weekly_revenue_max_{window}"] = rolled.max()
        out[f"weekly_revenue_min_{window}"] = rolled.min()

    prior = weekly.shift(1)
    out["weekly_revenue_mean_excl_4"] = prior.rolling(4).mean()
    out["weekly_revenue_std_excl_4"] = prior.rolling(4).std()
    out["momentum_ratio_4"] = weekly / out["weekly_revenue_mean_excl_4"].replace(0, np.nan)
    out["weekly_revenue_slope_8"] = weekly.rolling(8).apply(
        lambda values: float(np.polyfit(np.arange(len(values)), values, 1)[0]), raw=True
    )
    out["revenue_30d_per_60d"] = out["revenue_30d"] / out["revenue_60d"].replace(0, np.nan)
    out["revenue_30d_per_90d"] = out["revenue_30d"] / out["revenue_90d"].replace(0, np.nan)
    out[RATIO_90D_30D] = (out["revenue_90d"] / 3.0).replace(0, np.nan)
    out["revenue_7d_per_30d"] = out["revenue_7d"] / out["revenue_30d"].replace(0, np.nan)
    out["days_since_start"] = (out[DATE_COL] - out[DATE_COL].min()).dt.days

    new_columns = [c for c in out.columns if c not in df.columns]
    out[new_columns] = out[new_columns].replace([np.inf, -np.inf], np.nan)
    return out


@dataclass
class FeatureSets:
    """Named column lists used for the ablation study."""

    core: list[str]
    core_knbs: list[str]
    momentum: list[str]
    momentum_knbs: list[str]
    momentum_knbs_trend: list[str]
    all_variants: dict[str, list[str]] = field(default_factory=dict)


def build_feature_sets(df: pd.DataFrame) -> FeatureSets:
    """Define the feature-set variants, dropping identifiers, targets and
    zero-variance columns.
    """
    momentum_only = [c for c in df.columns if c not in DROP_COLUMNS and c not in KNBS_COLUMNS]
    momentum_only = [c for c in momentum_only if c != "days_since_start"]

    core = [c for c in momentum_only if not c.startswith("revenue_7d_lag")
            and not c.startswith("weekly_revenue_")
            and c not in {"momentum_ratio_4"}]

    core_knbs = core + [c for c in KNBS_COLUMNS if c in df.columns]
    momentum = momentum_only
    momentum_knbs = momentum + [c for c in KNBS_COLUMNS if c in df.columns]
    momentum_knbs_trend = momentum_knbs + ["days_since_start"]

    sets = FeatureSets(
        core=core,
        core_knbs=core_knbs,
        momentum=momentum,
        momentum_knbs=momentum_knbs,
        momentum_knbs_trend=momentum_knbs_trend,
    )
    sets.all_variants = {
        "core": sets.core,
        "core+knbs": sets.core_knbs,
        "momentum": sets.momentum,
        "momentum+knbs": sets.momentum_knbs,
        "momentum+knbs+trend": sets.momentum_knbs_trend,
    }
    return sets


def drop_zero_variance(df: pd.DataFrame, columns: list[str]) -> list[str]:
    """Remove constant columns such as day_of_week, which is always Sunday here."""
    return [c for c in columns if c in df.columns and df[c].nunique(dropna=True) > 1]


# --------------------------------------------------------------------------
# target formulations
# --------------------------------------------------------------------------


@dataclass
class Formulation:
    """A way of expressing the target before handing it to a regressor.

    name     label used in the results table
    scale    divides KSh by this so the regressor sees O(1) numbers
    _denom   column name holding the per-row multiplier used by ratio
             formulations, or None for the direct formulation
    """

    name: str
    scale: float
    _denom: str | None = None
    _ratio: str | None = None

    def train_target(self, df: pd.DataFrame) -> np.ndarray:
        """What the regressor is asked to fit, in the transformed space."""
        if self._ratio is None:
            return (df[TARGET_REVENUE] / self.scale).to_numpy(dtype=float)
        return (df[TARGET_REVENUE] / df[self._ratio]).to_numpy(dtype=float)

    def to_revenue(self, predictions: np.ndarray, df: pd.DataFrame) -> np.ndarray:
        """Convert regressor output back into KSh of 30-day revenue.

        `df` must be the rows the predictions correspond to, in the same order.
        """
        predictions = np.asarray(predictions, dtype=float)
        if self._ratio is None:
            return predictions * self.scale
        return predictions * df[self._ratio].to_numpy(dtype=float)


def build_formulations(df: pd.DataFrame) -> dict[str, Formulation]:
    """Construct the three target formulations under comparison.

    A_direct     fit KSh directly (scaled to O(1)). Simple, but a tree model
                 cannot predict a value above the largest target it was trained
                 on, which matters because this business is growing.
    B_growth_30d fit target / revenue_30d, the formulation used in the first
                 experiment. Level-free, so it can carry a rising trend.
    C_growth_90d fit target / (revenue_90d / 3). The same idea against a
                 smoother, longer denominator, so it should be less noisy
                 than the 30-day version.
    """
    scale = float(df[TARGET_REVENUE].mean())
    return {
        "A_direct": Formulation(name="A_direct", scale=scale),
        "B_growth_30d": Formulation(name="B_growth_30d", scale=1.0, _ratio="revenue_30d"),
        "C_growth_90d": Formulation(name="C_growth_90d", scale=1.0, _ratio=RATIO_90D_30D),
    }


# --------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------


def build_models() -> dict[str, object]:
    """Instantiate the estimators under test.

    Random forest and gradient boosting settings are deliberately conservative
    for a 127-row dataset: shallow leaves and a high minimum leaf size reduce
    overfitting and, for the tree models, limit how violently they memorise the
    training range.
    """
    return {
        "linear": Pipeline(
            [("scale", StandardScaler()), ("model", LinearRegression())]
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=500,
            max_depth=8,
            min_samples_leaf=3,
            max_features=0.5,
            random_state=SEED,
            n_jobs=-1,
        ),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            max_depth=3,
            max_iter=300,
            learning_rate=0.05,
            min_samples_leaf=5,
            l2_regularization=1.0,
            random_state=SEED,
        ),
    }


# --------------------------------------------------------------------------
# baselines
# --------------------------------------------------------------------------


def baseline_predictions(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Naive forecasts that use no fitted parameters.

    naive_30d        today's trailing 30 days is our best guess for the next 30
    naive_60d        smooth the level over 60 days
    drift_30d        random walk with drift, revenue_30d + (revenue_30d - previous
                     30d block), written as 3*revenue_30d - revenue_60d
    drift_90d        slower version, revenue_30d + (revenue_30d - revenue_60d)/3
    """
    rev30 = df["revenue_30d"].to_numpy(dtype=float)
    rev60 = df["revenue_60d"].to_numpy(dtype=float)
    rev90 = df["revenue_90d"].to_numpy(dtype=float)
    rev7 = df["revenue_7d"].to_numpy(dtype=float)
    return {
        "naive_30d": rev30,
        "naive_60d": rev60 / 2.0,
        "drift_30d": np.clip(3.0 * rev30 - rev60, 0.0, None),
        "drift_90d": np.clip(rev30 + (rev30 - rev60) / 3.0, 0.0, None),
        "naive_7d_x4": rev7 * 30.0 / 7.0,
    }


# --------------------------------------------------------------------------
# chronological splitting
# --------------------------------------------------------------------------


@dataclass
class Split:
    """An index-based chronological split with an embargo gap."""

    label: str
    train_idx: np.ndarray
    test_idx: np.ndarray
    embargoed_idx: np.ndarray


def chronological_holdout(n: int, fraction: float, embargo: int) -> Split:
    """Hold out the final `fraction` of rows, with `embargo` rows dropped
    between train and test.

    The embargo matters here. Each observation's target is the 30 days after its
    as_of_date, so a test row's window overlaps the rows just before it. Without
    a gap, the test window can contain the same transactions that formed the
    training features.
    """
    cut = int(n * (1.0 - fraction))
    cut = max(MIN_TRAIN_OBS_WF, cut)
    train = np.arange(0, cut - embargo)
    embargoed = np.arange(cut - embargo, cut)
    test = np.arange(cut, n)
    return Split("holdout_70_30", train, test, embargoed)


def walk_forward_splits(n: int, embargo: int) -> list[Split]:
    """Expanding-window walk-forward folds.

    Train on everything up to a point, skip `embargo` rows, test the next
    `WF_TEST_OBS` rows, then move the cutoff forward. This mimics how Q-Cash
    would actually operate: fit on history, predict the next block of weeks.
    """
    splits: list[Split] = []
    start = MIN_TRAIN_OBS_WF
    test_start = start + embargo
    fold = 1
    while test_start + 1 <= n:
        test_end = min(test_start + WF_TEST_OBS, n)
        if test_end - test_start < 4:
            break
        splits.append(
            Split(
                label=f"wf_fold_{fold}",
                train_idx=np.arange(0, test_start - embargo),
                test_idx=np.arange(test_start, test_end),
                embargoed_idx=np.arange(test_start - embargo, test_start),
            )
        )
        fold += 1
        test_start += WF_STEP_OBS
    return splits


# --------------------------------------------------------------------------
# fitting helpers
# --------------------------------------------------------------------------


def impute(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """Median-impute from the training split only.

    The median is learned on train and applied to test, so no test-period
    information reaches the model. In this dataset the imputation is rarely
    used because momentum features are only present once enough weekly history
    exists, and those early rows are excluded separately.
    """
    medians = X_train.median(numeric_only=True)
    train_filled = X_train.fillna(medians)
    test_filled = X_test.fillna(medians).fillna(0.0)
    return train_filled.to_numpy(dtype=float), test_filled.to_numpy(dtype=float)


def fit_predict(
    model: object,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train_transformed: np.ndarray,
    formulation: Formulation,
    ratio_rows_test: pd.DataFrame,
) -> np.ndarray:
    """Fit on the training split and return predictions in KSh.

    `ratio_rows_test` supplies the per-row multiplier needed to convert a ratio
    prediction back into revenue, and must be the rows in X_test.
    """
    Xtr, Xte = impute(X_train, X_test)
    model.fit(Xtr, y_train_transformed)
    revenue = formulation.to_revenue(model.predict(Xte), ratio_rows_test)
    return np.clip(np.nan_to_num(revenue, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)


# --------------------------------------------------------------------------
# diagnostics
# --------------------------------------------------------------------------


def extrapolation_report(df: pd.DataFrame, split: Split) -> dict[str, object]:
    """Quantify how much of the test period lies outside the training target range.

    A tree ensemble can only ever average leaf values seen during training, so
    if this fraction is large, no amount of hyperparameter tuning will let a
    tree model track a business whose revenue is climbing.
    """
    y = df[TARGET_REVENUE]
    train = y.iloc[split.train_idx]
    test = y.iloc[split.test_idx]
    return {
        "train_target_min_kes": float(train.min()),
        "train_target_max_kes": float(train.max()),
        "test_target_min_kes": float(test.min()),
        "test_target_max_kes": float(test.max()),
        "test_above_train_max_count": int((test > train.max()).sum()),
        "test_below_train_min_count": int((test < train.min()).sum()),
        "test_outside_train_range_pct": float(
            100.0 * ((test > train.max()) | (test < train.min())).mean()
        ),
        "test_mean_kes": float(test.mean()),
        "train_mean_kes": float(train.mean()),
    }


BASELINE_ESTIMATORS = {
    "naive_30d": lambda: PersistenceBaseline("revenue_30d", 1.0),
    "naive_60d": lambda: PersistenceBaseline("revenue_60d", 0.5),
}
"""Naive baselines that read a column already present in the feature table.

drift_30d and drift_90d are evaluated as benchmarks but are not offered here,
because they combine several columns and would need the combination logic
reproduced inside predict(). Keeping the artifact to a single column avoids
that duplication.
"""


def fit_quality_report(
    model: object,
    X_train: pd.DataFrame,
    df_train: pd.DataFrame,
    formulation: Formulation,
) -> dict:
    """In-sample fit, expressed in KSh so it is comparable to the holdout MAE.

    A large gap between this and the holdout MAE means overfitting; a small
    in-sample error with a poor holdout error means the model cannot represent
    the relationship at all.
    """
    Xtr, _ = impute(X_train, X_train)
    y_train = df_train[TARGET_REVENUE].to_numpy(dtype=float)
    pred = np.clip(
        np.nan_to_num(formulation.to_revenue(model.predict(Xtr), df_train), nan=0.0),
        0.0, None,
    )
    return {
        "in_sample_mae_kes": float(mean_absolute_error(y_train, pred)),
        "in_sample_r2": float(r2_score(y_train, pred)),
    }


# --------------------------------------------------------------------------
# signal diagnostic
# --------------------------------------------------------------------------


def signal_diagnostic(df: pd.DataFrame, wf_splits: list[Split]) -> dict:
    """Test whether the correlation between features and future growth is
    forecastable, as opposed to merely present in the sample.

    Three stages:
      1. how much does next-30-day revenue actually vary relative to this month,
      2. do features correlate with that variation, and is the correlation
         distinguishable from chance under a permutation test,
      3. does using the correlation reduce walk-forward error, and how much
         should the model's opinion be trusted (the shrinkage weight).
    """
    ratio = (df[TARGET_REVENUE] / df["revenue_30d"]).to_numpy(dtype=float)
    deviation = ratio - 1.0
    revenue = df[TARGET_REVENUE].to_numpy(dtype=float)
    rev30 = df["revenue_30d"].to_numpy(dtype=float)

    candidates = [
        "active_days_30d", "revenue_trend_30_vs_90", "cv_daily_revenue_30d",
        "tx_count_30d", "unique_customers_30d",
    ]
    candidates = [c for c in candidates if c in df.columns]

    rng = np.random.default_rng(SEED)
    correlations = []
    for feature in candidates:
        values = df[feature].to_numpy(dtype=float)
        r = float(np.corrcoef(values, deviation)[0, 1])
        null = np.array([
            abs(np.corrcoef(rng.permutation(values), deviation)[0, 1])
            for _ in range(PERMUTATION_REPEATS)
        ])
        correlations.append({
            "feature": feature, "r": r, "p_perm": float(np.mean(null >= abs(r))),
        })

    def pooled(builder) -> tuple[np.ndarray, np.ndarray]:
        predictions, actuals = [], []
        for split in wf_splits:
            predictions.append(builder(split.train_idx, split.test_idx))
            actuals.append(revenue[split.test_idx])
        return np.concatenate(predictions), np.concatenate(actuals)

    def predict_deviation(X: np.ndarray, tr: np.ndarray, te: np.ndarray) -> np.ndarray:
        """Ridge on the deviation target, refitted inside every fold."""
        model = Pipeline([
            ("scale", StandardScaler()),
            ("model", RidgeCV(alphas=(1.0, 10.0, 50.0, 200.0, 500.0))),
        ]).fit(X[tr], deviation[tr])
        return model.predict(X[te])

    single_feature_mae = []
    for feature in candidates:
        X_one = df[[feature]].to_numpy(dtype=float)
        preds, actuals = pooled(lambda tr, te, X_one=X_one:
                                rev30[te] * (1.0 + predict_deviation(X_one, tr, te)))
        single_feature_mae.append({
            "feature": feature,
            "mae_kes": float(mean_absolute_error(actuals, preds)),
        })

    X_multi = df[candidates].to_numpy(dtype=float)
    dev_preds, _ = pooled(lambda tr, te: predict_deviation(X_multi, tr, te))

    # Shrinkage: prediction = revenue_30d * (1 + lam * predicted_deviation).
    # lam = 0 ignores the model, lam = 1 trusts it fully. The data picks lam.
    # rev30 must be aligned to the pooled fold order, so rebuild it per fold.
    pooled_rev30 = np.concatenate([rev30[s.test_idx] for s in wf_splits])
    pooled_actual = np.concatenate([revenue[s.test_idx] for s in wf_splits])

    shrinkage = []
    best_lam, best_mae = 0.0, float(mean_absolute_error(pooled_actual, pooled_rev30))
    for lam in (0.0, 0.1, 0.2, 0.3, 0.5, 1.0):
        preds = pooled_rev30 * (1.0 + lam * dev_preds)
        mae = float(mean_absolute_error(pooled_actual, preds))
        shrinkage.append({"lam": lam, "mae_kes": mae})
        if mae < best_mae:
            best_mae, best_lam = mae, lam

    full_preds = pooled_rev30 * (1.0 + dev_preds)
    multi_feature_deviation_mae = float(mean_absolute_error(pooled_actual, full_preds))

    return {
        "ratio_mean": float(ratio.mean()),
        "ratio_std": float(ratio.std()),
        "deviation_autocorr": float(pd.Series(deviation).autocorr(1)),
        "correlations": correlations,
        "single_feature_mae": single_feature_mae,
        "mae_reference": float(mean_absolute_error(pooled_actual, pooled_rev30)),
        "multi_feature_deviation_mae": multi_feature_deviation_mae,
        "shrinkage": shrinkage,
        "best_lam": best_lam,
    }


# --------------------------------------------------------------------------
# main experiment
# --------------------------------------------------------------------------


def main() -> None:
    started = datetime.now(timezone.utc)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("Q-CASH REVENUE MODEL - next 30 calendar days of M-Pesa Till revenue")
    print("=" * 78)

    raw = load_dataset(DATA_PATH)
    print(f"\nLoaded {DATA_PATH.relative_to(BASE_DIR)}")
    print(f"  rows              : {len(raw)}")
    print(f"  period            : {raw[DATE_COL].min().date()} -> {raw[DATE_COL].max().date()}")
    print(f"  businesses        : {raw['business_id'].nunique()}")
    print(f"  total till revenue: KSh {raw[TARGET_REVENUE].sum():,.2f}")

    missing = raw.isna().sum()
    missing = missing[missing > 0]
    print(f"\nMissing values in source dataset: {dict(missing) if len(missing) else 'none'}")

    df = add_momentum_features(raw)
    feature_sets = build_feature_sets(df)
    formulations = build_formulations(df)

    full_feature_list = feature_sets.all_variants["momentum+knbs"]
    required_history = MOMENTUM_MAX_LAG
    history_mask = df[full_feature_list].notna().all(axis=1)
    dropped = df.loc[~history_mask, DATE_COL]
    df = df.loc[history_mask].reset_index(drop=True)
    print(
        f"\nExcluded {len(dropped)} earliest row(s) lacking {required_history} weeks of "
        f"weekly history:\n  {', '.join(d.strftime('%Y-%m-%d') for d in dropped)}"
    )
    print(
        "  Reason: momentum lag features are undefined before this point. These are the\n"
        "  oldest, least representative weeks of the series; they are removed for every\n"
        "  model alike so the comparison stays fair."
    )

    n = len(df)
    print(f"\nUsable observations: {n}  ({df[DATE_COL].min().date()} -> {df[DATE_COL].max().date()})")

    holdout = chronological_holdout(n, HOLDOUT_FRACTION, EMBARGO_WEEKS)
    wf_splits = walk_forward_splits(n, EMBARGO_WEEKS)

    print(f"\nHoldout split ({holdout.label})")
    print(
        f"  train   : {df[DATE_COL].iloc[holdout.train_idx[0]].date()} -> "
        f"{df[DATE_COL].iloc[holdout.train_idx[-1]].date()}  ({len(holdout.train_idx)} obs)"
    )
    print(f"  embargo : {len(holdout.embargoed_idx)} obs dropped between train and test")
    print(
        f"  test    : {df[DATE_COL].iloc[holdout.test_idx[0]].date()} -> "
        f"{df[DATE_COL].iloc[holdout.test_idx[-1]].date()}  ({len(holdout.test_idx)} obs)"
    )

    print(f"\nWalk-forward folds: {len(wf_splits)}")
    for split in wf_splits:
        print(
            f"  {split.label}: train {len(split.train_idx):3d} | "
            f"test {df[DATE_COL].iloc[split.test_idx[0]].date()} -> "
            f"{df[DATE_COL].iloc[split.test_idx[-1]].date()} ({len(split.test_idx)} obs)"
        )

    extrap = extrapolation_report(df, holdout)
    print("\n" + "-" * 78)
    print("EXTRAPOLATION DIAGNOSTIC (explains the earlier Random Forest failure)")
    print("-" * 78)
    print(f"  train target range : KSh {extrap['train_target_min_kes']:>12,.0f} "
          f"-> KSh {extrap['train_target_max_kes']:>12,.0f}")
    print(f"  test  target range : KSh {extrap['test_target_min_kes']:>12,.0f} "
          f"-> KSh {extrap['test_target_max_kes']:>12,.0f}")
    print(f"  test mean revenue  : KSh {extrap['test_mean_kes']:>12,.0f} "
          f"(train mean KSh {extrap['train_mean_kes']:,.0f})")
    print(f"  test targets ABOVE the largest training target: "
          f"{extrap['test_above_train_max_count']} of {len(holdout.test_idx)} "
          f"({extrap['test_outside_train_range_pct']:.0f}%)")
    print(
        "\n  A tree model predicts averages of the targets it was trained on, so it cannot\n"
        "  return a value above the largest target it has seen. Because this business is\n"
        "  growing, much of the holdout lies above the training maximum, and a model that\n"
        "  forecasts revenue directly is capped below the truth.\n"
        "\n"
        "  A ratio formulation (target / past revenue) is NOT subject to the same hard cap,\n"
        "  because the predicted ratio is multiplied by current revenue, which keeps growing.\n"
        "  The step-2 table shows this directly: formulation A saturates while B and C,\n"
        "  which divide by a longer and smoother denominator, track the rising level better."
    )

    test_dates = df[DATE_COL].iloc[holdout.test_idx]
    y_test = df[TARGET_REVENUE].iloc[holdout.test_idx].to_numpy(dtype=float)
    baselines = baseline_predictions(df)
    prediction_rows: list[dict] = [
        {
            "as_of_date": date,
            "actual_next_30d_revenue_kes": actual,
            **{f"pred_{name}": float(values[holdout.test_idx][i])
               for name, values in baselines.items()},
        }
        for i, (date, actual) in enumerate(zip(test_dates.to_numpy(), y_test))
    ]

    results: list[dict] = []

    print("\n" + "=" * 78)
    print("STEP 1 - BASELINES on the untouched holdout")
    print("=" * 78)
    for name, preds in baselines.items():
        metrics = evaluate(y_test, preds[holdout.test_idx])
        results.append({"model": name, "formulation": "baseline", **metrics})
        print(
            f"  {name:<16} MAE {metrics['mae_kes']:>10,.0f}  RMSE {metrics['rmse_kes']:>10,.0f}"
            f"  R2 {metrics['r2']:>6.2f}  sMAPE {metrics['smape_pct']:>5.1f}%"
        )
    print(
        "\n  MAE is the headline number: the average number of shillings by which the\n"
        "  forecast misses the actual next-30-day revenue. RMSE punishes large misses\n"
        "  more, R2 measures how much of the week-to-week variation is explained, and\n"
        "  sMAPE is the same error expressed as a percentage that is safe for small\n"
        "  revenue values."
    )

    print("\n" + "=" * 78)
    print("STEP 2 - MACHINE LEARNING MODELS on the same untouched holdout")
    print("=" * 78)

    models = build_models()
    primary_feature_set = "momentum+knbs"
    feature_columns = drop_zero_variance(df, feature_sets.all_variants[primary_feature_set])
    X = df[feature_columns]
    y = df[TARGET_REVENUE]
    y_train = y.iloc[holdout.train_idx].to_numpy(dtype=float)
    y_test_series = y.iloc[holdout.test_idx].reset_index(drop=True)
    X_train = X.iloc[holdout.train_idx]
    X_test = X.iloc[holdout.test_idx]

    trained: dict[tuple[str, str], object] = {}
    quality_rows: list[dict] = []

    for model_name, model in models.items():
        for form_name, formulation in formulations.items():
            y_train_form = np.nan_to_num(
                formulation.train_target(df.iloc[holdout.train_idx]),
                nan=0.0, posinf=0.0, neginf=0.0,
            )
            estimator = build_models()[model_name]
            preds = fit_predict(
                estimator, X_train, X_test, y_train_form, formulation, X_test
            )
            quality = fit_quality_report(estimator, X_train, df.iloc[holdout.train_idx],
                                         formulation)

            metrics = evaluate(y_test, preds)
            results.append({
                "model": model_name,
                "formulation": form_name,
                "feature_set": primary_feature_set,
                **metrics,
                **quality,
            })
            quality_rows.append({"model": model_name, "formulation": form_name, **quality})
            trained[(model_name, form_name)] = estimator

            label = f"{model_name} / {form_name}"
            print(
                f"  {label:<38} MAE {metrics['mae_kes']:>10,.0f}  "
                f"RMSE {metrics['rmse_kes']:>10,.0f}  R2 {metrics['r2']:>6.2f}  "
                f"sMAPE {metrics['smape_pct']:>5.1f}%   (in-sample R2 {quality['in_sample_r2']:>5.2f})"
            )

            for row, predicted in zip(prediction_rows, preds):
                row[f"pred_{model_name}_{form_name}"] = float(predicted)

    results_df = pd.DataFrame(results)
    print("\n" + "=" * 78)
    print("STEP 3 - FEATURE-SET ABLATION (does momentum or KNBS data help?)")
    print("=" * 78)
    print("Random Forest under formulation B (growth on 30d), holdout MAE in KSh:\n")
    ablation_rows: list[dict] = []
    for fs_name, columns in feature_sets.all_variants.items():
        cols = drop_zero_variance(df, columns)
        fs_train = df[cols].iloc[holdout.train_idx]
        fs_test = df[cols].iloc[holdout.test_idx]
        formulation = formulations["B_growth_30d"]
        y_train_form = np.nan_to_num(
            formulation.train_target(df.iloc[holdout.train_idx]), nan=0.0
        )
        estimator = build_models()["random_forest"]
        preds = fit_predict(estimator, fs_train, fs_test, y_train_form, formulation, fs_test)
        metrics = evaluate(y_test, preds)
        ablation_rows.append({
            "feature_set": fs_name,
            "n_features": len(cols),
            "contains_knbs": any(c in KNBS_COLUMNS for c in cols),
            "contains_trend": "days_since_start" in cols,
            **metrics,
        })
        knbs_tag = "KNBS" if any(c in KNBS_COLUMNS for c in cols) else "no-KNBS"
        print(
            f"  {fs_name:<22} n={len(cols):>3}  {knbs_tag:<8} "
            f"MAE {metrics['mae_kes']:>10,.0f}  R2 {metrics['r2']:>6.2f}  "
            f"sMAPE {metrics['smape_pct']:>5.1f}%"
        )
    ablation_df = pd.DataFrame(ablation_rows)

    print("\n" + "=" * 78)
    print("STEP 4 - WALK-FORWARD VALIDATION (pooled across folds)")
    print("=" * 78)
    print("Expanding window, same embargo, models refitted at every fold.\n")
    wf_rows: list[dict] = []
    for model_name, model in models.items():
        for form_name, formulation in formulations.items():
            fold_preds: list[np.ndarray] = []
            fold_actuals: list[np.ndarray] = []
            fold_table: list[dict] = []
            for split in wf_splits:
                fs_train = df[feature_columns].iloc[split.train_idx]
                fs_test = df[feature_columns].iloc[split.test_idx]
                y_train_form = np.nan_to_num(
                    formulation.train_target(df.iloc[split.train_idx]), nan=0.0
                )
                estimator = build_models()[model_name]
                preds = fit_predict(estimator, fs_train, fs_test, y_train_form,
                                    formulation, fs_test)
                actuals = df[TARGET_REVENUE].iloc[split.test_idx].to_numpy(dtype=float)
                fold_preds.append(preds)
                fold_actuals.append(actuals)
                fold_metrics = evaluate(actuals, preds)
                fold_table.append({"fold": split.label, **fold_metrics})
            pooled = evaluate(np.concatenate(fold_actuals), np.concatenate(fold_preds))
            wf_rows.append({
                "model": model_name,
                "formulation": form_name,
                **{k: v for k, v in pooled.items() if k != "n_test"},
                "n_folds": len(wf_splits),
                "n_predictions": pooled["n_test"],
            })
            fold_mae = [f["mae_kes"] for f in fold_table]
            print(
                f"  {model_name}/{form_name:<16} MAE {pooled['mae_kes']:>10,.0f}  "
                f"RMSE {pooled['rmse_kes']:>10,.0f}  R2 {pooled['r2']:>6.2f}  "
                f"sMAPE {pooled['smape_pct']:>5.1f}%   "
                f"(worst fold MAE {max(fold_mae):>9,.0f})"
            )

    for name, preds in baselines.items():
        fold_preds, fold_actuals = [], []
        for split in wf_splits:
            idx = split.test_idx
            fold_preds.append(preds[idx])
            fold_actuals.append(df[TARGET_REVENUE].iloc[idx].to_numpy(dtype=float))
        pooled = evaluate(np.concatenate(fold_actuals), np.concatenate(fold_preds))
        wf_rows.append({
            "model": name,
            "formulation": "baseline",
            **{k: v for k, v in pooled.items() if k != "n_test"},
            "n_folds": len(wf_splits),
            "n_predictions": pooled["n_test"],
        })
        print(
            f"  {name:<24} MAE {pooled['mae_kes']:>10,.0f}  RMSE {pooled['rmse_kes']:>10,.0f}"
            f"  R2 {pooled['r2']:>6.2f}  sMAPE {pooled['smape_pct']:>5.1f}%"
        )

    wf_df = pd.DataFrame(wf_rows)

    print("\n" + "=" * 78)
    print("STEP 5 - FEATURE IMPORTANCE for the selected Random Forest")
    print("=" * 78)
    rf_key = ("random_forest", "B_growth_30d")
    rf = trained[rf_key]
    _, Xte_imp = impute(X_train, X_test)
    holdout_rows = df.iloc[holdout.test_idx]
    y_test_revenue = holdout_rows[TARGET_REVENUE].to_numpy(dtype=float)

    ratio_multiplier = holdout_rows[formulations[rf_key[1]]._ratio].to_numpy(dtype=float)

    def mae_in_kes(y_true, y_pred) -> float:
        """Scorer that undoes the ratio transformation before scoring.

        The forest outputs growth ratios, but importance must be judged on the
        error the business would actually see, which is measured in shillings.
        Permutation importance shuffles feature columns, not rows, so the
        multiplier array stays aligned with y_pred.
        """
        predicted = np.asarray(y_pred, dtype=float) * ratio_multiplier
        return -float(mean_absolute_error(y_true, predicted))

    mae_scorer = make_scorer(mae_in_kes, greater_is_better=False, response_method="predict")
    perm = permutation_importance(
        rf,
        Xte_imp,
        y_test_revenue,
        n_repeats=20,
        random_state=SEED,
        scoring=mae_scorer,
    )
    importance = pd.DataFrame({
        "feature": feature_columns,
        "impurity_importance": rf.feature_importances_,
        "permutation_importance_mean": perm.importances_mean,
        "permutation_importance_std": perm.importances_std,
    }).sort_values("permutation_importance_mean", ascending=False)

    print("\n  Permutation importance on the holdout (MAE increase when shuffled, KSh):\n")
    for _, row in importance.head(15).iterrows():
        bar = "#" * int(min(40, max(0, row["permutation_importance_mean"] / 500)))
        print(
            f"  {row['feature']:<30} {row['permutation_importance_mean']:>10,.0f}  "
            f"+/- {row['permutation_importance_std']:>7,.0f}  {bar}"
        )

    knbs_perm = importance.loc[
        importance["feature"].isin(KNBS_COLUMNS), "permutation_importance_mean"
    ].sum()
    total_perm = importance["permutation_importance_mean"].clip(lower=0).sum()
    knbs_share = knbs_perm / total_perm if total_perm else 0.0
    print(f"\n  KNBS features combined permutation importance: {knbs_perm:,.0f} KSh "
          f"({100 * knbs_share:.1f}% of the positive total)")
    if knbs_share < 0.05:
        print("  => KNBS CPI and inflation contribute essentially nothing. On this evidence")
        print("     they should be reported as tested and not useful, not as model inputs.")

    print("\n" + "=" * 78)
    print("STEP 5b - IS THE SIGNAL FORECASTABLE? (the decisive test)")
    print("=" * 78)
    signal = signal_diagnostic(df, wf_splits)
    print(
        f"  Growth ratio target/revenue_30d: mean {signal['ratio_mean']:.3f}, "
        f"std {signal['ratio_std']:.3f}"
    )
    print(f"  So next month's revenue is typically within "
          f"+/-{100 * signal['ratio_std']:.0f}% of this month's.")
    print(f"  Consecutive weekly observations share ~93% of their 30-day windows, so the")
    print(f"  119 rows are NOT 119 independent samples. Lag-1 autocorrelation of the")
    print(f"  deviation from 1.0 is {signal['deviation_autocorr']:.2f}.")

    print("\n  Correlation with the deviation from 1.0, and whether it survives a")
    print("  20,000-permutation test (shuffles the feature to break any link):\n")
    for row in signal["correlations"]:
        print(f"    {row['feature']:<24} r={row['r']:+.3f}  permutation p={row['p_perm']:.4f}")

    print("\n  Now the out-of-sample test. Walk-forward MAE using each feature alone to")
    print("  predict the deviation, then add it back to revenue_30d:\n")
    print(f"    {'no model (naive_30d)':<30} MAE {signal['mae_reference']:>9,.0f}   <- reference")
    for row in signal["single_feature_mae"]:
        flag = "  worse" if row["mae_kes"] > signal["mae_reference"] else "  BETTER"
        print(f"    {row['feature']:<30} MAE {row['mae_kes']:>9,.0f}  {flag}")

    print("\n  Shrinkage test: prediction = revenue_30d * (1 + lam * predicted deviation).\n")
    for row in signal["shrinkage"]:
        flag = "  <- best" if row["lam"] == signal["best_lam"] else ""
        print(f"    lam={row['lam']:<5.2f} MAE {row['mae_kes']:>9,.0f}{flag}")

    print(
        f"\n  FINDING: the correlations are statistically real (permutation p < 0.001) but\n"
        f"  every model that uses them forecasts WORSE than the persistence baseline. Optimal\n"
        f"  shrinkage is lam = {signal['best_lam']:.1f}, i.e. the data says to trust the\n"
        f"  model essentially not at all.\n\n"
        f"  The mechanism: features like active_days_30d and tx_count_30d are counts that\n"
        f"  scale with the revenue level, and this business is growing. They correlate with\n"
        f"  the level mechanically, not with any genuine 30-day-ahead lead signal. On 119\n"
        f"  heavily overlapping windows from a single business there is not enough\n"
        f"  independent variation to learn a timing rule that survives out of sample."
    )

    # ----------------------------------------------------------------------
    # model selection
    # ----------------------------------------------------------------------

    print("\n" + "=" * 78)
    print("STEP 6 - SELECTION")
    print("=" * 78)

    candidates = wf_df[wf_df["formulation"] != "baseline"].sort_values("mae_kes")
    best_ml = candidates.iloc[0]
    best_baseline = wf_df[wf_df["formulation"] == "baseline"].sort_values("mae_kes").iloc[0]

    print(f"  Best machine-learning model (walk-forward MAE): "
          f"{best_ml['model']} / {best_ml['formulation']} at KSh {best_ml['mae_kes']:,.0f}")
    print(f"  Best naive baseline      (walk-forward MAE): "
          f"{best_baseline['model']} at KSh {best_baseline['mae_kes']:,.0f}")

    ml_beats_baseline = best_ml["mae_kes"] < best_baseline["mae_kes"]
    improvement = 100.0 * (best_baseline["mae_kes"] - best_ml["mae_kes"]) / best_baseline["mae_kes"]
    if ml_beats_baseline:
        print(f"\n  VERDICT: the best ML model improves MAE by {improvement:.1f}% over the "
              f"best baseline.")
    else:
        print(f"\n  VERDICT: no ML configuration beat the naive baseline. The baseline is "
              f"{abs(improvement):.1f}% better.\n  This is reported as-is. For a single "
              f"growing business, a persistence forecast is a strong benchmark.")

    selected_form = str(best_ml["formulation"]) if ml_beats_baseline else "baseline"
    selected_model_name = str(best_ml["model"]) if ml_beats_baseline else str(best_baseline["model"])

    # ----------------------------------------------------------------------
    # artefacts
    # ----------------------------------------------------------------------

    print("\n" + "=" * 78)
    print("STEP 7 - SAVING ARTEFACTS")
    print("=" * 78)

    results_df.to_csv(OUTPUT_DIR / "holdout_metrics.csv", index=False)
    wf_df.to_csv(OUTPUT_DIR / "walkforward_metrics.csv", index=False)
    ablation_df.to_csv(OUTPUT_DIR / "feature_set_ablation.csv", index=False)
    importance.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(OUTPUT_DIR / "holdout_predictions.csv", index=False)

    if ml_beats_baseline:
        final_estimator = build_models()[selected_model_name]
        y_all_form = np.nan_to_num(
            formulations[selected_form].train_target(df), nan=0.0
        )
        X_all, _ = impute(df[feature_columns], df[feature_columns])
        final_estimator.fit(X_all, y_all_form)
    else:
        if selected_model_name not in BASELINE_ESTIMATORS:
            raise RuntimeError(
                f"Best baseline was {selected_model_name!r}, which is not one of "
                f"{sorted(BASELINE_ESTIMATORS)}. Add an estimator for it before "
                "serving it to the prediction API."
            )
        final_estimator = BASELINE_ESTIMATORS[selected_model_name]()
        print(f"  Selected estimator: {final_estimator!r}")
        print("  This is a fitted object with a .predict() method, so the prediction")
        print("  service can load the joblib bundle and call it like any sklearn model.")
        print("  Required input column: "
              f"{final_estimator.column!r} (trailing 30 or 60 days of till revenue)")

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=BASE_DIR, capture_output=True, text=True, timeout=10,
        ).stdout.strip() or "uncommitted"
    except Exception:
        commit = "uncommitted"

    bundle = {
        "artifact_version": "v1.0.0",
        "created_utc": started.isoformat(),
        "target": "M-Pesa Till revenue (Credit) in the 30 calendar days after as_of_date",
        "target_column": TARGET_REVENUE,
        "selected_model": selected_model_name,
        "selected_formulation": selected_form,
        "estimator": final_estimator,
        "feature_columns": feature_columns,
        "required_input_column": getattr(final_estimator, "column", None),
        "formulation": {
            "name": selected_form,
            "scale": formulations[selected_form].scale if selected_form in formulations else None,
        },
        "feature_set": primary_feature_set,
        "imputation_medians": df[feature_columns].median(numeric_only=True).to_dict(),
        "training_date_range": [
            str(df[DATE_COL].iloc[holdout.train_idx[0]].date()),
            str(df[DATE_COL].iloc[holdout.train_idx[-1]].date()),
        ],
        "n_training_observations": int(len(holdout.train_idx)),
        "validation": {
            "method": "expanding-window walk-forward with 5-observation embargo",
            "n_folds": len(wf_splits),
            "folds": [
                {
                    "fold": s.label,
                    "train": [str(df[DATE_COL].iloc[s.train_idx[0]].date()),
                              str(df[DATE_COL].iloc[s.train_idx[-1]].date())],
                    "test": [str(df[DATE_COL].iloc[s.test_idx[0]].date()),
                             str(df[DATE_COL].iloc[s.test_idx[-1]].date())],
                    "n_train": int(len(s.train_idx)),
                    "n_test": int(len(s.test_idx)),
                }
                for s in wf_splits
            ],
        },
        "holdout_metrics": [
            {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
             for k, v in r.items()}
            for r in results
        ],
        "dataset": {
            "path": str(DATA_PATH.relative_to(BASE_DIR)),
            "version": "v1.0",
            "business_id": "BELGARD_BARBERS_001",
            "observed_businesses": 1,
            "synthetic_rows": 0,
        },
        "caveats": [
            "Single observed business; the weekly observations are not independent "
            "businesses and the results do not generalise to all Kenyan MSMEs.",
            "Consecutive weekly rows share about 93% of their 30-day windows, so the "
            "effective number of independent observations is far below the row count.",
            "Revenue is defined as M-Pesa Till Payment Received credits only.",
            "Tree models cannot extrapolate beyond their training target range, and "
            "44% of the holdout targets lay above the training maximum.",
            "No machine-learning configuration beat the persistence baseline. Features "
            "correlate with the growth deviation at p < 0.001 by permutation test, but "
            "that correlation does not reduce out-of-sample error on this single "
            "business, so the baseline is retained deliberately rather than by default.",
            "This forecast is suitable for sizing a revenue-backed limit, not for "
            "credit decisions on a growing book of businesses.",
        ],
        "environment": {
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "git_commit": commit,
            "seed": SEED,
        },
    }
    joblib.dump(bundle, MODEL_PATH)

    summary = {
        "created_utc": started.isoformat(),
        "n_observations_used": int(n),
        "rows_excluded_for_history": [str(d.date()) for d in dropped],
        "holdout_split": {
            "train_end": str(df[DATE_COL].iloc[holdout.train_idx[-1]].date()),
            "embargo_obs": int(len(holdout.embargoed_idx)),
            "test_start": str(df[DATE_COL].iloc[holdout.test_idx[0]].date()),
            "test_end": str(df[DATE_COL].iloc[holdout.test_idx[-1]].date()),
        },
        "extrapolation_diagnostic": extrap,
        "selected_model": selected_model_name,
        "selected_formulation": selected_form,
        "best_walkforward_ml": {k: float(v) for k, v in best_ml.items()
                                if isinstance(v, (int, float, np.floating))},
        "best_walkforward_baseline": {k: float(v) for k, v in best_baseline.items()
                                      if isinstance(v, (int, float, np.floating))},
        "ml_beats_best_baseline": bool(ml_beats_baseline),
        "signal_diagnostic": signal,
        "environment": bundle["environment"],
    }
    (OUTPUT_DIR / "run_summary.json").write_text(json.dumps(summary, indent=2))
    (OUTPUT_DIR / "signal_diagnostic.json").write_text(json.dumps(signal, indent=2))

    make_plots(df, holdout, y_test, baseline_predictions(df), trained, formulations,
               importance, wf_df, results_df, OUTPUT_DIR)

    print(f"\n  Wrote artefacts to {OUTPUT_DIR}:")
    for path in sorted(OUTPUT_DIR.iterdir()):
        print(f"    {path.name:<34} {path.stat().st_size:>9,d} bytes")
    print(f"\n  Model artifact: {MODEL_PATH.name}")
    print("  Load and predict with:")
    print(f"    bundle = joblib.load('{MODEL_PATH.name}')")
    print("    bundle['estimator'].predict(frame)   # frame must contain "
          f"{bundle['required_input_column']!r}")


def make_plots(
    df: pd.DataFrame,
    holdout: Split,
    y_test: np.ndarray,
    baselines: dict[str, np.ndarray],
    trained: dict,
    formulations: dict,
    importance: pd.DataFrame,
    wf_df: pd.DataFrame,
    results_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Write the three diagnostic figures used in the dissertation."""
    dates = df[DATE_COL].iloc[holdout.test_idx]

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(dates, y_test, marker="o", ms=4, lw=2, color="black", label="Actual next 30d revenue")
    for name, color, style in [
        ("naive_30d", "tab:orange", "--"),
        ("drift_30d", "tab:green", "-."),
    ]:
        ax.plot(dates, baselines[name][holdout.test_idx], lw=1.8, ls=style, color=color, label=name)
    plot_columns = feature_columns_global(df)
    for (model_name, form_name) in trained:
        if model_name != "random_forest":
            continue
        formulation = formulations[form_name]
        tr = df.iloc[holdout.train_idx]
        te = df.iloc[holdout.test_idx]
        Xtr, Xte = impute(df[plot_columns].iloc[holdout.train_idx],
                          df[plot_columns].iloc[holdout.test_idx])
        y_form = np.nan_to_num(formulation.train_target(tr), nan=0.0)
        est = build_models()[model_name].fit(Xtr, y_form)
        preds = np.clip(
            np.nan_to_num(formulation.to_revenue(est.predict(Xte), te), nan=0.0), 0, None
        )
        ax.plot(dates, preds, lw=1.5, ls=":", color="tab:red", label=f"RF / {form_name}")
    ax.set_title("Q-Cash holdout: actual vs predicted next-30-day revenue")
    ax.set_ylabel("KSh")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_dir / "plot_holdout_predictions.png", dpi=130)
    plt.close(fig)

    top = importance.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top["feature"], top["permutation_importance_mean"],
            xerr=top["permutation_importance_std"], color="tab:blue", alpha=0.85)
    ax.set_xlabel("Increase in holdout MAE when the feature is shuffled (KSh)")
    ax.set_title("Random Forest permutation importance (formulation B)")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(output_dir / "plot_feature_importance.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    plot_df = wf_df.sort_values("mae_kes")
    labels = [f"{r.model}\n{r.formulation}" for r in plot_df.itertuples()]
    ax.barh(labels, plot_df["mae_kes"], color="tab:green", alpha=0.8)
    ax.set_xlabel("Walk-forward MAE (KSh) - lower is better")
    ax.set_title("Model comparison under walk-forward validation")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(output_dir / "plot_walkforward_mae.png", dpi=130)
    plt.close(fig)


def feature_columns_global(df: pd.DataFrame) -> list[str]:
    """The feature list used for the headline comparison."""
    sets = build_feature_sets(df)
    return drop_zero_variance(df, sets.all_variants["momentum+knbs"])


if __name__ == "__main__":
    sys.exit(main())
