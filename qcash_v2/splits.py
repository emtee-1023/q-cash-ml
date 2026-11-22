"""Grouped-by-business, ordered-by-time splitting for Q-Cash v2.

The split protocol follows the retraining prompt section 5:

  * a business appears in exactly one of train / test_business;
  * within a business the data are also ordered by time;
  * walk-forward (expanding-window) folds select the model;
  * three disjoint test sets are reported separately:
      - test_temporal  : latest months of businesses seen in training
      - test_business  : businesses never seen in training
      - test_tail      : churned / shocked businesses in the test window

The module also implements the four leakage tests the prompt requires.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from common import DATA_DIR, derive_seed, load_config


@dataclass
class Partitions:
    train: pd.DataFrame
    validation: pd.DataFrame
    test_temporal: pd.DataFrame
    test_business: pd.DataFrame
    test_tail: pd.DataFrame
    train_business_ids: np.ndarray
    test_business_ids: np.ndarray

    def all_test(self) -> pd.DataFrame:
        return pd.concat(
            [self.test_temporal, self.test_business], ignore_index=True
        )


def _mark_tail(df: pd.DataFrame) -> pd.DataFrame:
    """Flag rows where the business churned or suffered a large revenue shock.

    A row is a tail case when its 30-day target is less than 75% of the trailing
    30-day revenue (a >25% drop), or the business churned within the target
    window.  A churned business has no future receipts, so its target collapses.
    """
    out = df.copy()
    businesses = pd.read_csv(DATA_DIR / "businesses.csv")
    churn = businesses.set_index("business_id")["churn_date"].dropna()
    churn = churn[churn.astype(bool)]
    churn_ts = churn.map(lambda v: pd.Timestamp(v))

    def churned_within(row: pd.Series) -> bool:
        if row["business_id"] not in churn_ts.index:
            return False
        c = churn_ts[row["business_id"]]
        return (c >= row["as_of_date"]) and (c <= row["as_of_date"] + pd.Timedelta(days=30))

    out["is_tail"] = (
        ((out["target_minor"] < 0.75 * out["rev_sum_30"]) & (out["rev_sum_30"] > 0))
        | out.apply(churned_within, axis=1)
    )
    return out


def build_partitions(supervised: pd.DataFrame) -> Partitions:
    cfg = load_config()
    sp = cfg["splits"]
    seed = cfg["generator"]["master_seed"]
    rng = np.random.default_rng(seed + 55)

    biz_ids = np.sort(supervised["business_id"].unique())
    rng.shuffle(biz_ids)
    n_test = int(len(biz_ids) * sp["test_business_frac"])
    test_biz = biz_ids[:n_test]
    train_biz = biz_ids[n_test:]

    max_date = supervised["as_of_date"].max()
    temporal_test_start = max_date - pd.DateOffset(months=sp["temporal_test_months"])
    val_start = temporal_test_start - pd.DateOffset(months=sp["val_months"])
    embargo = pd.Timedelta(days=sp["embargo_days"])

    in_train_biz = supervised["business_id"].isin(train_biz)
    in_test_biz = supervised["business_id"].isin(test_biz)

    train = supervised[in_train_biz & (supervised["as_of_date"] < val_start - embargo)]
    validation = supervised[
        in_train_biz
        & (supervised["as_of_date"] >= val_start)
        & (supervised["as_of_date"] < temporal_test_start - embargo)
    ]
    test_temporal = supervised[in_train_biz & (supervised["as_of_date"] >= temporal_test_start)]
    test_business = supervised[in_test_biz & (supervised["as_of_date"] >= temporal_test_start)]

    test_temporal = _mark_tail(test_temporal)
    test_business = _mark_tail(test_business)
    test_tail = pd.concat(
        [test_temporal[test_temporal["is_tail"]], test_business[test_business["is_tail"]]],
        ignore_index=True,
    )

    return Partitions(
        train=train.reset_index(drop=True),
        validation=validation.reset_index(drop=True),
        test_temporal=test_temporal.reset_index(drop=True),
        test_business=test_business.reset_index(drop=True),
        test_tail=test_tail.reset_index(drop=True),
        train_business_ids=train_biz,
        test_business_ids=test_biz,
    )


def walk_forward_folds(train: pd.DataFrame) -> list[dict[str, Any]]:
    """Expanding-window walk-forward folds over the training set (by time)."""
    cfg = load_config()
    sp = cfg["splits"]
    n_folds = sp["walk_forward_folds"]
    embargo = pd.Timedelta(days=sp["embargo_days"])

    dates = np.sort(train["as_of_date"].unique())
    folds: list[dict[str, Any]] = []
    # expand the training window; validate on the next ~2 months block
    block = max(2, len(dates) // (n_folds + 1))
    starts = [block * (i + 1) for i in range(n_folds)]
    for i, s in enumerate(starts):
        tr_end = dates[min(s, len(dates) - 1)]
        val_start = tr_end + pd.Timedelta(days=1)
        val_end_idx = min(s + block, len(dates) - 1)
        if val_end_idx <= s:
            continue
        val_end = dates[val_end_idx]
        tr = train[train["as_of_date"] <= tr_end - embargo]
        va = train[
            (train["as_of_date"] >= val_start)
            & (train["as_of_date"] <= val_end)
        ]
        if len(tr) == 0 or len(va) == 0:
            continue
        folds.append({
            "fold": i + 1,
            "train": tr.reset_index(drop=True),
            "val": va.reset_index(drop=True),
            "train_end": pd.Timestamp(tr_end),
            "val_end": pd.Timestamp(val_end),
        })
    return folds


# ---------------------------------------------------------------------------
# leakage tests
# ---------------------------------------------------------------------------


def run_leakage_tests(
    partitions: Partitions, supervised: pd.DataFrame, feature_columns: list[str]
) -> dict[str, bool]:
    results: dict[str, bool] = {}

    # 1. business-identity test: business_id is not a model feature
    results["business_identity"] = "business_id" not in feature_columns

    # 2. overlap test: business ids disjoint; train/test date ranges do not overlap
    train_biz = set(partitions.train_business_ids)
    test_biz = set(partitions.test_business_ids)
    biz_disjoint = len(train_biz & test_biz) == 0
    test_all = partitions.all_test()
    date_overlap = False
    if len(partitions.train) and len(test_all):
        date_overlap = partitions.train["as_of_date"].max() >= test_all["as_of_date"].min()
    results["overlap"] = biz_disjoint and (not date_overlap)

    # 3. every business appears in exactly one business partition
    train_b = set(partitions.train["business_id"]) | set(partitions.validation["business_id"]) | set(
        partitions.test_temporal["business_id"]
    )
    test_b = set(partitions.test_business["business_id"])
    results["business_grouped"] = len(train_b & test_b) == 0

    # 4. no month appears in both train and test_temporal
    train_months = set(partitions.train["as_of_date"].dt.to_period("M"))
    tt_months = set(partitions.test_temporal["as_of_date"].dt.to_period("M"))
    results["no_month_overlap"] = len(train_months & tt_months) == 0

    return results


def manifest(partitions: Partitions) -> dict[str, Any]:
    def describe(name: str, df: pd.DataFrame) -> dict[str, Any]:
        return {
            "n_rows": int(len(df)),
            "n_businesses": int(df["business_id"].nunique()) if len(df) else 0,
            "n_business_months": int(len(df.groupby(["business_id", df["as_of_date"].dt.to_period("M")]))) if len(df) else 0,
            "date_range": [str(df["as_of_date"].min().date()), str(df["as_of_date"].max().date())] if len(df) else [],
        }

    return {
        "train": describe("train", partitions.train),
        "validation": describe("validation", partitions.validation),
        "test_temporal": describe("test_temporal", partitions.test_temporal),
        "test_business": describe("test_business", partitions.test_business),
        "test_tail": describe("test_tail", partitions.test_tail),
        "train_business_ids": [str(b) for b in partitions.train_business_ids],
        "test_business_ids": [str(b) for b in partitions.test_business_ids],
    }


if __name__ == "__main__":
    supervised = pd.read_csv(DATA_DIR / "supervised.csv", parse_dates=["as_of_date"])
    parts = build_partitions(supervised)
    print("partitions:")
    for name, df in [
        ("train", parts.train), ("validation", parts.validation),
        ("test_temporal", parts.test_temporal),
        ("test_business", parts.test_business), ("test_tail", parts.test_tail),
    ]:
        print(f"  {name:<14} rows={len(df):>6}  businesses={df['business_id'].nunique():>4}")
    feature_columns = [c for c in supervised.columns if c not in {
        "business_id", "as_of_date", "sector", "size_class", "target_minor",
        "oracle_forecast_minor", "is_tail",
    }]
    tests = run_leakage_tests(parts, supervised, feature_columns)
    print("leakage tests:", tests)
    print("folds:", len(walk_forward_folds(parts.train)))
