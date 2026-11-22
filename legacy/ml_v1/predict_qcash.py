"""Smoke test for the saved Q-Cash model artifact.

Checks that the joblib bundle produced by train_qcash.py can be loaded in a
fresh process and produces a usable forecast, and re-verifies the headline
metrics by recomputing them from the saved predictions.

Run:  python predict_qcash.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "outputs" / "qcash_revenue_model_v1.joblib"
PREDICTIONS_PATH = BASE_DIR / "outputs" / "holdout_predictions.csv"
METRICS_PATH = BASE_DIR / "outputs" / "holdout_metrics.csv"
SUMMARY_PATH = BASE_DIR / "outputs" / "run_summary.json"

failures: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}" + (f"  ({detail})" if detail else ""))
    if not condition:
        failures.append(label)


def main() -> int:
    print("Q-Cash model artifact smoke test\n")

    bundle = joblib.load(MODEL_PATH)
    print(f"Artifact version : {bundle['artifact_version']}")
    print(f"Selected model   : {bundle['selected_model']} / {bundle['selected_formulation']}")
    print(f"Target           : {bundle['target']}")
    print(f"Estimator        : {bundle['estimator']!r}")
    print(f"Training period  : {bundle['training_date_range'][0]} -> "
          f"{bundle['training_date_range'][1]} ({bundle['n_training_observations']} obs)")
    print()

    required = ["artifact_version", "estimator", "feature_columns", "target_column",
                "training_date_range", "validation", "dataset", "environment",
                "holdout_metrics", "caveats", "required_input_column"]
    for key in required:
        check(f"bundle contains {key!r}", key in bundle)

    check("dataset declares observed_businesses == 1",
          bundle["dataset"]["observed_businesses"] == 1)
    check("dataset declares synthetic_rows == 0",
          bundle["dataset"]["synthetic_rows"] == 0)
    check("validation used walk-forward, not shuffling",
          "walk-forward" in bundle["validation"]["method"],
          bundle["validation"]["method"])
    check("an embargo gap is recorded",
          all("embargo" in bundle["validation"]["method"] for _ in [0]))
    check("caveats mention the single-business limitation",
          any("Single observed business" in c for c in bundle["caveats"]))

    print("\nEstimator produces a forecast from the declared input column")
    column = bundle["required_input_column"]
    if column:
        sample = pd.DataFrame({column: [150000.0, 180000.0, 210000.0]})
        preds = np.asarray(bundle["estimator"].predict(sample), dtype=float)
        print(f"  input {column} = {sample[column].tolist()}")
        print(f"  predicted next-30d revenue = {[round(p) for p in preds]}")
        check("returns one prediction per input row", len(preds) == 3)
        check("predictions are finite and non-negative",
              bool(np.all(np.isfinite(preds)) and np.all(preds >= 0)))
        ratio = preds / sample[column].to_numpy()
        print(f"  implied ratio to input = {[round(r, 3) for r in ratio]}")
    else:
        check("required_input_column is set for the selected estimator", False,
              "artifact does not declare which column to feed the estimator")

    print("\nRe-verifying the headline metrics from saved predictions")
    preds_df = pd.read_csv(PREDICTIONS_PATH, parse_dates=["as_of_date"])
    metrics_df = pd.read_csv(METRICS_PATH)
    summary = json.loads(SUMMARY_PATH.read_text())
    check("predictions file has one row per distinct as_of_date",
          len(preds_df) == preds_df["as_of_date"].nunique(),
          f"{len(preds_df)} rows, {preds_df['as_of_date'].nunique()} dates")
    check("no missing baseline predictions on the holdout rows",
          not preds_df["pred_naive_60d"].isna().any())
    actual = preds_df["actual_next_30d_revenue_kes"].to_numpy(dtype=float)
    baseline_col = "pred_naive_60d" if "pred_naive_60d" in preds_df.columns else None
    if baseline_col is None:
        check("holdout_predictions.csv contains a baseline prediction column", False)
    else:
        predicted = preds_df[baseline_col].to_numpy(dtype=float)
        mae = float(np.mean(np.abs(actual - predicted)))
        reported = float(
            metrics_df.loc[
                (metrics_df["model"] == "naive_60d")
                & (metrics_df["formulation"] == "baseline"),
                "mae_kes",
            ].iloc[0]
        )
        wf_baseline = summary["best_walkforward_baseline"]["mae_kes"]
        print(f"  recomputed holdout MAE of naive_60d ({len(actual)} weeks): KSh {mae:,.0f}")
        print(f"  same number recorded in holdout_metrics.csv        : KSh {reported:,.0f}")
        print(f"  best pooled walk-forward baseline MAE in summary   : KSh {wf_baseline:,.0f}")
        check("recomputed holdout MAE matches holdout_metrics.csv",
              abs(mae - reported) < 1.0, f"KSh {mae:,.0f} vs {reported:,.0f}")
        check("walk-forward MAE is in a plausible range for this business",
              10000 < wf_baseline < 40000, f"KSh {wf_baseline:,.0f}")

    check("summary reports that ML did not beat the baseline",
          summary["ml_beats_best_baseline"] is False)
    check("run recorded a signal diagnostic",
          "signal_diagnostic" in summary and "best_lam" in summary["signal_diagnostic"],
          f"optimal shrinkage lam = {summary.get('signal_diagnostic', {}).get('best_lam')}")

    print()
    if failures:
        print(f"{len(failures)} CHECK(S) FAILED:")
        for item in failures:
            print(f"  - {item}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
