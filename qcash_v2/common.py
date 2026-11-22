"""Shared helpers for the Q-Cash v2 pipeline.

Money is handled as integer MINOR UNITS everywhere (1 KSh = 100 minor units).
No float is used to represent money at rest; floats appear only inside
transient statistical computations (means, standard deviations) and are
converted back to integers before any money value is stored or reported.

Rates are integer basis points (1% = 100 bps).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"
OUTPUTS_DIR = ROOT / "outputs"
PLOTS_DIR = ROOT / "plots"

KSH_PER_MINOR = 100.0  # 1 KSh = 100 minor units


def load_config() -> dict[str, Any]:
    with open(ROOT / "config.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def derive_seed(master_seed: int, index: int) -> int:
    """Deterministic per-entity seed from the master seed and a 0-based index."""
    return int(master_seed + 1_000_003 * (index + 1))


def minor_to_ksh(minor: int | float | np.ndarray) -> float | np.ndarray:
    return np.asarray(minor, dtype=float) / KSH_PER_MINOR


def ksh_to_minor(ksh: float) -> int:
    return int(round(float(ksh) * KSH_PER_MINOR))


def round_down_minor(minor: int, step_minor: int = 5000) -> int:
    """Round an amount DOWN to the nearest ``step_minor`` (default KSh 50)."""
    return int(minor // step_minor * step_minor)


# ---------------------------------------------------------------------------
# metrics (minor-unit inputs; reported in KSh where applicable)
# ---------------------------------------------------------------------------


def smape(actual: np.ndarray, predicted: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    denom = (np.abs(actual) + np.abs(predicted)) / 2.0
    ratio = np.where(denom == 0, 0.0, np.abs(actual - predicted) / denom)
    return float(np.mean(ratio) * 100.0)


def mase(actual: np.ndarray, predicted: np.ndarray, naive_actual: np.ndarray) -> float:
    """Mean absolute scaled error vs the seasonal-naive baseline."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    naive_actual = np.asarray(naive_actual, dtype=float)
    denom = np.mean(np.abs(actual - naive_actual))
    if denom == 0:
        return float("nan")
    return float(np.mean(np.abs(actual - predicted)) / denom)


def evaluate(actual: np.ndarray, predicted: np.ndarray, naive: np.ndarray | None = None) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    err = actual - predicted
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err * err)))
    r2 = float(1.0 - np.sum(err * err) / np.sum((actual - actual.mean()) ** 2)) if actual.size > 1 else float("nan")
    medae = float(np.median(np.abs(err)))
    p90ae = float(np.percentile(np.abs(err), 90))
    within20 = float(np.mean(np.abs(err) <= 0.20 * np.abs(actual)))
    within50 = float(np.mean(np.abs(err) <= 0.50 * np.abs(actual)))
    mean_actual = float(np.mean(np.abs(actual))) if actual.size else 0.0
    median_actual = float(np.median(np.abs(actual))) if actual.size else 0.0
    result: dict[str, float] = {
        "mae_minor": mae,
        "rmse_minor": rmse,
        "r2": r2,
        "median_ae_minor": medae,
        "p90_ae_minor": p90ae,
        "smape_pct": smape(actual, predicted),
        "within_20pct": within20,
        "within_50pct": within50,
        "mean_actual_minor": mean_actual,
        "median_actual_minor": median_actual,
        # MAE expressed as a percentage of typical monthly revenue
        "mae_pct_of_mean": (mae / mean_actual * 100.0) if mean_actual > 0 else float("nan"),
        "mae_pct_of_median": (mae / median_actual * 100.0) if median_actual > 0 else float("nan"),
        "n": float(actual.size),
    }
    if naive is not None:
        result["mase"] = mase(actual, predicted, naive)
    return result


def minor_metrics_to_ksh(metrics: dict[str, float]) -> dict[str, float]:
    """Duplicate a metrics dict, converting the *_minor values to KSh."""
    out: dict[str, float] = {}
    for key, value in metrics.items():
        if key.endswith("_minor") and isinstance(value, (int, float)):
            out[key[:-6] + "_ksh"] = float(np.asarray(value, dtype=float) / KSH_PER_MINOR)
        else:
            out[key] = value
    return out
