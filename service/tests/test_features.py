"""Tests for feature engineering and revenue prediction."""

from __future__ import annotations

import math
import unittest

import pandas as pd

from app import features as fe
from app import model


def daily_frame(n_days: int = 120, revenue: float = 4000.0, tx_count: int = 10) -> pd.DataFrame:
    dates = pd.date_range("2026-06-01", periods=n_days, freq="D")
    return pd.DataFrame(
        {
            "transaction_date": dates,
            "revenue_kes": [revenue] * n_days,
            "transaction_count": [tx_count] * n_days,
        }
    )


class TestFeatureEngineering(unittest.TestCase):
    def test_compute_features_returns_expected_keys(self):
        row = fe.compute_features(daily_frame())
        for key in ("revenue_7d", "revenue_30d", "revenue_90d", "tx_count_30d",
                    "active_days_30d", "cv_daily_revenue_30d", "revenue_trend_30_vs_90"):
            self.assertIn(key, row)

    def test_constant_revenue_has_zero_cv(self):
        row = fe.compute_features(daily_frame())
        self.assertAlmostEqual(row["cv_daily_revenue_30d"], 0.0, places=6)

    def test_requires_minimum_history(self):
        with self.assertRaises(ValueError):
            fe.compute_features(daily_frame(n_days=30))

    def test_requires_required_columns(self):
        with self.assertRaises(ValueError):
            fe.compute_features(pd.DataFrame({"foo": [1] * 100}))


class TestRevenuePrediction(unittest.TestCase):
    def test_prediction_is_finite_and_positive(self):
        row = fe.compute_features(daily_frame())
        prediction = float(model.predict_revenue(pd.DataFrame([row]))[0])
        self.assertTrue(math.isfinite(prediction))
        self.assertGreater(prediction, 0.0)

    def test_prediction_returns_one_value_per_row(self):
        rows = [fe.compute_features(daily_frame(revenue=1000 * i)) for i in (1, 2, 3)]
        predictions = model.predict_revenue(pd.DataFrame(rows))
        self.assertEqual(len(predictions), 3)


if __name__ == "__main__":
    unittest.main()
