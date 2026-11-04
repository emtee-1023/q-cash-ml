"""Tests for the hybrid repayment recommendation."""

from __future__ import annotations

import unittest

from app.repayment import RepaymentRecommender
from tests.helpers import default_policy


class TestRepaymentRecommendation(unittest.TestCase):
    def test_base_recovery_percentage(self):
        recommender = RepaymentRecommender(default_policy())
        result = recommender.recommend({"cv_daily_revenue_30d": 0.2, "revenue_trend_30_vs_90": 0.0})
        self.assertAlmostEqual(result.recovery_percentage, 10.0)
        self.assertIn("FIXED_AGREED_RATE", result.reason_codes)

    def test_high_volatility_increases_recovery(self):
        recommender = RepaymentRecommender(default_policy())
        result = recommender.recommend({"cv_daily_revenue_30d": 0.8, "revenue_trend_30_vs_90": 0.0})
        self.assertGreater(result.recovery_percentage, 10.0)
        self.assertIn("VOLATILITY_ADJUSTMENT", result.reason_codes)
        self.assertIn("HIGH_REVENUE_VOLATILITY", result.risk_flags)

    def test_declining_revenue_increases_recovery(self):
        recommender = RepaymentRecommender(default_policy())
        result = recommender.recommend({"cv_daily_revenue_30d": 0.2, "revenue_trend_30_vs_90": -0.4})
        self.assertGreater(result.recovery_percentage, 10.0)
        self.assertIn("DECLINE_ADJUSTMENT", result.reason_codes)

    def test_recovery_is_clamped_to_maximum(self):
        recommender = RepaymentRecommender(default_policy())
        result = recommender.recommend({"cv_daily_revenue_30d": 5.0, "revenue_trend_30_vs_90": -1.0})
        self.assertLessEqual(result.recovery_percentage, 20.0)

    def test_maximum_repayment_days_from_policy(self):
        recommender = RepaymentRecommender(default_policy())
        result = recommender.recommend({"cv_daily_revenue_30d": 0.2, "revenue_trend_30_vs_90": 0.0})
        self.assertEqual(result.maximum_repayment_days, 30)
        self.assertTrue(result.fallback_description)


if __name__ == "__main__":
    unittest.main()
