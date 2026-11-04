"""Tests for reliability scoring."""

from __future__ import annotations

import unittest

from app.reliability import ReliabilityScorer
from tests.helpers import default_policy


def analytics(**overrides):
    base = {
        "active_day_rate_30d": 0.9,
        "cv_daily_revenue_30d": 0.2,
        "avg_tx_per_active_day_30d": 15.0,
        "active_days_30d": 28.0,
        "revenue_trend_30_vs_90": 0.1,
        "transaction_concentration_30d": 0.05,
        "history_days": 200.0,
    }
    base.update(overrides)
    return base


class TestReliabilityScoring(unittest.TestCase):
    def test_score_is_between_zero_and_hundred(self):
        scorer = ReliabilityScorer(default_policy())
        result = scorer.score(analytics(), None)
        self.assertGreaterEqual(result.score, 0.0)
        self.assertLessEqual(result.score, 100.0)

    def test_weights_sum_to_one_for_available_components(self):
        scorer = ReliabilityScorer(default_policy())
        result = scorer.score(analytics(), None)
        total = sum(c.weight for c in result.components if c.available)
        self.assertAlmostEqual(total, 0.9, places=6)

    def test_missing_repayment_history_is_not_penalised(self):
        scorer = ReliabilityScorer(default_policy())
        no_history = scorer.score(analytics(), None)
        with_history = scorer.score(analytics(), {"advances_count": 0, "on_time_count": 0})
        self.assertEqual(no_history.score, with_history.score)
        self.assertIn("repayment_history", no_history.unavailable_components)
        self.assertIn("NO_PRIOR_REPAYMENT_HISTORY", no_history.reason_codes)

    def test_good_repayment_history_raises_score(self):
        scorer = ReliabilityScorer(default_policy())
        no_history = scorer.score(analytics(), None)
        good = scorer.score(
            analytics(),
            {"advances_count": 10, "on_time_count": 10, "late_count": 0, "default_count": 0},
        )
        self.assertGreater(good.score, no_history.score)
        self.assertIn("GOOD_REPAYMENT_HISTORY", good.reason_codes)

    def test_poor_repayment_history_lowers_score(self):
        scorer = ReliabilityScorer(default_policy())
        no_history = scorer.score(analytics(), None)
        poor = scorer.score(
            analytics(),
            {"advances_count": 10, "on_time_count": 1, "late_count": 4, "default_count": 5},
        )
        self.assertLess(poor.score, no_history.score)
        self.assertIn("POOR_REPAYMENT_HISTORY", poor.reason_codes)

    def test_volatile_business_scores_lower(self):
        scorer = ReliabilityScorer(default_policy())
        stable = scorer.score(analytics(cv_daily_revenue_30d=0.1), None)
        volatile = scorer.score(analytics(cv_daily_revenue_30d=1.5), None)
        self.assertGreater(stable.score, volatile.score)

    def test_components_are_explainable(self):
        scorer = ReliabilityScorer(default_policy())
        result = scorer.score(analytics(), None)
        self.assertEqual(len(result.components), 7)
        for component in result.components:
            self.assertGreaterEqual(component.normalized_score, 0.0)
            self.assertLessEqual(component.normalized_score, 100.0)

    def test_insufficient_history_missing_code(self):
        scorer = ReliabilityScorer(default_policy())
        result = scorer.score(analytics(history_days=30.0), None)
        self.assertNotIn("SUFFICIENT_TRANSACTION_HISTORY", result.reason_codes)


if __name__ == "__main__":
    unittest.main()
