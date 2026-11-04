"""Tests for the ERA decision engine."""

from __future__ import annotations

import unittest

import pandas as pd

from app.decision_engine import DecisionEngine
from app.reliability import ReliabilityResult
from app.transactions import ClassifiedTransaction, TransactionAnalysis
from tests.helpers import default_policy, policy


def reliability(score: float) -> ReliabilityResult:
    return ReliabilityResult(score=score, band="medium", components=[], reason_codes=[], unavailable_components=[])


def analysis(
    n_records: int = 100,
    anomaly_count: int = 0,
    unknown_count: int = 0,
    history_days: float = 200.0,
    n_transactions: float = 100.0,
) -> TransactionAnalysis:
    records = [
        ClassifiedTransaction(
            raw={}, transaction_id=None, transaction_date=pd.Timestamp("2026-09-30"),
            amount=1.0, direction="credit", category=None, confidence="high",
            reason="", is_eligible_revenue=True,
        )
        for _ in range(n_records)
    ]
    return TransactionAnalysis(
        records=records,
        category_counts={},
        anomaly_count=anomaly_count,
        unknown_count=unknown_count,
        eligible_revenue_total=0.0,
        eligible_transaction_count=n_records,
        eligible_daily=pd.DataFrame(),
        analytics={"history_days": history_days, "n_transactions": n_transactions},
        warnings=[],
    )


class TestAdvanceCalculation(unittest.TestCase):
    def test_advance_is_revenue_times_rate(self):
        engine = DecisionEngine(default_policy())
        result = engine.decide(
            predicted_revenue=100000.0,
            reliability=reliability(80.0),
            analytics={"history_days": 200.0, "n_transactions": 100.0},
            analysis=analysis(),
            exposure=None,
            business_profile=None,
        )
        self.assertEqual(result.status, "eligible")
        self.assertAlmostEqual(result.recommended_advance_rate, 0.35)
        self.assertAlmostEqual(result.recommended_advance_amount, 35000.0)

    def test_max_advance_cap(self):
        engine = DecisionEngine(default_policy())
        result = engine.decide(
            predicted_revenue=10_000_000.0,
            reliability=reliability(80.0),
            analytics={"history_days": 200.0, "n_transactions": 100.0},
            analysis=analysis(),
            exposure=None,
            business_profile=None,
        )
        self.assertLessEqual(result.recommended_advance_amount, 100000.0)


class TestEligibility(unittest.TestCase):
    def test_low_reliability_score_is_ineligible(self):
        engine = DecisionEngine(default_policy())
        result = engine.decide(
            predicted_revenue=100000.0,
            reliability=reliability(20.0),
            analytics={"history_days": 200.0, "n_transactions": 100.0},
            analysis=analysis(),
            exposure=None,
            business_profile=None,
        )
        self.assertEqual(result.status, "ineligible")
        self.assertEqual(result.recommended_advance_amount, 0.0)
        self.assertIn("LOW_RELIABILITY_SCORE", result.reason_codes)

    def test_no_revenue_signal_is_ineligible(self):
        engine = DecisionEngine(default_policy())
        result = engine.decide(
            predicted_revenue=0.0,
            reliability=reliability(80.0),
            analytics={"history_days": 200.0, "n_transactions": 100.0},
            analysis=analysis(),
            exposure=None,
            business_profile=None,
        )
        self.assertEqual(result.status, "ineligible")
        self.assertIn("NO_REVENUE_SIGNAL", result.reason_codes)


class TestExposure(unittest.TestCase):
    def test_exposure_headroom_caps_advance(self):
        engine = DecisionEngine(default_policy())
        result = engine.decide(
            predicted_revenue=100000.0,
            reliability=reliability(80.0),
            analytics={"history_days": 200.0, "n_transactions": 100.0},
            analysis=analysis(),
            exposure={"outstanding_principal": 90000.0, "max_exposure_kes": 100000.0},
            business_profile=None,
        )
        self.assertEqual(result.status, "eligible")
        self.assertLessEqual(result.recommended_advance_amount, 10000.0)

    def test_exposure_limit_reached_is_ineligible(self):
        engine = DecisionEngine(default_policy())
        result = engine.decide(
            predicted_revenue=100000.0,
            reliability=reliability(80.0),
            analytics={"history_days": 200.0, "n_transactions": 100.0},
            analysis=analysis(),
            exposure={"outstanding_principal": 100000.0, "max_exposure_kes": 100000.0},
            business_profile=None,
        )
        self.assertEqual(result.status, "ineligible")
        self.assertIn("EXPOSURE_LIMIT_REACHED", result.reason_codes)


class TestDataQuality(unittest.TestCase):
    def test_high_unknown_ratio_triggers_manual_review(self):
        engine = DecisionEngine(default_policy())
        result = engine.decide(
            predicted_revenue=100000.0,
            reliability=reliability(80.0),
            analytics={"history_days": 200.0, "n_transactions": 100.0},
            analysis=analysis(n_records=100, unknown_count=60),
            exposure=None,
            business_profile=None,
        )
        self.assertEqual(result.status, "manual_review")

    def test_high_anomaly_ratio_triggers_manual_review(self):
        engine = DecisionEngine(default_policy())
        result = engine.decide(
            predicted_revenue=100000.0,
            reliability=reliability(80.0),
            analytics={"history_days": 200.0, "n_transactions": 100.0},
            analysis=analysis(n_records=100, anomaly_count=20),
            exposure=None,
            business_profile=None,
        )
        self.assertEqual(result.status, "manual_review")


class TestInsufficientData(unittest.TestCase):
    def test_insufficient_history_without_profile_is_insufficient_data(self):
        engine = DecisionEngine(default_policy())
        result = engine.decide(
            predicted_revenue=None,
            reliability=reliability(0.0),
            analytics={"history_days": 10.0, "n_transactions": 5.0},
            analysis=analysis(n_records=5, history_days=10.0, n_transactions=5.0),
            exposure=None,
            business_profile=None,
        )
        self.assertEqual(result.status, "insufficient_data")
        self.assertEqual(result.recommended_advance_amount, 0.0)

    def test_cold_start_with_profile_gives_conservative_amount(self):
        engine = DecisionEngine(default_policy())
        result = engine.decide(
            predicted_revenue=None,
            reliability=reliability(0.0),
            analytics={"history_days": 10.0, "n_transactions": 5.0},
            analysis=analysis(n_records=5, history_days=10.0, n_transactions=5.0),
            exposure=None,
            business_profile={"cohort": "micro_retail", "sector": "retail"},
        )
        self.assertEqual(result.status, "eligible")
        self.assertTrue(result.cold_start)
        self.assertAlmostEqual(result.recommended_advance_amount, 5000.0)
        self.assertIn("COLD_START_CONSERVATIVE_ASSESSMENT", result.reason_codes)

    def test_cold_start_disabled(self):
        p = default_policy()
        p.advance.cold_start.enabled = False
        engine = DecisionEngine(p)
        result = engine.decide(
            predicted_revenue=None,
            reliability=reliability(0.0),
            analytics={"history_days": 10.0, "n_transactions": 5.0},
            analysis=analysis(n_records=5, history_days=10.0, n_transactions=5.0),
            exposure=None,
            business_profile={"cohort": "micro_retail"},
        )
        self.assertEqual(result.status, "insufficient_data")

    def test_too_few_transactions_is_insufficient_data(self):
        engine = DecisionEngine(default_policy())
        result = engine.decide(
            predicted_revenue=100000.0,
            reliability=reliability(80.0),
            analytics={"history_days": 200.0, "n_transactions": 5.0},
            analysis=analysis(n_records=5, history_days=200.0, n_transactions=5.0),
            exposure=None,
            business_profile=None,
        )
        self.assertEqual(result.status, "insufficient_data")
        self.assertIn("INSUFFICIENT_TRANSACTIONS", result.reason_codes)


if __name__ == "__main__":
    unittest.main()
