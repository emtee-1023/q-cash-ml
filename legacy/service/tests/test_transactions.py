"""Tests for transaction validation, classification and anomaly flagging."""

from __future__ import annotations

import unittest

import pandas as pd

from app import transactions as txn
from app.policy import load_policy
from tests.helpers import make_tx, default_policy


class TestValidation(unittest.TestCase):
    def test_valid_transaction_has_no_issues(self):
        self.assertEqual(txn.validate_transaction(make_tx(100.0)), [])

    def test_missing_amount(self):
        record = make_tx(100.0)
        del record["amount"]
        self.assertIn("missing_amount", txn.validate_transaction(record))

    def test_non_numeric_amount(self):
        record = make_tx(100.0)
        record["amount"] = "not-a-number"
        self.assertIn("non_numeric_amount", txn.validate_transaction(record))

    def test_zero_amount(self):
        self.assertIn("zero_amount", txn.validate_transaction(make_tx(0.0)))

    def test_negative_amount_is_debit(self):
        config = default_policy().classification
        result = txn.classify_one(make_tx(-500.0), config)
        self.assertEqual(result.direction, "debit")
        self.assertFalse(result.is_eligible_revenue)

    def test_invalid_direction(self):
        record = make_tx(100.0, direction="sideways")
        self.assertIn("invalid_direction", txn.validate_transaction(record))


class TestClassification(unittest.TestCase):
    def setUp(self):
        self.config = default_policy().classification

    def test_sale_keyword_is_business_sale(self):
        result = txn.classify_one(make_tx(100.0, transaction_type="till_payment_received"), self.config)
        self.assertEqual(result.category, txn.TransactionCategory.BUSINESS_SALE)
        self.assertTrue(result.is_eligible_revenue)

    def test_refund_is_not_revenue(self):
        result = txn.classify_one(make_tx(100.0, transaction_type="customer_refund"), self.config)
        self.assertEqual(result.category, txn.TransactionCategory.REFUND)
        self.assertFalse(result.is_eligible_revenue)

    def test_reversal_is_not_revenue(self):
        result = txn.classify_one(make_tx(100.0, transaction_type="reversal"), self.config)
        self.assertEqual(result.category, txn.TransactionCategory.REVERSAL)
        self.assertFalse(result.is_eligible_revenue)

    def test_capital_injection_is_not_revenue(self):
        result = txn.classify_one(make_tx(100000.0, transaction_type="owner_capital_injection"), self.config)
        self.assertEqual(result.category, txn.TransactionCategory.CAPITAL_INJECTION)
        self.assertFalse(result.is_eligible_revenue)

    def test_internal_transfer_is_not_revenue(self):
        result = txn.classify_one(make_tx(100.0, transaction_type="internal_transfer"), self.config)
        self.assertEqual(result.category, txn.TransactionCategory.INTERNAL_TRANSFER)
        self.assertFalse(result.is_eligible_revenue)

    def test_unknown_when_no_information(self):
        result = txn.classify_one(make_tx(100.0, transaction_type=None), self.config)
        self.assertEqual(result.category, txn.TransactionCategory.UNKNOWN)
        self.assertFalse(result.is_eligible_revenue)

    def test_failed_status_is_reversal(self):
        result = txn.classify_one(make_tx(100.0, status="failed"), self.config)
        self.assertEqual(result.category, txn.TransactionCategory.REVERSAL)
        self.assertFalse(result.is_eligible_revenue)

    def test_caller_supplied_category_wins(self):
        result = txn.classify_one(make_tx(100.0, category="refund"), self.config)
        self.assertEqual(result.category, txn.TransactionCategory.REFUND)


class TestAnomalyFlagging(unittest.TestCase):
    def setUp(self):
        self.config = default_policy().classification

    def test_unusual_large_inflow_flagged_but_not_eligible_if_unknown(self):
        records = [
            txn.classify_one(make_tx(100.0, transaction_id="T001"), self.config),
            txn.classify_one(make_tx(100.0, transaction_id="T002"), self.config),
            txn.classify_one(make_tx(100.0, transaction_id="T003"), self.config),
            txn.classify_one(make_tx(100.0, transaction_id="T004"), self.config),
            txn.classify_one(make_tx(100.0, transaction_id="T005"), self.config),
            txn.classify_one(make_tx(100.0, transaction_id="T006"), self.config),
            txn.classify_one(make_tx(100.0, transaction_id="T007"), self.config),
            txn.classify_one(make_tx(100.0, transaction_id="T008"), self.config),
            txn.classify_one(make_tx(100.0, transaction_id="T009"), self.config),
            txn.classify_one(make_tx(100.0, transaction_id="T010"), self.config),
            txn.classify_one(make_tx(1_000_000.0, transaction_id="T011", transaction_type=None), self.config),
        ]
        txn.detect_anomalies(records, self.config, pd.Timestamp("2026-10-01"))
        large = records[-1]
        self.assertEqual(large.category, txn.TransactionCategory.UNKNOWN)
        self.assertTrue(large.is_unusual)
        self.assertFalse(large.is_eligible_revenue)

    def test_genuine_large_sale_remains_eligible(self):
        records = [
            txn.classify_one(make_tx(100.0, transaction_id=f"T{i:03d}"), self.config)
            for i in range(10)
        ]
        big = txn.classify_one(make_tx(50_000.0, transaction_id="T100"), self.config)
        records.append(big)
        txn.detect_anomalies(records, self.config, pd.Timestamp("2026-10-01"))
        self.assertEqual(big.category, txn.TransactionCategory.BUSINESS_SALE)
        self.assertTrue(big.is_eligible_revenue)
        self.assertTrue(big.is_unusual)
        self.assertEqual(big.unusual_reason, "unusually_large_amount")

    def test_duplicate_id_flagged(self):
        records = [
            txn.classify_one(make_tx(100.0, transaction_id="T001"), self.config),
            txn.classify_one(make_tx(100.0, transaction_id="T001"), self.config),
        ]
        txn.detect_anomalies(records, self.config, pd.Timestamp("2026-10-01"))
        self.assertFalse(records[0].is_unusual)
        self.assertTrue(records[1].is_unusual)
        self.assertEqual(records[1].unusual_reason, "duplicate_transaction_id")


class TestEligibleDailyAndAnalytics(unittest.TestCase):
    def test_eligible_revenue_excludes_non_revenue(self):
        config = default_policy().classification
        records = [
            txn.classify_one(make_tx(100.0, transaction_id="T001"), config),
            txn.classify_one(make_tx(200.0, transaction_id="T002"), config),
            txn.classify_one(make_tx(500.0, transaction_id="T003", transaction_type="refund"), config),
            txn.classify_one(make_tx(1000.0, transaction_id="T004", transaction_type="owner_capital_injection"), config),
        ]
        daily = txn.build_eligible_daily(records)
        self.assertEqual(float(daily["revenue_kes"].sum()), 300.0)
        self.assertEqual(int(daily["transaction_count"].sum()), 2)

    def test_concentration_is_one_for_single_transaction(self):
        config = default_policy().classification
        records = [txn.classify_one(make_tx(100.0, transaction_id="T001"), config)]
        hhi = txn.transaction_concentration(records, pd.Timestamp("2026-10-01"))
        self.assertAlmostEqual(hhi, 1.0)

    def test_concentration_lower_with_many_transactions(self):
        config = default_policy().classification
        records = [
            txn.classify_one(make_tx(10.0, transaction_id=f"T{i:03d}"), config)
            for i in range(10)
        ]
        hhi = txn.transaction_concentration(records, pd.Timestamp("2026-10-01"))
        self.assertLess(hhi, 1.0)


if __name__ == "__main__":
    unittest.main()
