"""Tests for the assessment service and API request/response models."""

from __future__ import annotations

import unittest

from pydantic import ValidationError

from app.assessment import AssessmentService
from app.schemas import AssessmentRequest, AssessmentResponse, TransactionRecord


def make_transactions(n_days: int = 120, per_day: int = 10, amount: float = 400.0):
    import pandas as pd

    dates = pd.date_range("2026-06-01", periods=n_days, freq="D")
    records = []
    counter = 0
    for day in range(n_days):
        date = f"{dates[day].date()}T09:00:00"
        for _ in range(per_day):
            records.append(
                {
                    "transaction_id": f"T{counter:06d}",
                    "transaction_date": date,
                    "amount": amount,
                    "direction": "credit",
                    "status": "completed",
                    "transaction_type": "till_payment_received",
                }
            )
            counter += 1
    return records


class TestRequestValidation(unittest.TestCase):
    def test_valid_request(self):
        request = AssessmentRequest(
            business_id="BUS-001",
            transactions=[TransactionRecord(transaction_date="2026-09-30T10:00:00", amount=100.0)],
        )
        self.assertEqual(request.business_id, "BUS-001")

    def test_missing_business_id_rejected(self):
        with self.assertRaises(ValidationError):
            AssessmentRequest(transactions=[])

    def test_negative_direction_rejected(self):
        with self.assertRaises(ValidationError):
            TransactionRecord(transaction_date="2026-09-30T10:00:00", amount=100.0, direction="sideways")

    def test_non_numeric_amount_rejected(self):
        with self.assertRaises(ValidationError):
            TransactionRecord(transaction_date="2026-09-30T10:00:00", amount="abc")


class TestAssessmentService(unittest.TestCase):
    def test_cold_start_returns_insufficient_data(self):
        service = AssessmentService()
        result = service.assess(
            business_id="BUS-001",
            transactions=make_transactions(n_days=10),
            assessment_reference="ASSESSMENT-COLD",
        )
        self.assertEqual(result["assessment_status"], "insufficient_data")
        self.assertIsNone(result["predicted_revenue"])

    def test_cold_start_with_profile_is_eligible(self):
        service = AssessmentService()
        result = service.assess(
            business_id="BUS-001",
            transactions=make_transactions(n_days=10),
            assessment_reference="ASSESSMENT-COLD-PROFILE",
            business_profile={"cohort": "micro_retail"},
        )
        self.assertEqual(result["assessment_status"], "eligible")
        self.assertTrue(result["recommended_advance_amount"] > 0)

    def test_full_assessment_returns_valid_response(self):
        service = AssessmentService()
        result = service.assess(
            business_id="BUS-001",
            transactions=make_transactions(),
            assessment_reference="ASSESSMENT-002",
        )
        self.assertEqual(result["assessment_status"], "eligible")
        self.assertIsNotNone(result["predicted_revenue"])
        self.assertIsNotNone(result["model_version"])
        self.assertEqual(result["decision_policy_version"], "era-policy-v1")
        self.assertEqual(result["forecast_period_days"], 30)
        self.assertEqual(result["currency"], "KES")
        AssessmentResponse(**result)

    def test_empty_transactions_is_insufficient_data(self):
        service = AssessmentService()
        result = service.assess(business_id="BUS-001", transactions=[])
        self.assertEqual(result["assessment_status"], "insufficient_data")
        AssessmentResponse(**result)

    def test_timezone_aware_transactions_are_supported(self):
        service = AssessmentService()
        records = make_transactions(n_days=10)
        for record in records:
            record["transaction_date"] = record["transaction_date"].replace("T", "T") + "+00:00"
        result = service.assess(
            business_id="BUS-001",
            transactions=records,
            assessment_date="2026-06-15T00:00:00+00:00",
            business_profile={"cohort": "micro_retail"},
        )
        AssessmentResponse(**result)


if __name__ == "__main__":
    unittest.main()
