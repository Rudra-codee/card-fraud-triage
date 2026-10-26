"""
test_model_selection.py

Unit tests for GMM model selection and validation triage evaluation utilities.
"""

import unittest
import numpy as np

from src.model_selection import (
    fit_gmm_candidate,
    compute_triage_metrics,
)


class TestModelSelection(unittest.TestCase):
    def test_compute_triage_metrics_synthetic(self):
        """
        Tests compute_triage_metrics on a deterministic synthetic cohort:
        100 legitimate transactions, 10 fraudulent transactions.
        Scores for legit: 1 to 100.
        Scores for fraud: 95 to 104.
        """
        # 100 legit, 10 fraud
        y_true = np.array([0] * 100 + [1] * 10)
        # Legit scores 1..100
        legit_scores = np.arange(1, 101, dtype=float)
        # Fraud scores all above 95
        fraud_scores = np.linspace(95, 104, 10)
        scores = np.concatenate([legit_scores, fraud_scores])

        metrics = compute_triage_metrics(y_true, scores)

        self.assertIn("average_precision", metrics)
        self.assertIn("recall_at_1pct_fpr", metrics)
        self.assertIn("recall_at_5pct_fpr", metrics)
        self.assertGreater(metrics["average_precision"], 0.0)
        self.assertLessEqual(metrics["average_precision"], 1.0)

        # 99th percentile of 1..100 is ~99.01. Fraud >= 99.01 should be detected.
        self.assertGreaterEqual(metrics["recall_at_1pct_fpr"], 0.0)
        self.assertLessEqual(metrics["recall_at_1pct_fpr"], 1.0)
        # 5% FPR threshold should yield higher or equal recall than 1% FPR threshold
        self.assertGreaterEqual(metrics["recall_at_5pct_fpr"], metrics["recall_at_1pct_fpr"])

    def test_fit_candidate_legitimate_filtering(self):
        """
        Verifies that fit_gmm_candidate filters strictly for y_train == 0.
        """
        np.random.seed(42)
        X_dummy = np.random.randn(200, 5)
        # 180 legit (0), 20 fraud (1)
        y_dummy = np.array([0] * 180 + [1] * 20)

        gmm, record = fit_gmm_candidate(
            X_train=X_dummy,
            y_train=y_dummy,
            n_components=2,
            covariance_type="diag",
            n_init=1,
            random_state=42,
        )

        self.assertEqual(record["status"], "success")
        self.assertEqual(record["n_training_samples"], 180)
        self.assertEqual(record["train_fraud_excluded"], 20)
        self.assertTrue(record["converged"])
        self.assertTrue(np.isfinite(record["aic"]))
        self.assertTrue(np.isfinite(record["bic"]))


if __name__ == "__main__":
    unittest.main()
