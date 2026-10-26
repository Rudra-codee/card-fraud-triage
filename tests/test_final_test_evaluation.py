"""
test_final_test_evaluation.py

Targeted regression tests for the final held-out test evaluation workflow (Member 4, Task A).

Verifies research-integrity invariants:
1. Missing, duplicate, or mismatched row IDs are strictly caught.
2. Score and label alignment is robust to row shuffling and matches by row_id.
3. Non-finite anomaly scores are rejected.
4. Frozen thresholds are applied directly without being recalculated from test labels.
5. Test FPR is strictly computed with legitimate test rows as denominator.
6. Alert-budget triage calculations respect the requested number of alerts and use deterministic tie handling.
7. No model-fitting (fit/refit) or scaler-fitting routines are invoked.
8. Existing validation evaluation functions remain unchanged and backward-compatible.
"""

import os
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from src.triage_evaluation import (
    align_scores_with_labels,
    check_test_manifest_matches,
    evaluate_at_frozen_threshold,
    evaluate_scores,
    evaluate_test_scores,
    load_test_data,
    metrics_at_alert_budget,
    validate_labels,
    validate_scores,
)
from src.scoring import compute_anomaly_scores, create_score_dataframe


class TestRowAlignmentAndIntegrity(unittest.TestCase):
    def setUp(self):
        self.row_ids = np.array([101, 102, 103, 104, 105])
        self.labels = np.array([0, 0, 1, 0, 1])
        self.scores = np.array([1.5, 2.3, 8.9, 0.4, 9.1])
        self.score_df = pd.DataFrame({"row_id": self.row_ids, "anomaly_score": self.scores})

    def test_score_label_alignment_after_row_shuffling(self):
        """Joining scores to labels must match by row_id, regardless of row ordering."""
        shuffled_df = self.score_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        aligned = align_scores_with_labels(shuffled_df, self.row_ids, self.labels)

        np.testing.assert_array_equal(aligned["row_id"].to_numpy(), self.row_ids)
        np.testing.assert_array_equal(aligned["Class"].to_numpy(), self.labels)
        np.testing.assert_allclose(aligned["anomaly_score"].to_numpy(), self.scores)

    def test_missing_row_ids_detected(self):
        """Scores missing any labelled row ID must raise a ValueError."""
        incomplete_df = self.score_df.iloc[:4]  # missing row 105
        with self.assertRaises(ValueError) as ctx:
            align_scores_with_labels(incomplete_df, self.row_ids, self.labels)
        self.assertIn("have no score", str(ctx.exception))

    def test_extra_row_ids_detected(self):
        """Scores containing extraneous row IDs not in the labelled set must raise a ValueError."""
        extra_df = pd.concat([
            self.score_df,
            pd.DataFrame({"row_id": [999], "anomaly_score": [3.0]}),
        ], ignore_index=True)
        with self.assertRaises(ValueError) as ctx:
            align_scores_with_labels(extra_df, self.row_ids, self.labels)
        self.assertIn("not in the labelled set", str(ctx.exception))

    def test_duplicate_row_ids_detected(self):
        """Duplicate row IDs in either labels or scores must raise a ValueError."""
        dup_score_df = pd.concat([
            self.score_df,
            pd.DataFrame({"row_id": [101], "anomaly_score": [2.0]}),
        ], ignore_index=True)
        with self.assertRaises(ValueError) as ctx:
            align_scores_with_labels(dup_score_df, self.row_ids, self.labels)
        self.assertIn("duplicated row IDs", str(ctx.exception))

        dup_row_ids = np.array([101, 101, 103, 104, 105])
        with self.assertRaises(ValueError) as ctx:
            align_scores_with_labels(self.score_df, dup_row_ids, self.labels)
        self.assertIn("Label row IDs contain duplicates", str(ctx.exception))


class TestScoreValidation(unittest.TestCase):
    def test_non_finite_scores_rejected(self):
        """Scores containing NaN, +inf, or -inf must be rejected."""
        with self.assertRaises(ValueError) as ctx:
            validate_scores(np.array([1.0, np.nan, 3.0]), expected_length=3)
        self.assertIn("NaN or infinite", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            validate_scores(np.array([1.0, np.inf, 3.0]), expected_length=3)
        self.assertIn("NaN or infinite", str(ctx.exception))

    def test_length_mismatch_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            validate_scores(np.array([1.0, 2.0]), expected_length=3)
        self.assertIn("does not match expected length", str(ctx.exception))


class TestFrozenThresholdIntegrity(unittest.TestCase):
    def setUp(self):
        # 10 legit rows, 2 fraud rows
        self.labels = np.array([0] * 10 + [1] * 2)
        self.scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 8.5, 12.0])

    def test_frozen_threshold_applied_without_recalculation(self):
        """Frozen threshold must be applied strictly as score >= threshold without tuning."""
        frozen_th = 8.2
        res = evaluate_at_frozen_threshold(self.labels, self.scores, frozen_th)

        # Flagged should be scores >= 8.2: 9.0 (legit), 10.0 (legit), 8.5 (fraud), 12.0 (fraud)
        # Total alerts = 4, TP = 2, FP = 2
        self.assertEqual(res["threshold"], 8.2)
        self.assertEqual(res["alerts"], 4)
        self.assertEqual(res["true_positives"], 2)
        self.assertEqual(res["false_positives"], 2)
        self.assertAlmostEqual(res["recall"], 1.0)
        self.assertAlmostEqual(res["actual_fpr"], 2 / 10)  # 0.20
        self.assertAlmostEqual(res["precision"], 0.50)

    def test_test_fpr_computed_using_legitimate_rows_only(self):
        """FPR denominator must be strictly the count of legitimate rows (labels == 0)."""
        # 4 legit rows (0..3), 1 fraud row (4)
        labels = np.array([0, 0, 0, 0, 1])
        scores = np.array([10.0, 20.0, 30.0, 40.0, 50.0])

        res = evaluate_at_frozen_threshold(labels, scores, threshold=35.0)
        # Flagged: score 40.0 (legit) and 50.0 (fraud)
        # FP = 1. Legitimate rows = 4. Total rows = 5.
        self.assertEqual(res["false_positives"], 1)
        self.assertEqual(res["total_legit"], 4)
        self.assertAlmostEqual(res["actual_fpr"], 1 / 4)  # 0.25, NOT 1/5 = 0.20

    def test_evaluate_test_scores_preserves_thresholds(self):
        """evaluate_test_scores preserves frozen thresholds and records actual test FPRs."""
        row = evaluate_test_scores(
            model_name="test_model",
            labels=self.labels,
            scores=self.scores,
            threshold_1pct=9.5,
            threshold_5pct=7.5,
            n_bootstrap=0,
        )
        self.assertEqual(row["threshold_1pct_fpr"], 9.5)
        self.assertEqual(row["threshold_5pct_fpr"], 7.5)
        self.assertAlmostEqual(row["actual_fpr_1pct"], 1 / 10)  # score 10.0 is FP
        self.assertAlmostEqual(row["recall_at_1pct_fpr"], 1 / 2)  # score 12.0 is TP


class TestAlertBudgetCapacity(unittest.TestCase):
    def test_alert_budget_respects_requested_count(self):
        labels = np.array([0] * 90 + [1] * 10)
        scores = np.linspace(0, 100, 100)  # all unique scores

        for budget in [10, 25, 50]:
            res = metrics_at_alert_budget(labels, scores, budget)
            self.assertEqual(res["alerts"], budget)
            self.assertEqual(res["budget"], budget)

    def test_alert_budget_deterministic_tie_policy(self):
        """If tied scores at cutoff exceed the budget, the whole tie is dropped."""
        labels = np.array([0, 0, 0, 1, 1])
        scores = np.array([1.0, 2.0, 5.0, 5.0, 10.0])

        # Budget = 2: scores are [10.0, 5.0, 5.0, 2.0, 1.0].
        # Top 1 is 10.0. Scores at rank 2 and 3 are tied at 5.0.
        # Including both tied scores gives 3 alerts (> budget 2), so tied group is dropped.
        res = metrics_at_alert_budget(labels, scores, budget=2)
        self.assertEqual(res["alerts"], 1)
        self.assertEqual(res["frauds_caught"], 1)


class TestNoModelFitting(unittest.TestCase):
    def test_evaluation_functions_do_not_call_fit(self):
        """Test evaluation code must never call model.fit() or scaler.fit()."""
        with patch.object(GaussianMixture, "fit") as mock_gmm_fit, \
             patch.object(StandardScaler, "fit") as mock_scaler_fit:

            labels = np.array([0] * 10 + [1] * 2)
            scores = np.array([1.0] * 10 + [2.0] * 2)

            evaluate_test_scores("m", labels, scores, 1.5, 1.2, n_bootstrap=0)
            metrics_at_alert_budget(labels, scores, 5)

            mock_gmm_fit.assert_not_called()
            mock_scaler_fit.assert_not_called()


class TestExistingValidationEvaluation(unittest.TestCase):
    def test_existing_evaluate_scores_unchanged(self):
        """Confirms that evaluate_scores computes validation FPR and recall identically."""
        labels = np.array([0, 0, 1, 1])
        scores = np.array([0.1, 0.4, 0.35, 0.8])
        row = evaluate_scores("ref_model", labels, scores, n_bootstrap=0)
        self.assertIn("recall_at_1pct_fpr", row)
        self.assertIn("recall_at_5pct_fpr", row)
        self.assertIn("average_precision", row)
        self.assertAlmostEqual(row["average_precision"], 0.5 + 0.5 * (2 / 3))


if __name__ == "__main__":
    unittest.main()
