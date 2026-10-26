"""
test_triage_evaluation.py

Unit tests for the validation triage metrics in src/triage_evaluation.py,
plus a reproducibility check against the frozen baseline artifacts.
"""

import os
import unittest
import joblib
import numpy as np
import pandas as pd

from src.scoring import compute_anomaly_scores
from src.triage_evaluation import (
    align_scores_with_labels,
    bootstrap_average_precision,
    check_manifest_matches,
    evaluate_scores,
    load_validation_data,
    metrics_at_alert_budget,
    metrics_at_fpr_limit,
    precision_recall_data,
    recall_at_fpr,
    validate_scores,
)


class TestAveragePrecisionAndPRCurve(unittest.TestCase):
    def setUp(self):
        # Small example worked by hand:
        # ranking by score: 0.8 (fraud), 0.4 (legit), 0.35 (fraud), 0.1 (legit)
        # AP = 0.5 * 1.0 + 0.5 * (2/3) = 0.8333...
        self.labels = np.array([0, 0, 1, 1])
        self.scores = np.array([0.1, 0.4, 0.35, 0.8])

    def test_average_precision_known_value(self):
        row = evaluate_scores("toy", self.labels, self.scores, n_bootstrap=0)
        self.assertAlmostEqual(row["average_precision"], 0.5 + 0.5 * (2 / 3), places=10)
        self.assertEqual(row["n_samples"], 4)
        self.assertEqual(row["fraud_count"], 2)
        self.assertAlmostEqual(row["fraud_prevalence"], 0.5)

    def test_pr_curve_known_points(self):
        curve = precision_recall_data(self.labels, self.scores)
        np.testing.assert_allclose(curve["precision"], [0.5, 2 / 3, 0.5, 1.0, 1.0])
        np.testing.assert_allclose(curve["recall"], [1.0, 1.0, 0.5, 0.5, 0.0])
        np.testing.assert_allclose(curve["threshold"].iloc[:-1], [0.1, 0.35, 0.4, 0.8])
        # Last point (recall = 0) has no threshold
        self.assertTrue(np.isnan(curve["threshold"].iloc[-1]))

    def test_perfect_ranking_has_ap_one(self):
        labels = np.array([0, 0, 0, 1, 1])
        scores = np.array([1.0, 2.0, 3.0, 10.0, 11.0])
        row = evaluate_scores("perfect", labels, scores, n_bootstrap=0)
        self.assertAlmostEqual(row["average_precision"], 1.0)
        self.assertAlmostEqual(row["recall_at_1pct_fpr"], 1.0)


class TestRecallAtFPR(unittest.TestCase):
    def test_most_permissive_threshold_without_ties(self):
        # 100 legit scores 1..100, 10 fraud scores 95.5..104.5
        labels = np.array([0] * 100 + [1] * 10)
        scores = np.concatenate([np.arange(1, 101, dtype=float), np.arange(95.5, 105.5, 1.0)])

        # 1% FPR -> at most 1 legit flagged (legit 100).
        # Lowering the threshold to the fraud score 99.5 adds no false positive,
        # but going to legit 99 would, so the most permissive threshold is 99.5.
        result = recall_at_fpr(labels, scores, max_fpr=0.01)
        self.assertEqual(result["threshold"], 99.5)
        self.assertEqual(result["false_positives"], 1)
        self.assertEqual(result["true_positives"], 6)  # fraud 99.5 .. 104.5
        self.assertAlmostEqual(result["recall"], 0.6)
        self.assertAlmostEqual(result["actual_fpr"], 0.01)

        # 5% FPR -> at most 5 legit flagged (legit 96..100) -> threshold drops to fraud 95.5
        result = recall_at_fpr(labels, scores, max_fpr=0.05)
        self.assertEqual(result["threshold"], 95.5)
        self.assertEqual(result["false_positives"], 5)
        self.assertEqual(result["true_positives"], 10)
        self.assertAlmostEqual(result["recall"], 1.0)

    def test_tied_scores_are_never_split(self):
        # 10 legit, 2 fraud. Score 10.0 is shared by 2 legit rows and 1 fraud row.
        labels = np.array([0] * 8 + [0, 0, 1] + [1])
        scores = np.array([1, 2, 3, 4, 5, 6, 7, 8] + [10, 10, 10] + [11], dtype=float)

        # Limit 10%: flagging the tied group would give 2/10 = 20% FPR, so the whole tie stays unflagged
        result = recall_at_fpr(labels, scores, max_fpr=0.10)
        self.assertEqual(result["threshold"], 11.0)
        self.assertEqual(result["true_positives"], 1)
        self.assertEqual(result["false_positives"], 0)
        self.assertAlmostEqual(result["recall"], 0.5)
        self.assertAlmostEqual(result["actual_fpr"], 0.0)

        # Limit 20%: the tied group now fits, and all three tied rows are flagged together
        result = recall_at_fpr(labels, scores, max_fpr=0.20)
        self.assertEqual(result["threshold"], 10.0)
        self.assertEqual(result["true_positives"], 2)
        self.assertEqual(result["false_positives"], 2)
        self.assertAlmostEqual(result["recall"], 1.0)

    def test_fpr_never_exceeds_limit_on_random_scores(self):
        rng = np.random.default_rng(0)
        labels = np.array([0] * 2000 + [1] * 20)
        # Rounding creates many ties
        scores = np.round(rng.normal(size=2020) + labels * 2, 1)
        for limit in [0.0, 0.001, 0.01, 0.05, 0.2]:
            result = recall_at_fpr(labels, scores, max_fpr=limit)
            self.assertLessEqual(result["actual_fpr"], limit)
            flagged = scores >= result["threshold"]
            self.assertEqual(result["alerts"], int(flagged.sum()))

    def test_no_threshold_possible_gives_no_alerts(self):
        # Top score is legit, so any alert at all breaks a 0% FPR limit
        labels = np.array([0, 0, 1])
        scores = np.array([1.0, 9.0, 5.0])
        result = recall_at_fpr(labels, scores, max_fpr=0.0)
        self.assertTrue(np.isinf(result["threshold"]))
        self.assertEqual(result["alerts"], 0)
        self.assertEqual(result["recall"], 0.0)


class TestAlertBudget(unittest.TestCase):
    def setUp(self):
        # 8 legit, 4 fraud. Ranking by score (high to low):
        # 20 F, 19 F, 18 L, 17 F, 16 L, 15 L, 14 F, then legit 1..5
        self.labels = np.array([1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0])
        self.scores = np.array([20, 19, 18, 17, 16, 15, 14, 1, 2, 3, 4, 5], dtype=float)

    def test_top_k_counts(self):
        result = metrics_at_alert_budget(self.labels, self.scores, budget=4)
        self.assertEqual(result["alerts"], 4)
        self.assertEqual(result["frauds_caught"], 3)
        self.assertEqual(result["false_positives"], 1)
        self.assertAlmostEqual(result["precision"], 0.75)
        self.assertAlmostEqual(result["recall"], 0.75)
        self.assertAlmostEqual(result["fpr"], 1 / 8)
        self.assertEqual(result["threshold"], 17.0)

    def test_budget_larger_than_dataset_flags_everything(self):
        result = metrics_at_alert_budget(self.labels, self.scores, budget=100)
        self.assertEqual(result["alerts"], 12)
        self.assertAlmostEqual(result["recall"], 1.0)

    def test_tie_at_cutoff_never_exceeds_budget(self):
        # Scores 9, 8, 8, 8, 1: a budget of 2 cuts through the tie at 8
        labels = np.array([1, 1, 0, 0, 0])
        scores = np.array([9.0, 8.0, 8.0, 8.0, 1.0])
        result = metrics_at_alert_budget(labels, scores, budget=2)
        self.assertEqual(result["alerts"], 1)
        self.assertEqual(result["frauds_caught"], 1)
        self.assertEqual(result["threshold"], 9.0)

        # A budget of 4 fits the whole tie
        result = metrics_at_alert_budget(labels, scores, budget=4)
        self.assertEqual(result["alerts"], 4)
        self.assertEqual(result["frauds_caught"], 2)
        self.assertEqual(result["false_positives"], 2)

    def test_invalid_budget_rejected(self):
        with self.assertRaises(ValueError):
            metrics_at_alert_budget(self.labels, self.scores, budget=0)

    def test_fpr_limit_reports_precision_and_volume(self):
        result = metrics_at_fpr_limit(self.labels, self.scores, max_fpr=1 / 8)
        # One legit row allowed: flag 20, 19, 18, 17 (stop before legit 16)
        self.assertEqual(result["alerts"], 4)
        self.assertEqual(result["frauds_caught"], 3)
        self.assertAlmostEqual(result["precision"], 0.75)


class TestInputValidation(unittest.TestCase):
    def test_nan_scores_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_scores("bad", np.array([0, 1, 0]), np.array([1.0, np.nan, 2.0]), n_bootstrap=0)

    def test_infinite_scores_rejected(self):
        with self.assertRaises(ValueError):
            recall_at_fpr(np.array([0, 1, 0]), np.array([1.0, np.inf, 2.0]), max_fpr=0.01)

    def test_length_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_scores("bad", np.array([0, 1, 0]), np.array([1.0, 2.0]), n_bootstrap=0)

    def test_two_dimensional_scores_rejected(self):
        with self.assertRaises(ValueError):
            validate_scores(np.ones((3, 1)), expected_length=3)

    def test_non_binary_labels_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_scores("bad", np.array([0, 2, 1]), np.array([1.0, 2.0, 3.0]), n_bootstrap=0)

    def test_single_class_labels_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_scores("bad", np.array([0, 0, 0]), np.array([1.0, 2.0, 3.0]), n_bootstrap=0)


class TestScoreLabelAlignment(unittest.TestCase):
    def setUp(self):
        self.row_ids = np.array([10, 11, 12, 13])
        self.labels = np.array([0, 1, 0, 1])
        self.scores = np.array([1.0, 9.0, 2.0, 8.0])

    def test_shuffled_score_rows_are_realigned_by_row_id(self):
        score_df = pd.DataFrame({"row_id": self.row_ids, "anomaly_score": self.scores})
        shuffled = score_df.sample(frac=1.0, random_state=3).reset_index(drop=True)
        # Make sure the shuffle actually changed the order
        self.assertFalse(np.array_equal(shuffled["row_id"].to_numpy(), self.row_ids))

        aligned = align_scores_with_labels(shuffled, self.row_ids, self.labels)
        np.testing.assert_array_equal(aligned["row_id"], self.row_ids)
        np.testing.assert_array_equal(aligned["anomaly_score"], self.scores)
        np.testing.assert_array_equal(aligned["Class"], self.labels)

    def test_missing_row_id_rejected(self):
        score_df = pd.DataFrame({"row_id": [10, 11, 12], "anomaly_score": [1.0, 9.0, 2.0]})
        with self.assertRaises(ValueError):
            align_scores_with_labels(score_df, self.row_ids, self.labels)

    def test_extra_row_id_rejected(self):
        score_df = pd.DataFrame({"row_id": [10, 11, 12, 13, 99], "anomaly_score": [1.0, 9.0, 2.0, 8.0, 5.0]})
        with self.assertRaises(ValueError):
            align_scores_with_labels(score_df, self.row_ids, self.labels)

    def test_duplicate_row_id_rejected(self):
        score_df = pd.DataFrame({"row_id": [10, 11, 11, 13], "anomaly_score": [1.0, 9.0, 2.0, 8.0]})
        with self.assertRaises(ValueError):
            align_scores_with_labels(score_df, self.row_ids, self.labels)


class TestReproducibility(unittest.TestCase):
    def test_evaluation_is_deterministic(self):
        rng = np.random.default_rng(1)
        labels = np.array([0] * 500 + [1] * 10)
        scores = rng.normal(size=510) + labels * 3
        first = evaluate_scores("m", labels, scores, n_bootstrap=50)
        second = evaluate_scores("m", labels, scores, n_bootstrap=50)
        self.assertEqual(first, second)

    def test_bootstrap_interval_contains_point_estimate(self):
        rng = np.random.default_rng(2)
        labels = np.array([0] * 500 + [1] * 20)
        scores = rng.normal(size=520) + labels * 2
        row = evaluate_scores("m", labels, scores, n_bootstrap=200)
        self.assertLessEqual(row["ap_ci_low"], row["average_precision"])
        self.assertGreaterEqual(row["ap_ci_high"], row["average_precision"])
        self.assertEqual(
            bootstrap_average_precision(labels, scores, n_resamples=200),
            bootstrap_average_precision(labels, scores, n_resamples=200),
        )


@unittest.skipUnless(os.path.exists("models/gmm_baseline.joblib"), "Frozen baseline artifacts not available")
class TestFrozenBaselineArtifacts(unittest.TestCase):
    """Uses validation data only. Test labels are never loaded."""

    @classmethod
    def setUpClass(cls):
        cls.data = load_validation_data("data/processed")
        cls.gmm = joblib.load("models/gmm_baseline.joblib")
        cls.scores = compute_anomaly_scores(cls.gmm, cls.data["X_val"])

    def test_validation_rows_match_manifest(self):
        n_rows = check_manifest_matches(self.data["row_ids_val"], "data/splits/split_manifest.csv")
        self.assertEqual(n_rows, 56957)
        self.assertEqual(int(self.data["y_val"].sum()), 57)

    def test_rescored_baseline_matches_frozen_score_file(self):
        frozen = pd.read_csv("data/processed/scores/gmm_baseline_scores_val.csv")
        aligned = align_scores_with_labels(frozen, self.data["row_ids_val"], self.data["y_val"])
        np.testing.assert_allclose(aligned["anomaly_score"].to_numpy(), self.scores, rtol=0, atol=1e-8)

    @unittest.skipUnless(
        os.path.exists("results/tables/triage_model_comparison.csv"),
        "Run scripts/run_triage_evaluation.py first",
    )
    def test_generated_baseline_row_is_reproducible(self):
        table = pd.read_csv("results/tables/triage_model_comparison.csv")
        saved_row = table[table["model_name"] == "gmm_baseline_k1_full"].iloc[0]
        new_row = evaluate_scores("gmm_baseline_k1_full", self.data["y_val"], self.scores)
        for column in ["average_precision", "recall_at_1pct_fpr", "recall_at_5pct_fpr",
                       "threshold_1pct_fpr", "threshold_5pct_fpr", "ap_ci_low", "ap_ci_high"]:
            self.assertAlmostEqual(float(saved_row[column]), new_row[column], places=9, msg=column)


if __name__ == "__main__":
    unittest.main()
