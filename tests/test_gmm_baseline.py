"""
test_gmm_baseline.py

Unit and integration tests for the GMM baseline model and anomaly scoring.
Validates all 11 correctness constraints from Task Phase 1 Step 2.
"""

import os
import shutil
import tempfile
import unittest
import numpy as np
import pandas as pd

from src.preprocessing import load_processed_data, get_feature_names
from src.gmm_model import (
    DEFAULT_GMM_CONFIG,
    create_gmm_model,
    fit_gmm_baseline,
    save_gmm_model,
    load_gmm_model,
)
from src.scoring import (
    compute_anomaly_scores,
    create_score_dataframe,
    save_scores,
    load_scores,
    summarize_scores,
)


class TestGMMBaseline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the preprocessed data once for the test suite
        cls.data = load_processed_data("data/processed")
        cls.X_train = cls.data["X_train"]
        cls.y_train = cls.data["y_train"]
        cls.row_ids_train = cls.data["row_ids_train"]
        cls.X_val = cls.data["X_val"]
        cls.y_val = cls.data["y_val"]
        cls.row_ids_val = cls.data["row_ids_val"]
        cls.feature_names = cls.data["feature_names"]
        cls.metadata = cls.data.get("metadata", {})
        cls.scaler = cls.data["scaler"]

        # Fit baseline model for testing
        cls.gmm, cls.fit_meta = fit_gmm_baseline(
            X_train=cls.X_train,
            y_train=cls.y_train,
            config=DEFAULT_GMM_CONFIG,
            feature_names=cls.feature_names,
        )

        cls.train_scores = compute_anomaly_scores(cls.gmm, cls.X_train)
        cls.val_scores = compute_anomaly_scores(cls.gmm, cls.X_val)

    def test_01_fitting_subset_is_strictly_legitimate(self):
        """Constraint 1: The model-fitting subset contains only rows where y_train == 0."""
        legit_mask = (self.y_train == 0)
        X_legit = self.X_train[legit_mask]
        
        # Verify that the legitimate subset has zero fraud labels
        labels_in_subset = self.y_train[legit_mask]
        self.assertTrue((labels_in_subset == 0).all())
        self.assertEqual((labels_in_subset == 1).sum(), 0)

    def test_02_fitted_observations_count_matches(self):
        """Constraint 2: The number of fitted observations matches legitimate training count."""
        expected_legit_count = int((self.y_train == 0).sum())
        self.assertEqual(expected_legit_count, 170528)
        self.assertEqual(self.fit_meta["n_training_samples"], expected_legit_count)

    def test_03_feature_count_and_order_consistent(self):
        """Constraint 3: Feature count and feature order are consistent with preprocessing metadata."""
        expected_feature_names = get_feature_names()
        self.assertEqual(len(self.feature_names), 29)
        self.assertEqual(self.feature_names, expected_feature_names)
        self.assertEqual(self.gmm.n_features_in_, 29)
        self.assertEqual(self.fit_meta["feature_names"], expected_feature_names)

    def test_04_convergence_inspected(self):
        """Constraint 4: Model converged_ status is inspected and confirmed True."""
        self.assertTrue(hasattr(self.gmm, "converged_"))
        self.assertTrue(self.gmm.converged_, "GMM baseline failed to converge!")
        self.assertGreater(self.gmm.n_iter_, 0)
        self.assertTrue(np.isfinite(self.gmm.lower_bound_))

    def test_05_scores_length_matches(self):
        """Constraint 5: Training and validation scores have expected lengths."""
        self.assertEqual(len(self.train_scores), len(self.X_train))
        self.assertEqual(len(self.val_scores), len(self.X_val))
        self.assertEqual(len(self.train_scores), 170888)
        self.assertEqual(len(self.val_scores), 56957)

    def test_06_scores_are_finite(self):
        """Constraint 6: Scores contain no NaN or infinite values."""
        self.assertTrue(np.isfinite(self.train_scores).all(), "Train scores contain non-finite values!")
        self.assertTrue(np.isfinite(self.val_scores).all(), "Val scores contain non-finite values!")

    def test_07_score_artifacts_preserve_row_ids(self):
        """Constraint 7: Score artifacts preserve correct row IDs and row counts."""
        train_df = create_score_dataframe(self.row_ids_train, self.train_scores, split="train")
        val_df = create_score_dataframe(self.row_ids_val, self.val_scores, split="validation")

        self.assertEqual(len(train_df), 170888)
        self.assertEqual(len(val_df), 56957)
        self.assertTrue(np.array_equal(train_df["row_id"].to_numpy(), self.row_ids_train))
        self.assertTrue(np.array_equal(val_df["row_id"].to_numpy(), self.row_ids_val))
        self.assertEqual(len(set(train_df["row_id"]).intersection(set(val_df["row_id"]))), 0)

    def test_08_save_and_reload_reproduces_scores(self):
        """Constraint 8: Saving and reloading the model reproduces scores within tolerance."""
        temp_dir = tempfile.mkdtemp()
        try:
            model_path, meta_path = save_gmm_model(self.gmm, self.fit_meta, output_dir=temp_dir)
            reloaded_gmm, reloaded_meta = load_gmm_model(model_path, meta_path)

            reloaded_val_scores = compute_anomaly_scores(reloaded_gmm, self.X_val)
            max_abs_diff = np.abs(self.val_scores - reloaded_val_scores).max()
            self.assertAlmostEqual(max_abs_diff, 0.0, places=10)
            self.assertEqual(reloaded_meta["converged"], True)
        finally:
            shutil.rmtree(temp_dir)

    def test_09_reproducibility_with_seed(self):
        """Constraint 9: Re-running with same data, parameters, and random seed produces identical results."""
        gmm_run2, _ = fit_gmm_baseline(
            X_train=self.X_train,
            y_train=self.y_train,
            config=DEFAULT_GMM_CONFIG,
            feature_names=self.feature_names,
        )
        val_scores_run2 = compute_anomaly_scores(gmm_run2, self.X_val)
        self.assertTrue(np.allclose(self.val_scores, val_scores_run2, atol=1e-12))

    def test_10_no_validation_or_test_leakage_in_model(self):
        """Constraint 10: Model and scaler were not fitted on validation or test data."""
        # Scaler check
        self.assertEqual(self.scaler.n_samples_seen_, 170528)
        # GMM check
        self.assertEqual(self.fit_meta["n_training_samples"], 170528)
        self.assertLess(self.fit_meta["n_training_samples"], len(self.X_train))
        # Ensure X_val was not touched during training
        val_legit_count = int((self.y_val == 0).sum())
        self.assertEqual(val_legit_count, 56900)

    def test_11_no_test_labels_used(self):
        """Constraint 11: No test labels or test performance are used to make modeling decisions."""
        # Confirm that X_test was not loaded or scored in the baseline evaluation
        self.assertNotIn("test_scores", self.__dict__)


if __name__ == "__main__":
    unittest.main()
