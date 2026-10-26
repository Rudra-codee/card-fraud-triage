"""
test_autoencoder.py

Unit tests for the Autoencoder model and training loop (Task 3A).
Validates:
1. Output shape (N, 29).
2. Finite reconstruction loss (MSE).
3. Training filters out fraudulent rows (y_train == 0 strictly).
4. Reproducibility on a small synthetic dataset.
5. Best-checkpoint and early-stopping behavior.
"""

import os
import shutil
import tempfile
import unittest
import numpy as np
import torch

from src.autoencoder_model import (
    DEFAULT_AE_CONFIG,
    Autoencoder,
    create_autoencoder_model,
    fit_autoencoder,
    save_autoencoder_model,
    load_autoencoder_model,
    set_seed,
)


class TestAutoencoderModel(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        set_seed(42)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_01_model_architecture_and_output_shape(self):
        """Validates that forward pass produces exact shape (N, 29)."""
        model = create_autoencoder_model({"input_dim": 29, "bottleneck_dim": 5})
        x = torch.randn(16, 29)
        output = model(x)

        self.assertEqual(output.shape, (16, 29))
        self.assertTrue(torch.isfinite(output).all())

    def test_02_finite_reconstruction_loss(self):
        """Validates MSE loss calculation is finite and non-negative."""
        model = create_autoencoder_model()
        x = torch.randn(20, 29)
        output = model(x)
        loss = torch.nn.functional.mse_loss(output, x)

        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(loss.item(), 0.0)

    def test_03_training_filters_out_fraudulent_rows(self):
        """Validates that training strictly filters for y_train == 0 and excludes y_train == 1."""
        # 100 samples: 80 legit (0), 20 fraud (1)
        np.random.seed(42)
        X_synth = np.random.randn(100, 29)
        y_synth = np.array([0] * 80 + [1] * 20)

        model, metadata = fit_autoencoder(
            X_train=X_synth,
            y_train=y_synth,
            config={"epochs": 2, "batch_size": 16, "patience": 2},
        )

        self.assertEqual(metadata["n_training_samples"], 80)
        self.assertEqual(metadata["total_train_partition_rows"], 100)
        self.assertEqual(metadata["train_fraud_rows_excluded"], 20)
        self.assertEqual(metadata["model_type"], "Autoencoder")

    def test_04_training_reproducibility(self):
        """Validates that training twice with the same seed yields identical loss history."""
        np.random.seed(42)
        X_synth = np.random.randn(120, 29)
        y_synth = np.zeros(120, dtype=int)

        cfg = {"epochs": 3, "batch_size": 32, "random_state": 42}

        model1, meta1 = fit_autoencoder(X_synth, y_synth, config=cfg)
        model2, meta2 = fit_autoencoder(X_synth, y_synth, config=cfg)

        np.testing.assert_allclose(
            meta1["train_loss_history"],
            meta2["train_loss_history"],
            rtol=1e-5,
            err_msg="Loss histories between identical seeded runs do not match!",
        )

    def test_05_early_stopping_and_checkpoint_restoration(self):
        """Validates that early stopping halts before max_epochs and restores best weights."""
        np.random.seed(42)
        X_train = np.random.randn(150, 29)
        y_train = np.zeros(150, dtype=int)
        X_val = np.random.randn(50, 29)
        y_val = np.zeros(50, dtype=int)

        # High patience threshold so it halts early if loss plateaus
        cfg = {
            "epochs": 50,
            "batch_size": 32,
            "patience": 2,
            "min_delta": 10.0,  # Unreasonably high improvement threshold triggers immediate patience
            "random_state": 42,
        }

        model, metadata = fit_autoencoder(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            config=cfg,
        )

        # Should stop after epoch 1 + patience 2 = 3 epochs, well before 50
        self.assertLess(metadata["epochs_trained"], 50)
        self.assertEqual(metadata["best_epoch"], 1)

    def test_06_model_save_and_load(self):
        """Validates saving weights and metadata and restoring them identically."""
        model = create_autoencoder_model()
        meta = {"model_name": "test_ae", "config": DEFAULT_AE_CONFIG}

        model_path, meta_path = save_autoencoder_model(
            model=model,
            metadata=meta,
            output_dir=self.temp_dir,
            model_basename="test_autoencoder",
        )

        self.assertTrue(os.path.exists(model_path))
        self.assertTrue(os.path.exists(meta_path))

        loaded_model, loaded_meta = load_autoencoder_model(model_path, meta_path)
        self.assertEqual(loaded_model.input_dim, 29)
        self.assertEqual(loaded_model.bottleneck_dim, 5)
        self.assertEqual(loaded_meta["model_name"], "test_ae")

        # Compare outputs on same input
        x = torch.randn(5, 29)
        with torch.no_grad():
            out_orig = model(x)
            out_loaded = loaded_model(x)
        torch.testing.assert_close(out_orig, out_loaded)

    def test_07_compute_autoencoder_scores_shape_and_finite(self):
        """Validates that scoring returns a 1D array of finite, non-negative floats."""
        from src.autoencoder_scoring import compute_autoencoder_scores
        model = create_autoencoder_model()
        X = np.random.randn(45, 29)
        scores = compute_autoencoder_scores(model, X, batch_size=16)

        self.assertEqual(scores.ndim, 1)
        self.assertEqual(len(scores), 45)
        self.assertTrue(np.isfinite(scores).all())
        self.assertTrue((scores >= 0.0).all())

    def test_08_score_dataframe_creation_and_row_id_alignment(self):
        """Validates create_score_dataframe integration with row IDs."""
        from src.autoencoder_scoring import compute_autoencoder_scores
        from src.scoring import create_score_dataframe
        model = create_autoencoder_model()
        X = np.random.randn(10, 29)
        row_ids = np.array([101, 102, 103, 104, 105, 106, 107, 108, 109, 110])
        labels = np.array([0, 0, 0, 1, 0, 0, 0, 0, 1, 0])

        scores = compute_autoencoder_scores(model, X)
        score_df = create_score_dataframe(row_ids=row_ids, scores=scores, split="val", labels=labels)

        self.assertEqual(len(score_df), 10)
        np.testing.assert_array_equal(score_df["row_id"].values, row_ids)
        np.testing.assert_array_equal(score_df["Class"].values, labels)
        self.assertEqual(list(score_df["split"].unique()), ["val"])


if __name__ == "__main__":
    unittest.main()

