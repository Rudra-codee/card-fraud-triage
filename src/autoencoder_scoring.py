"""
autoencoder_scoring.py

Anomaly scoring and artifact preservation for the Autoencoder model.
Higher scores represent higher mean squared reconstruction error:
    score(x) = mean((x - reconstructed_x) ** 2, axis=1)
"""

from typing import Optional
import numpy as np
import pandas as pd
import torch

from src.autoencoder_model import Autoencoder
from src.scoring import create_score_dataframe, save_scores, load_scores, summarize_scores


def compute_autoencoder_scores(
    model: Autoencoder,
    X: np.ndarray,
    batch_size: int = 1024,
    device: str = "cpu",
) -> np.ndarray:
    """
    Computes per-transaction anomaly scores as the Mean Squared Reconstruction Error:
        score(x) = mean((x - reconstructed_x) ** 2, axis=1)

    Higher scores indicate higher reconstruction error (more anomalous).

    Validates:
    - X is a 2D numpy array.
    - All output scores are strictly finite and non-negative.
    - Output length matches X row count.
    """
    if X.ndim != 2:
        raise ValueError(f"X must be a 2D array, got shape {X.shape}")

    if not np.isfinite(X).all():
        raise ValueError("X contains NaN or infinite values!")

    model.eval()
    dev = torch.device(device)
    model.to(dev)

    scores_list = []
    num_samples = len(X)

    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            batch_np = X[i : i + batch_size]
            batch_tensor = torch.tensor(batch_np, dtype=torch.float32).to(dev)

            # Reconstruct batch
            reconstructed = model(batch_tensor)

            # Compute per-sample MSE across the feature dimension (dim=1)
            # squared error: (x - recon)^2 -> mean across dim 1
            batch_mse = torch.mean((batch_tensor - reconstructed) ** 2, dim=1)
            scores_list.append(batch_mse.cpu().numpy())

    scores = np.concatenate(scores_list, axis=0)

    if len(scores) != num_samples:
        raise RuntimeError(f"Score array length ({len(scores)}) does not match input rows ({num_samples})")

    if not np.isfinite(scores).all():
        raise RuntimeError("Computed Autoencoder scores contain NaN or infinite values!")

    if (scores < 0.0).any():
        raise RuntimeError("Reconstruction error cannot be negative!")

    return scores
