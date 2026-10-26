"""
scoring.py

Anomaly scoring and score artifact preservation for the GMM baseline.
Higher scores represent lower likelihood under the learned legitimate distribution
(negative log-likelihood).
"""

import os
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture


def compute_anomaly_scores(gmm: GaussianMixture, X: np.ndarray) -> np.ndarray:
    """
    Computes negative log-likelihood anomaly scores:
    anomaly_score = -gmm.score_samples(X)

    Higher scores indicate observations that are less probable under
    the legitimate transaction density model.

    Validates:
    - X is a 2D numpy array.
    - All output scores are strictly finite (no NaN, +inf, -inf).
    - Output length matches X row count.
    """
    if X.ndim != 2:
        raise ValueError(f"X must be a 2D array, got shape {X.shape}")

    if not np.isfinite(X).all():
        raise ValueError("X contains NaN or infinite values!")

    # Negative log-likelihood
    log_likelihood = gmm.score_samples(X)
    scores = -log_likelihood

    if len(scores) != len(X):
        raise RuntimeError(f"Score array length ({len(scores)}) does not match input rows ({len(X)})")

    if not np.isfinite(scores).all():
        raise RuntimeError("Computed anomaly scores contain NaN or infinite values!")

    return scores


def create_score_dataframe(
    row_ids: np.ndarray,
    scores: np.ndarray,
    split: str,
    labels: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Packages anomaly scores into a structured pandas DataFrame preserving original row IDs.
    Row IDs must be explicitly tracked with every score record.
    """
    if len(row_ids) != len(scores):
        raise ValueError(f"Row IDs length ({len(row_ids)}) does not match scores length ({len(scores)})")

    data_dict = {
        "row_id": np.asarray(row_ids, dtype=int),
        "anomaly_score": np.asarray(scores, dtype=float),
        "split": str(split),
    }

    if labels is not None:
        if len(labels) != len(scores):
            raise ValueError(f"Labels length ({len(labels)}) does not match scores length ({len(scores)})")
        data_dict["Class"] = np.asarray(labels, dtype=int)

    df = pd.DataFrame(data_dict)
    return df


def save_scores(
    score_df: pd.DataFrame,
    output_path: str,
) -> str:
    """
    Persists score dataframe to CSV, ensuring parent directory exists.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    score_df.to_csv(output_path, index=False)
    return output_path


def load_scores(file_path: str) -> pd.DataFrame:
    """
    Loads score dataframe and validates expected columns.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Score file not found at: {file_path}")

    df = pd.read_csv(file_path)
    required_cols = {"row_id", "anomaly_score"}
    if not required_cols.issubset(set(df.columns)):
        raise ValueError(f"Score file missing required columns: {required_cols - set(df.columns)}")

    return df


def summarize_scores(
    scores: np.ndarray,
    labels: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Computes statistical diagnostics on anomaly scores.
    If labels are provided, computes breakdown for legitimate vs fraud classes.
    """
    def _stats(arr: np.ndarray) -> Dict[str, float]:
        return {
            "count": int(len(arr)),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "p25": float(np.percentile(arr, 25)),
            "median": float(np.median(arr)),
            "p75": float(np.percentile(arr, 75)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
            "max": float(np.max(arr)),
        }

    summary: Dict[str, Any] = {"overall": _stats(scores)}

    if labels is not None:
        legit_mask = (labels == 0)
        fraud_mask = (labels == 1)

        if legit_mask.any():
            summary["legitimate"] = _stats(scores[legit_mask])
        if fraud_mask.any():
            summary["fraud"] = _stats(scores[fraud_mask])

    return summary
