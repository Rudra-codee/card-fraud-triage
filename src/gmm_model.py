"""
gmm_model.py

Configuration, training, serialization, and deserialization
for the Gaussian Mixture Model (GMM) baseline.
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
import sklearn
from sklearn.mixture import GaussianMixture


DEFAULT_GMM_CONFIG: Dict[str, Any] = {
    "n_components": 1,
    "covariance_type": "full",
    "reg_covar": 1e-6,
    "n_init": 5,
    "random_state": 42,
}


def create_gmm_model(config: Optional[Dict[str, Any]] = None) -> GaussianMixture:
    """
    Instantiates a scikit-learn GaussianMixture model using the specified
    configuration, falling back to DEFAULT_GMM_CONFIG for unspecified parameters.
    """
    cfg = DEFAULT_GMM_CONFIG.copy()
    if config:
        cfg.update(config)

    return GaussianMixture(
        n_components=cfg["n_components"],
        covariance_type=cfg["covariance_type"],
        reg_covar=cfg["reg_covar"],
        n_init=cfg["n_init"],
        random_state=cfg["random_state"],
    )


def fit_gmm_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: Optional[Dict[str, Any]] = None,
    feature_names: Optional[List[str]] = None,
) -> Tuple[GaussianMixture, Dict[str, Any]]:
    """
    Fits a GMM strictly on legitimate training transactions (y_train == 0).

    Validates:
    - X_train and y_train have matching row counts.
    - y_train contains valid binary labels.
    - No fraudulent transactions (y_train == 1) enter the training subset.
    - Model convergence is verified (raises RuntimeError if converged_ is False).

    Returns:
    - gmm: The fitted GaussianMixture instance.
    - metadata: Comprehensive metadata dictionary describing configuration and fit results.
    """
    if len(X_train) != len(y_train):
        raise ValueError(
            f"X_train length ({len(X_train)}) does not match y_train length ({len(y_train)})"
        )

    unique_labels = set(np.unique(y_train))
    if not unique_labels.issubset({0, 1}):
        raise ValueError(f"y_train must contain only binary labels 0 and 1, got {unique_labels}")

    # Extract legitimate subset exclusively
    legit_mask = (y_train == 0)
    num_legit = int(legit_mask.sum())
    num_fraud = int((y_train == 1).sum())

    if num_legit == 0:
        raise ValueError("No legitimate transactions (y_train == 0) found in training set!")

    X_train_legit = X_train[legit_mask]

    # Sanity checks on subset
    assert len(X_train_legit) == num_legit, "Subset size mismatch with legitimate count"
    assert not np.isnan(X_train_legit).any(), "X_train_legit contains NaN values"
    assert np.isfinite(X_train_legit).all(), "X_train_legit contains non-finite values"

    # Instantiate model
    cfg = DEFAULT_GMM_CONFIG.copy()
    if config:
        cfg.update(config)

    gmm = create_gmm_model(cfg)

    # Fit exclusively on legitimate observations
    gmm.fit(X_train_legit)

    # Check convergence explicitly
    if not getattr(gmm, "converged_", False):
        raise RuntimeError(
            f"GMM baseline failed to converge after {getattr(gmm, 'n_iter_', 'unknown')} iterations!"
        )

    # Package metadata
    metadata: Dict[str, Any] = {
        "model_name": "gmm_baseline_1comp_full",
        "model_type": "GaussianMixture",
        "config": cfg,
        "n_training_samples": num_legit,
        "total_train_partition_rows": len(X_train),
        "train_fraud_rows_excluded": num_fraud,
        "n_features_in": int(gmm.n_features_in_),
        "feature_names": feature_names if feature_names else [f"feature_{i}" for i in range(X_train.shape[1])],
        "converged": bool(gmm.converged_),
        "n_iter": int(gmm.n_iter_),
        "lower_bound": float(gmm.lower_bound_),
        "scikit_learn_version": sklearn.__version__,
    }

    return gmm, metadata


def save_gmm_model(
    gmm: GaussianMixture,
    metadata: Dict[str, Any],
    output_dir: str = "models",
    model_basename: str = "gmm_baseline",
) -> Tuple[str, str]:
    """
    Saves the fitted GMM model and its corresponding metadata.

    Returns:
    - Tuple of (model_file_path, metadata_file_path)
    """
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(output_dir, f"{model_basename}.joblib")
    joblib.dump(gmm, model_path)

    metadata_path = os.path.join(output_dir, f"{model_basename}_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return model_path, metadata_path


def load_gmm_model(
    model_path: str = "models/gmm_baseline.joblib",
    metadata_path: Optional[str] = None,
) -> Tuple[GaussianMixture, Optional[Dict[str, Any]]]:
    """
    Loads a saved GMM model and its optional metadata dictionary.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    gmm = joblib.load(model_path)

    metadata = None
    if metadata_path is None:
        default_meta = model_path.replace(".joblib", "_metadata.json")
        if os.path.exists(default_meta):
            metadata_path = default_meta

    if metadata_path and os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    return gmm, metadata
