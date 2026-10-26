"""
model_selection.py

Grid search, model fitting, and validation triage evaluation for Gaussian Mixture Model (GMM)
candidate selection (Phase 1, Task 3A).

This module supports:
1. Training GMM candidates strictly on legitimate training rows (y_train == 0).
2. Goodness-of-fit evaluation using AIC and BIC on legitimate training data.
3. Validation ranking evaluation via Average Precision (PR-AUC) and Recall at fixed FPR (1% and 5%).
4. Exporting comparison tables and model artifacts.
"""

import os
import time
import json
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.metrics import average_precision_score

from src.scoring import compute_anomaly_scores


def fit_gmm_candidate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_components: int,
    covariance_type: str,
    reg_covar: float = 1e-6,
    n_init: int = 5,
    random_state: int = 42,
) -> Tuple[Optional[GaussianMixture], Dict[str, Any]]:
    """
    Fits a single GMM candidate model strictly on legitimate training data (y_train == 0).
    
    Parameters:
    - X_train: Training feature matrix (shape: N, 29).
    - y_train: Training binary labels (0 = legit, 1 = fraud).
    - n_components: Number of mixture components (K).
    - covariance_type: Covariance structure ('full', 'diag', or 'tied').
    - reg_covar: Non-negative regularization added to covariance diagonal.
    - n_init: Number of EM initializations to perform.
    - random_state: Random seed for deterministic reproducibility.

    Returns:
    - gmm: The fitted GaussianMixture instance (or None if fitting failed).
    - record: Dictionary containing training diagnostics and goodness-of-fit metrics.
    """
    if len(X_train) != len(y_train):
        raise ValueError(f"X_train length ({len(X_train)}) does not match y_train length ({len(y_train)})")

    # Step 1: Filter strictly for legitimate transactions (y_train == 0)
    legit_mask = (y_train == 0)
    X_train_legit = X_train[legit_mask]
    num_legit = int(len(X_train_legit))
    num_fraud = int((y_train == 1).sum())

    model_name = f"gmm_k{n_components}_{covariance_type}"

    record: Dict[str, Any] = {
        "model_name": model_name,
        "n_components": int(n_components),
        "covariance_type": str(covariance_type),
        "reg_covar": float(reg_covar),
        "n_init": int(n_init),
        "random_state": int(random_state),
        "n_training_samples": num_legit,
        "train_fraud_excluded": num_fraud,
        "n_features": int(X_train.shape[1]),
        "status": "pending",
        "error_message": None,
    }

    # Step 2: Instantiate model
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        reg_covar=reg_covar,
        n_init=n_init,
        random_state=random_state,
    )

    # Step 3: Fit model and measure duration
    t_start = time.time()
    try:
        gmm.fit(X_train_legit)
        fit_duration = time.time() - t_start

        converged = bool(getattr(gmm, "converged_", False))
        n_iter = int(getattr(gmm, "n_iter_", 0))
        lower_bound = float(getattr(gmm, "lower_bound_", np.nan))

        # Step 4: Compute AIC and BIC on legitimate training subset
        train_aic = float(gmm.aic(X_train_legit))
        train_bic = float(gmm.bic(X_train_legit))

        record.update({
            "status": "success",
            "fit_duration_seconds": round(fit_duration, 3),
            "converged": converged,
            "n_iter": n_iter,
            "lower_bound": lower_bound,
            "aic": train_aic,
            "bic": train_bic,
        })
        return gmm, record

    except Exception as e:
        fit_duration = time.time() - t_start
        record.update({
            "status": "failed",
            "fit_duration_seconds": round(fit_duration, 3),
            "converged": False,
            "n_iter": 0,
            "lower_bound": np.nan,
            "aic": np.nan,
            "bic": np.nan,
            "error_message": str(e),
        })
        return None, record


def compute_triage_metrics(
    y_true: np.ndarray,
    anomaly_scores: np.ndarray,
) -> Dict[str, float]:
    """
    Computes validation triage ranking metrics given ground truth labels and anomaly scores.
    Higher scores indicate higher likelihood of anomaly (suspicious transaction).

    Metrics computed:
    - Average Precision (AP / PR-AUC): Summarizes precision across all recall thresholds.
    - Recall @ 1% FPR: Fraction of fraud caught when the triage capacity allows a 1% false-positive rate.
    - Recall @ 5% FPR: Fraction of fraud caught when the triage capacity allows a 5% false-positive rate.
    - Associated triage score thresholds at 1% and 5% FPR.

    Definition of Recall at fixed FPR (e.g. 1%):
    1. Determine threshold tau on legitimate validation samples such that exactly 1% of normal transactions
       fall at or above tau: tau = percentile(scores[y_true == 0], 100 * (1 - 0.01)) = 99th percentile.
    2. Calculate recall as fraction of fraud samples scoring at or above tau:
       recall = mean(scores[y_true == 1] >= tau).
    """
    if len(y_true) != len(anomaly_scores):
        raise ValueError("y_true and anomaly_scores must have identical lengths")

    if not np.isfinite(anomaly_scores).all():
        raise ValueError("anomaly_scores must be finite to compute evaluation metrics")

    # 1. Average Precision (PR-AUC)
    ap = float(average_precision_score(y_true, anomaly_scores))

    # 2. Extract cohort scores
    legit_mask = (y_true == 0)
    fraud_mask = (y_true == 1)

    legit_scores = anomaly_scores[legit_mask]
    fraud_scores = anomaly_scores[fraud_mask]

    if len(legit_scores) == 0 or len(fraud_scores) == 0:
        raise ValueError("Both legitimate and fraud samples must be present in y_true")

    # 3. Recall @ 1% FPR (99th percentile threshold of legitimate scores)
    tau_1pct = float(np.percentile(legit_scores, 99.0))
    recall_1pct = float((fraud_scores >= tau_1pct).mean())

    # 4. Recall @ 5% FPR (95th percentile threshold of legitimate scores)
    tau_5pct = float(np.percentile(legit_scores, 95.0))
    recall_5pct = float((fraud_scores >= tau_5pct).mean())

    return {
        "average_precision": round(ap, 6),
        "recall_at_1pct_fpr": round(recall_1pct, 6),
        "recall_at_5pct_fpr": round(recall_5pct, 6),
        "threshold_1pct_fpr": round(tau_1pct, 4),
        "threshold_5pct_fpr": round(tau_5pct, 4),
    }


def run_gmm_grid_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    components_list: List[int] = [2, 3, 4, 5],
    covariance_types: List[str] = ["full", "diag", "tied"],
    reg_covar: float = 1e-6,
    n_init: int = 5,
    random_state: int = 42,
    save_models_dir: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, GaussianMixture]]:
    """
    Executes the full GMM candidate grid search across component counts and covariance types.
    Evaluates each candidate on the validation partition using negative log-likelihood scoring.

    Returns:
    - results_df: Summary DataFrame containing all configurations and comparative metrics.
    - fitted_models: Dictionary mapping model_name to fitted GaussianMixture instances.
    """
    if save_models_dir:
        os.makedirs(save_models_dir, exist_ok=True)

    records: List[Dict[str, Any]] = []
    fitted_models: Dict[str, GaussianMixture] = {}

    total_candidates = len(components_list) * len(covariance_types)
    current_idx = 0

    for k in components_list:
        for cov in covariance_types:
            current_idx += 1
            print(f"[{current_idx}/{total_candidates}] Fitting GMM (K={k}, covariance_type='{cov}')...", end=" ", flush=True)

            gmm, record = fit_gmm_candidate(
                X_train=X_train,
                y_train=y_train,
                n_components=k,
                covariance_type=cov,
                reg_covar=reg_covar,
                n_init=n_init,
                random_state=random_state,
            )

            if gmm is not None and record["status"] == "success":
                # Compute validation anomaly scores: -gmm.score_samples(X_val)
                val_scores = compute_anomaly_scores(gmm, X_val)
                scores_finite = bool(np.isfinite(val_scores).all())
                record["val_scores_finite"] = scores_finite

                # Validation score statistics
                record["val_score_mean"] = round(float(np.mean(val_scores)), 2)
                record["val_score_median"] = round(float(np.median(val_scores)), 2)
                record["val_score_std"] = round(float(np.std(val_scores)), 2)

                # Validation triage ranking metrics
                triage_metrics = compute_triage_metrics(y_val, val_scores)
                record.update(triage_metrics)

                # Save candidate model if directory specified
                if save_models_dir:
                    model_path = os.path.join(save_models_dir, f"{record['model_name']}.joblib")
                    joblib.dump(gmm, model_path)
                    meta_path = os.path.join(save_models_dir, f"{record['model_name']}_metadata.json")
                    with open(meta_path, "w", encoding="utf-8") as f:
                        json.dump(record, f, indent=2)

                fitted_models[record["model_name"]] = gmm
                print(f"Done in {record['fit_duration_seconds']}s | AIC: {record['aic']:.1f} | AP: {record['average_precision']:.4f} | R@1%FPR: {record['recall_at_1pct_fpr']:.4f}")
            else:
                record["val_scores_finite"] = False
                record["average_precision"] = np.nan
                record["recall_at_1pct_fpr"] = np.nan
                record["recall_at_5pct_fpr"] = np.nan
                print(f"FAILED ({record.get('error_message')})")

            records.append(record)

    results_df = pd.DataFrame(records)
    return results_df, fitted_models
