"""
triage_evaluation.py

Reusable validation metrics for comparing GMM anomaly scorers (Member 4, Task 1).

Conventions used everywhere in this module:
- Scores follow src/scoring.py: anomaly_score = -gmm.score_samples(X).
  Higher score = less likely under the legitimate model = more suspicious.
- A transaction is flagged when anomaly_score >= threshold.
- Labels: 0 = legitimate, 1 = fraud.

Only validation data is loaded here. Test features and labels are never read.
"""

import os
from typing import Any, Dict
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def load_validation_data(processed_dir: str = "data/processed") -> Dict[str, np.ndarray]:
    """
    Loads ONLY the validation arrays (features, labels, row IDs).
    We deliberately do not use load_processed_data() because it also loads test labels.
    """
    X_val = np.load(os.path.join(processed_dir, "X_val.npy"))
    y_val = np.load(os.path.join(processed_dir, "y_val.npy"))
    row_ids_val = np.load(os.path.join(processed_dir, "row_ids_val.npy"))

    if not (len(X_val) == len(y_val) == len(row_ids_val)):
        raise ValueError(
            f"Validation arrays have different lengths: X={len(X_val)}, y={len(y_val)}, ids={len(row_ids_val)}"
        )

    return {"X_val": X_val, "y_val": y_val, "row_ids_val": row_ids_val}


def check_manifest_matches(row_ids_val: np.ndarray, manifest_path: str) -> int:
    """
    Confirms the validation row IDs are exactly the rows marked 'validation'
    in the chronological split manifest. Returns the number of validation rows.
    """
    manifest = pd.read_csv(manifest_path)
    manifest_val_ids = manifest.loc[manifest["split"] == "validation", "row_id"].to_numpy()

    if len(np.unique(row_ids_val)) != len(row_ids_val):
        raise ValueError("Validation row IDs contain duplicates")

    if set(manifest_val_ids) != set(row_ids_val):
        raise ValueError("Validation row IDs do not match the split manifest")

    return int(len(manifest_val_ids))


def validate_labels(labels: np.ndarray) -> None:
    """Labels must be binary and contain both classes."""
    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError(f"labels must be 1D, got shape {labels.shape}")

    unique_labels = set(np.unique(labels).tolist())
    if not unique_labels.issubset({0, 1}):
        raise ValueError(f"labels must contain only 0 and 1, got {unique_labels}")

    if unique_labels != {0, 1}:
        raise ValueError("labels must contain both legitimate (0) and fraud (1) rows")


def validate_scores(scores: np.ndarray, expected_length: int) -> None:
    """Scores must be a 1D array of finite numbers with one score per row."""
    scores = np.asarray(scores)
    if scores.ndim != 1:
        raise ValueError(f"scores must be 1D, got shape {scores.shape}")

    if len(scores) != expected_length:
        raise ValueError(f"scores length ({len(scores)}) does not match expected length ({expected_length})")

    if not np.issubdtype(scores.dtype, np.number):
        raise ValueError(f"scores must be numeric, got dtype {scores.dtype}")

    if not np.isfinite(scores).all():
        raise ValueError("scores contain NaN or infinite values")


def align_scores_with_labels(
    score_df: pd.DataFrame,
    row_ids: np.ndarray,
    labels: np.ndarray,
) -> pd.DataFrame:
    """
    Joins scores to labels using the stable row_id, NOT the row position.

    score_df must have columns 'row_id' and 'anomaly_score'.
    Returns a DataFrame with columns row_id, anomaly_score, Class,
    in the same order as row_ids.

    Raises an error if any row ID is duplicated, missing, or unexpected.
    """
    if len(row_ids) != len(labels):
        raise ValueError(f"row_ids length ({len(row_ids)}) does not match labels length ({len(labels)})")

    if score_df["row_id"].duplicated().any():
        raise ValueError("Score file contains duplicated row IDs")

    if len(np.unique(row_ids)) != len(row_ids):
        raise ValueError("Label row IDs contain duplicates")

    score_ids = set(score_df["row_id"])
    label_ids = set(row_ids)
    missing_ids = label_ids - score_ids
    extra_ids = score_ids - label_ids
    if missing_ids:
        raise ValueError(f"{len(missing_ids)} labelled rows have no score")
    if extra_ids:
        raise ValueError(f"{len(extra_ids)} scored rows are not in the labelled set")

    label_df = pd.DataFrame({"row_id": np.asarray(row_ids, dtype=int), "Class": np.asarray(labels, dtype=int)})

    # Left join keeps the label order; every row must find exactly one score
    aligned = label_df.merge(
        score_df[["row_id", "anomaly_score"]],
        on="row_id",
        how="left",
        validate="one_to_one",
    )
    aligned = aligned[["row_id", "anomaly_score", "Class"]]

    validate_scores(aligned["anomaly_score"].to_numpy(), len(label_df))
    return aligned


def recall_at_fpr(labels: np.ndarray, scores: np.ndarray, max_fpr: float) -> Dict[str, Any]:
    """
    Finds the most permissive threshold whose false positive rate is <= max_fpr,
    and reports the recall at that threshold.

    Threshold convention:
    - A row is flagged when score >= threshold.
    - roc_curve(drop_intermediate=False) returns one threshold per distinct score,
      ordered from strictest (fewest alerts) to most permissive (most alerts).
    - We pick the LOWEST threshold that still keeps FPR <= max_fpr.

    Tie handling:
    - Rows with the same score always get the same decision. If adding a group of
      tied scores would push FPR above the limit, the whole group is left unflagged.
      So the achieved FPR can be below the limit, but never above it.
    - If even the highest score breaks the limit, the threshold is +inf (no alerts).

    The recall and FPR are then recomputed directly from the flagged predictions
    and checked against roc_curve, so we never trust a "nearest" threshold blindly.
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    validate_labels(labels)
    validate_scores(scores, len(labels))

    if not 0.0 <= max_fpr <= 1.0:
        raise ValueError(f"max_fpr must be between 0 and 1, got {max_fpr}")

    fpr, tpr, thresholds = roc_curve(labels, scores, drop_intermediate=False)

    # fpr never decreases along the curve, so the last allowed index is the most permissive one
    allowed_indices = np.where(fpr <= max_fpr)[0]
    best_index = int(allowed_indices[-1])
    threshold = float(thresholds[best_index])

    # Recompute everything from the actual predictions
    flagged = scores >= threshold
    is_fraud = labels == 1
    is_legit = labels == 0

    true_positives = int((flagged & is_fraud).sum())
    false_positives = int((flagged & is_legit).sum())
    recall = true_positives / int(is_fraud.sum())
    actual_fpr = false_positives / int(is_legit.sum())

    if actual_fpr > max_fpr:
        raise RuntimeError(f"Chosen threshold gives FPR {actual_fpr:.6f}, above the limit {max_fpr}")
    if not np.isclose(recall, tpr[best_index]) or not np.isclose(actual_fpr, fpr[best_index]):
        raise RuntimeError("Recomputed recall/FPR does not match roc_curve output")

    return {
        "threshold": threshold,
        "recall": float(recall),
        "actual_fpr": float(actual_fpr),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "alerts": int(flagged.sum()),
    }


def precision_recall_data(labels: np.ndarray, scores: np.ndarray) -> pd.DataFrame:
    """
    Precision-Recall curve points from sklearn.metrics.precision_recall_curve.
    The final point (precision=1, recall=0) has no threshold, so it is stored as NaN.
    """
    validate_labels(labels)
    validate_scores(scores, len(labels))

    precision, recall, thresholds = precision_recall_curve(labels, scores)
    thresholds_padded = np.append(thresholds, np.nan)

    return pd.DataFrame({"threshold": thresholds_padded, "precision": precision, "recall": recall})


def bootstrap_average_precision(
    labels: np.ndarray,
    scores: np.ndarray,
    n_resamples: int = 1000,
    random_state: int = 42,
) -> Dict[str, float]:
    """
    Stratified bootstrap 95% interval for Average Precision.
    Fraud and legitimate rows are resampled separately so every resample
    keeps the same number of fraud cases (57 is small, so AP is noisy).
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    rng = np.random.default_rng(random_state)

    fraud_idx = np.where(labels == 1)[0]
    legit_idx = np.where(labels == 0)[0]

    ap_values = []
    for _ in range(n_resamples):
        sample_idx = np.concatenate([
            rng.choice(fraud_idx, size=len(fraud_idx), replace=True),
            rng.choice(legit_idx, size=len(legit_idx), replace=True),
        ])
        ap_values.append(average_precision_score(labels[sample_idx], scores[sample_idx]))

    return {
        "ap_ci_low": float(np.percentile(ap_values, 2.5)),
        "ap_ci_high": float(np.percentile(ap_values, 97.5)),
    }


def evaluate_scores(
    model_name: str,
    labels: np.ndarray,
    scores: np.ndarray,
    n_bootstrap: int = 1000,
) -> Dict[str, Any]:
    """
    Computes one comparison-table row for a model's validation scores.
    Accuracy is intentionally not reported: with ~0.1% fraud, flagging nothing is 99.9% accurate.
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    validate_labels(labels)
    validate_scores(scores, len(labels))

    n_samples = int(len(labels))
    fraud_count = int((labels == 1).sum())

    at_1pct = recall_at_fpr(labels, scores, max_fpr=0.01)
    at_5pct = recall_at_fpr(labels, scores, max_fpr=0.05)

    row = {
        "model_name": model_name,
        "n_samples": n_samples,
        "fraud_count": fraud_count,
        "fraud_prevalence": fraud_count / n_samples,
        "average_precision": float(average_precision_score(labels, scores)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "recall_at_1pct_fpr": at_1pct["recall"],
        "threshold_1pct_fpr": at_1pct["threshold"],
        "actual_fpr_1pct": at_1pct["actual_fpr"],
        "tp_1pct_fpr": at_1pct["true_positives"],
        "fp_1pct_fpr": at_1pct["false_positives"],
        "recall_at_5pct_fpr": at_5pct["recall"],
        "threshold_5pct_fpr": at_5pct["threshold"],
        "actual_fpr_5pct": at_5pct["actual_fpr"],
        "tp_5pct_fpr": at_5pct["true_positives"],
        "fp_5pct_fpr": at_5pct["false_positives"],
    }

    if n_bootstrap > 0:
        row.update(bootstrap_average_precision(labels, scores, n_resamples=n_bootstrap))

    return row
