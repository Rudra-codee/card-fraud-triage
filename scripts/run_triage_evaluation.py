"""
run_triage_evaluation.py

Member 4, Task 1: reproducible validation evaluation of the frozen GMM baseline
and the shortlisted full-covariance candidates (K = 2, 3, 4, 5).

What it does:
1. Loads ONLY validation data (features, labels, row IDs). Test files are never opened.
2. Checks every model artifact (converged, 29 features, fitted on legitimate rows only).
3. Scores validation rows with the saved models (no retraining) and joins
   scores to labels by row_id, not by row position.
4. Computes AP, PR curves, Recall @ 1% / 5% FPR with the thresholds used.
5. Runs extra integrity checks on the K=4 full model, whose AP is unusually high.
6. Saves:
   - results/tables/triage_model_comparison.csv
   - results/tables/triage_pr_curve_data.csv
   - results/tables/triage_validation_checks.json
   - results/figures/triage_precision_recall_curve.png

Usage:
    ./.venv/bin/python3 scripts/run_triage_evaluation.py
    ./.venv/bin/python3 scripts/run_triage_evaluation.py --refit-check   # also refits K=4 once to verify the artifact
"""

import argparse
import json
import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath("."))

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
from sklearn.mixture import GaussianMixture

from src.scoring import compute_anomaly_scores, create_score_dataframe
from src.triage_evaluation import (
    align_scores_with_labels,
    check_manifest_matches,
    evaluate_scores,
    load_validation_data,
    precision_recall_data,
)


PROCESSED_DIR = os.path.join("data", "processed")
MANIFEST_PATH = os.path.join("data", "splits", "split_manifest.csv")
FROZEN_BASELINE_SCORES = os.path.join(PROCESSED_DIR, "scores", "gmm_baseline_scores_val.csv")
TABLES_DIR = os.path.join("results", "tables")
FIGURES_DIR = os.path.join("results", "figures")

# Frozen baseline (K=1) plus the full-covariance shortlist
MODELS = [
    {"name": "gmm_baseline_k1_full", "label": "Baseline K=1 full", "k": 1, "path": "models/gmm_baseline.joblib"},
    {"name": "gmm_k2_full", "label": "K=2 full", "k": 2, "path": "models/candidates/gmm_k2_full.joblib"},
    {"name": "gmm_k3_full", "label": "K=3 full", "k": 3, "path": "models/candidates/gmm_k3_full.joblib"},
    {"name": "gmm_k4_full", "label": "K=4 full", "k": 4, "path": "models/candidates/gmm_k4_full.joblib"},
    {"name": "gmm_k5_full", "label": "K=5 full", "k": 5, "path": "models/candidates/gmm_k5_full.joblib"},
]

# Categorical colours (fixed order) and line styles as a second, non-colour cue
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
LINE_STYLES = ["-", "--", "-.", ":", (0, (5, 1, 1, 1))]


def check_model_artifact(gmm: GaussianMixture, metadata: dict, expected_k: int,
                         feature_names: list, legit_train_count: int, fraud_train_count: int) -> dict:
    """Checks that a saved model is the one we expect and was fitted on legitimate training rows only."""
    if gmm.n_components != expected_k or gmm.covariance_type != "full":
        raise ValueError(f"Expected K={expected_k} full, got K={gmm.n_components} {gmm.covariance_type}")
    if not gmm.converged_:
        raise ValueError("Model did not converge")
    if gmm.n_features_in_ != len(feature_names):
        raise ValueError(f"Model expects {gmm.n_features_in_} features, data has {len(feature_names)}")

    # Baseline metadata uses slightly different key names than the candidates
    n_train = metadata["n_training_samples"]
    n_fraud_excluded = metadata.get("train_fraud_excluded", metadata.get("train_fraud_rows_excluded"))
    if n_train != legit_train_count or n_fraud_excluded != fraud_train_count:
        raise ValueError(
            f"Model metadata says {n_train} training rows / {n_fraud_excluded} fraud excluded; "
            f"expected {legit_train_count} / {fraud_train_count}"
        )

    return {
        "n_components": int(gmm.n_components),
        "converged": bool(gmm.converged_),
        "n_features_in": int(gmm.n_features_in_),
        "n_training_samples": int(n_train),
        "train_fraud_excluded": int(n_fraud_excluded),
        "reg_covar": float(gmm.reg_covar),
    }


def score_direction_check(labels: np.ndarray, scores: np.ndarray) -> dict:
    """Higher scores should mean more anomalous: fraud should rank above legitimate rows."""
    fraud_median = float(np.median(scores[labels == 1]))
    legit_median = float(np.median(scores[labels == 0]))
    ap_higher_is_fraud = float(average_precision_score(labels, scores))
    ap_lower_is_fraud = float(average_precision_score(labels, -scores))

    if fraud_median <= legit_median or ap_higher_is_fraud <= ap_lower_is_fraud:
        raise ValueError("Score direction check failed: higher scores do not indicate fraud")

    return {
        "fraud_median_score": fraud_median,
        "legit_median_score": legit_median,
        "ap_if_scores_flipped": ap_lower_is_fraud,
    }


def k4_integrity_checks(labels: np.ndarray, scores: np.ndarray) -> dict:
    """
    Extra checks for the K=4 full model:
    - Shuffling labels (breaking any score-label link) should drop AP to about the fraud rate.
    - Shifting scores by one row (a typical misalignment bug) should also destroy AP.
    - Top-k counts show where the AP comes from.
    """
    rng = np.random.default_rng(42)
    permuted_aps = [average_precision_score(rng.permutation(labels), scores) for _ in range(100)]

    order = np.argsort(-scores, kind="stable")
    top_k_fraud = {f"top_{k}": int(labels[order[:k]].sum()) for k in [10, 50, 100, 500, 1000]}

    max_legit_score = float(scores[labels == 0].max())
    fraud_above_all_legit = int((scores[labels == 1] > max_legit_score).sum())

    return {
        "permuted_label_ap_mean": float(np.mean(permuted_aps)),
        "permuted_label_ap_max": float(np.max(permuted_aps)),
        "shift_by_one_row_ap": float(average_precision_score(labels, np.roll(scores, 1))),
        "fraud_in_top_k": top_k_fraud,
        "max_legit_score": max_legit_score,
        "fraud_scored_above_every_legit_row": fraud_above_all_legit,
    }


def k4_refit_check(saved_gmm: GaussianMixture, X_val: np.ndarray) -> dict:
    """
    Refits K=4 full once on legitimate TRAINING rows only, with the documented settings,
    and compares it to the saved artifact. Identical parameters prove the artifact
    was fitted without fraud, validation or test rows. Uses training labels only.
    """
    X_train = np.load(os.path.join(PROCESSED_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(PROCESSED_DIR, "y_train.npy"))

    refit = GaussianMixture(
        n_components=4, covariance_type="full", reg_covar=1e-6, n_init=5, random_state=42
    ).fit(X_train[y_train == 0])

    saved_scores = compute_anomaly_scores(saved_gmm, X_val)
    refit_scores = compute_anomaly_scores(refit, X_val)

    return {
        "max_abs_mean_difference": float(np.abs(saved_gmm.means_ - refit.means_).max()),
        "max_abs_covariance_difference": float(np.abs(saved_gmm.covariances_ - refit.covariances_).max()),
        "max_abs_val_score_difference": float(np.abs(saved_scores - refit_scores).max()),
    }


def plot_pr_curves(curves: dict, results_df: pd.DataFrame, prevalence: float, output_path: str) -> None:
    """Overlays the validation PR curves of every evaluated model."""
    fig, ax = plt.subplots(figsize=(9, 6), dpi=200)

    for idx, (model_name, curve) in enumerate(curves.items()):
        row = results_df[results_df["model_name"] == model_name].iloc[0]
        ax.step(
            curve["recall"], curve["precision"], where="post",
            color=COLORS[idx], linestyle=LINE_STYLES[idx], linewidth=2,
            label=f"{row['label']}  (AP = {row['average_precision']:.3f})",
        )

    ax.axhline(prevalence, color="#888888", linewidth=1, linestyle="--",
               label=f"Random ranking (fraud rate = {prevalence:.4f})")

    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall (share of validation fraud caught)", fontsize=11)
    ax.set_ylabel("Precision (share of alerts that are fraud)", fontsize=11)
    n_samples = int(results_df["n_samples"].iloc[0])
    fraud_count = int(results_df["fraud_count"].iloc[0])
    ax.set_title(f"Validation Precision-Recall curves (N = {n_samples:,}, fraud = {fraud_count})",
                 fontsize=12, fontweight="bold")
    ax.grid(color="#e5e5e5", linewidth=0.8)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.legend(loc="upper right", fontsize=9, frameon=False)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Validation triage evaluation for GMM models")
    parser.add_argument("--refit-check", action="store_true",
                        help="Refit K=4 full once on legitimate training rows to verify the saved artifact")
    parser.add_argument("--n-bootstrap", type=int, default=1000,
                        help="Bootstrap resamples for the AP 95%% interval (default 1000)")
    args = parser.parse_args()

    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("=" * 78)
    print("MEMBER 4, TASK 1: GMM VALIDATION TRIAGE EVALUATION")
    print("=" * 78)

    # 1. Load validation data only
    data = load_validation_data(PROCESSED_DIR)
    X_val, y_val, row_ids_val = data["X_val"], data["y_val"], data["row_ids_val"]

    with open(os.path.join(PROCESSED_DIR, "feature_names.json"), "r", encoding="utf-8") as f:
        feature_names = json.load(f)
    with open(os.path.join(PROCESSED_DIR, "preprocessing_metadata.json"), "r", encoding="utf-8") as f:
        prep_meta = json.load(f)
    legit_train_count = prep_meta["partitions"]["train"]["legitimate_count"]
    fraud_train_count = prep_meta["partitions"]["train"]["fraud_count"]

    # 2. Dataset checks
    n_manifest_rows = check_manifest_matches(row_ids_val, MANIFEST_PATH)
    if X_val.shape[1] != len(feature_names):
        raise ValueError(f"X_val has {X_val.shape[1]} columns, expected {len(feature_names)}")
    if len(y_val) != prep_meta["partitions"]["validation"]["total_rows"]:
        raise ValueError("Validation row count does not match preprocessing metadata")
    if int(y_val.sum()) != prep_meta["partitions"]["validation"]["fraud_count"]:
        raise ValueError("Validation fraud count does not match preprocessing metadata")

    n_samples = int(len(y_val))
    fraud_count = int(y_val.sum())
    prevalence = fraud_count / n_samples
    print(f"\nValidation rows: {n_samples:,} | fraud: {fraud_count} | prevalence: {prevalence:.4%}")
    print(f"Row IDs match split manifest ({n_manifest_rows:,} validation rows), all unique.")

    checks = {
        "validation_rows": n_samples,
        "validation_fraud": fraud_count,
        "fraud_prevalence": prevalence,
        "row_ids_match_manifest": True,
        "test_data_loaded": False,
        "models": {},
    }

    # 3. Score, align and evaluate each model
    rows = []
    curves = {}
    curve_tables = []
    aligned_scores = {}

    for model_info in MODELS:
        name = model_info["name"]
        if not os.path.exists(model_info["path"]):
            print(f"\n[skip] {name}: artifact not found at {model_info['path']}")
            continue

        print(f"\n--- {name} ---")
        gmm = joblib.load(model_info["path"])
        with open(model_info["path"].replace(".joblib", "_metadata.json"), "r", encoding="utf-8") as f:
            metadata = json.load(f)

        model_checks = check_model_artifact(
            gmm, metadata, model_info["k"], feature_names, legit_train_count, fraud_train_count
        )

        # Score with the saved model (no retraining) and keep the row IDs with the scores
        scores = compute_anomaly_scores(gmm, X_val)
        score_df = create_score_dataframe(row_ids_val, scores, split="validation")

        # Shuffle the score rows on purpose so the join below must rely on row_id, not position
        score_df = score_df.sample(frac=1.0, random_state=0)
        aligned = align_scores_with_labels(score_df, row_ids_val, y_val)
        aligned_labels = aligned["Class"].to_numpy()
        aligned_values = aligned["anomaly_score"].to_numpy()
        aligned_scores[name] = aligned_values

        model_checks["scores_finite"] = bool(np.isfinite(aligned_values).all())
        model_checks["n_distinct_scores"] = int(len(np.unique(aligned_values)))
        model_checks.update(score_direction_check(aligned_labels, aligned_values))

        # The frozen baseline also has a saved score file: our scores must match it row by row
        if name == "gmm_baseline_k1_full":
            frozen = pd.read_csv(FROZEN_BASELINE_SCORES)
            frozen_aligned = align_scores_with_labels(frozen, row_ids_val, y_val)
            diff = float(np.abs(frozen_aligned["anomaly_score"].to_numpy() - aligned_values).max())
            if diff > 1e-8:
                raise ValueError(f"Rescored baseline differs from frozen score file by {diff}")
            model_checks["max_abs_diff_vs_frozen_score_file"] = diff

        row = evaluate_scores(name, aligned_labels, aligned_values, n_bootstrap=args.n_bootstrap)
        row["label"] = model_info["label"]

        # Compare against the AP recorded during model selection (if any)
        if "average_precision" in metadata:
            model_checks["model_selection_ap"] = metadata["average_precision"]
            model_checks["model_selection_recall_at_1pct_fpr"] = metadata["recall_at_1pct_fpr"]
            model_checks["model_selection_recall_at_5pct_fpr"] = metadata["recall_at_5pct_fpr"]
            if abs(metadata["average_precision"] - row["average_precision"]) > 1e-5:
                raise ValueError(f"{name}: AP differs from model-selection metadata")

        if name == "gmm_k4_full":
            model_checks["k4_integrity"] = k4_integrity_checks(aligned_labels, aligned_values)
            if args.refit_check:
                print("Refitting K=4 full on legitimate training rows to verify the artifact...")
                model_checks["k4_refit_check"] = k4_refit_check(gmm, X_val)

        checks["models"][name] = model_checks
        rows.append(row)

        curve = precision_recall_data(aligned_labels, aligned_values)
        curves[name] = curve
        curve_tables.append(curve.assign(model_name=name))

        print(f"AP = {row['average_precision']:.4f} (95% CI {row.get('ap_ci_low', np.nan):.3f}-{row.get('ap_ci_high', np.nan):.3f})"
              f" | Recall@1%FPR = {row['recall_at_1pct_fpr']:.4f} ({row['tp_1pct_fpr']}/{fraud_count}, FPR {row['actual_fpr_1pct']:.4%})"
              f" | Recall@5%FPR = {row['recall_at_5pct_fpr']:.4f} ({row['tp_5pct_fpr']}/{fraud_count}, FPR {row['actual_fpr_5pct']:.4%})")

    # Every model must have been scored on exactly the same rows
    for name, values in aligned_scores.items():
        if len(values) != n_samples:
            raise ValueError(f"{name} was scored on {len(values)} rows, expected {n_samples}")

    # 4. Save outputs
    results_df = pd.DataFrame(rows)
    column_order = [
        "model_name", "label", "n_samples", "fraud_count", "fraud_prevalence",
        "average_precision", "ap_ci_low", "ap_ci_high", "roc_auc",
        "recall_at_1pct_fpr", "threshold_1pct_fpr", "actual_fpr_1pct", "tp_1pct_fpr", "fp_1pct_fpr",
        "recall_at_5pct_fpr", "threshold_5pct_fpr", "actual_fpr_5pct", "tp_5pct_fpr", "fp_5pct_fpr",
    ]
    results_df = results_df[[c for c in column_order if c in results_df.columns]]

    table_path = os.path.join(TABLES_DIR, "triage_model_comparison.csv")
    results_df.to_csv(table_path, index=False)

    curve_path = os.path.join(TABLES_DIR, "triage_pr_curve_data.csv")
    pd.concat(curve_tables, ignore_index=True)[["model_name", "threshold", "precision", "recall"]].to_csv(curve_path, index=False)

    checks_path = os.path.join(TABLES_DIR, "triage_validation_checks.json")
    with open(checks_path, "w", encoding="utf-8") as f:
        json.dump(checks, f, indent=2)

    figure_path = os.path.join(FIGURES_DIR, "triage_precision_recall_curve.png")
    plot_pr_curves(curves, results_df, prevalence, figure_path)

    print("\n=== VALIDATION TRIAGE COMPARISON ===")
    print(results_df[["model_name", "average_precision", "ap_ci_low", "ap_ci_high",
                      "recall_at_1pct_fpr", "threshold_1pct_fpr", "recall_at_5pct_fpr",
                      "threshold_5pct_fpr"]].to_string(index=False))

    if "gmm_k4_full" in checks["models"]:
        print("\n=== K=4 FULL INTEGRITY CHECKS ===")
        print(json.dumps({k: v for k, v in checks["models"]["gmm_k4_full"].items()
                          if k.startswith("k4_")}, indent=2))

    print(f"\nSaved: {table_path}\n       {curve_path}\n       {checks_path}\n       {figure_path}")


if __name__ == "__main__":
    main()
