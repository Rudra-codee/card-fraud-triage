"""
run_final_test_evaluation.py

Member 4: Final held-out test evaluation of the frozen anomaly-detection models:
1. GMM Baseline (K=1 full covariance)
2. GMM K=4 Full Covariance (shortlisted candidate from validation)
3. Autoencoder Baseline (neural reconstruction anomaly detector)

Research integrity rules strictly enforced:
- Zero model retraining or parameter updating.
- Zero threshold tuning or adjusting using test labels.
- All operating thresholds are frozen from validation outputs.
- Test labels are accessed ONLY for final metric calculation after score generation.
- Exact alignment of features, row IDs, and labels verified.
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
from sklearn.metrics import average_precision_score, roc_auc_score

from src.scoring import compute_anomaly_scores, create_score_dataframe, save_scores
from src.autoencoder_model import load_autoencoder_model
from src.autoencoder_scoring import compute_autoencoder_scores
from src.triage_evaluation import (
    align_scores_with_labels,
    bootstrap_average_precision,
    check_test_manifest_matches,
    evaluate_at_frozen_threshold,
    evaluate_test_scores,
    load_test_data,
    metrics_at_alert_budget,
    precision_recall_data,
    validate_labels,
    validate_scores,
)

PROCESSED_DIR = os.path.join("data", "processed")
MANIFEST_PATH = os.path.join("data", "splits", "split_manifest.csv")
SCORES_DIR = os.path.join(PROCESSED_DIR, "scores")
TABLES_DIR = os.path.join("results", "tables")
FIGURES_DIR = os.path.join("results", "figures")

# Frozen model candidates
EVALUATION_MODELS = [
    {
        "model_id": "gmm_baseline",
        "display_name": "GMM Baseline (K=1 full)",
        "model_type": "GMM",
        "artifact_path": "models/gmm_baseline.joblib",
        "metadata_path": "models/gmm_baseline_metadata.json",
        "score_file": "gmm_baseline_scores_test.csv",
        "score_type": "Negative Log-Likelihood: -log p(x)",
        "val_threshold_1pct": 158.81380296025483,
        "val_threshold_5pct": 60.72575217322757,
        "val_ap": 0.074028,
        "val_recall_1pct": 0.736842,
        "val_actual_fpr_1pct": 0.009947,
        "val_recall_5pct": 0.877193,
        "val_actual_fpr_5pct": 0.050000,
    },
    {
        "model_id": "gmm_k4_full",
        "display_name": "GMM K=4 Full Covariance",
        "model_type": "GMM",
        "artifact_path": "models/candidates/gmm_k4_full.joblib",
        "metadata_path": "models/candidates/gmm_k4_full_metadata.json",
        "score_file": "gmm_k4_full_scores_test.csv",
        "score_type": "Negative Log-Likelihood: -log p(x)",
        "val_threshold_1pct": 66.43595478102685,
        "val_threshold_5pct": 50.24139607102004,
        "val_ap": 0.483960,
        "val_recall_1pct": 0.754386,
        "val_actual_fpr_1pct": 0.009965,
        "val_recall_5pct": 0.824561,
        "val_actual_fpr_5pct": 0.050000,
    },
    {
        "model_id": "autoencoder_baseline",
        "display_name": "Autoencoder Baseline",
        "model_type": "Autoencoder",
        "artifact_path": "models/autoencoder_baseline.pt",
        "metadata_path": "models/autoencoder_baseline_metadata.json",
        "score_file": "autoencoder_scores_test.csv",
        "score_type": "Reconstruction MSE: mean((x - x_rec)^2)",
        "val_threshold_1pct": 2.651143789291382,
        "val_threshold_5pct": 1.341099739074707,
        "val_ap": 0.076527,
        "val_recall_1pct": 0.719298,
        "val_actual_fpr_1pct": 0.010000,
        "val_recall_5pct": 0.877193,
        "val_actual_fpr_5pct": 0.050000,
    },
]

REVIEW_BUDGETS = [50, 100, 200, 500]
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]
LINE_STYLES = ["-", "--", "-."]


def plot_test_pr_curves(curves: dict, results_df: pd.DataFrame, prevalence: float, output_path: str) -> None:
    """Overlays the held-out test Precision-Recall curves of evaluated models."""
    fig, ax = plt.subplots(figsize=(9, 6), dpi=200)

    for idx, (model_id, curve) in enumerate(curves.items()):
        row = results_df[results_df["model_name"] == model_id].iloc[0]
        ax.step(
            curve["recall"],
            curve["precision"],
            where="post",
            color=COLORS[idx],
            linestyle=LINE_STYLES[idx],
            linewidth=2,
            label=f"{row['label']}  (AP = {row['average_precision']:.4f})",
        )

    ax.axhline(
        prevalence,
        color="#888888",
        linewidth=1,
        linestyle=":",
        label=f"Random ranking (prevalence = {prevalence:.4%})",
    )

    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall (share of test fraud caught)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Precision (share of alerts that are fraud)", fontsize=11, fontweight="bold")
    n_samples = int(results_df["n_samples"].iloc[0])
    fraud_count = int(results_df["fraud_count"].iloc[0])
    ax.set_title(
        f"Final Held-Out Test Precision-Recall Curves (N = {n_samples:,}, Fraud = {fraud_count})",
        fontsize=12,
        fontweight="bold",
    )
    ax.grid(color="#e5e5e5", linewidth=0.8)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.legend(loc="upper right", fontsize=9, frameon=True, facecolor="white", edgecolor="#cccccc")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_review_capacity_curves(capacity_df: pd.DataFrame, output_path: str) -> None:
    """Plots frauds caught vs review budget for test partition."""
    fig, ax = plt.subplots(figsize=(8, 5), dpi=200)

    for idx, model_id in enumerate(capacity_df["model_name"].unique()):
        sub_df = capacity_df[capacity_df["model_name"] == model_id].sort_values("budget")
        ax.plot(
            sub_df["budget"],
            sub_df["frauds_caught"],
            marker="o",
            linewidth=2,
            color=COLORS[idx],
            linestyle=LINE_STYLES[idx],
            label=sub_df["label"].iloc[0] if "label" in sub_df.columns else model_id,
        )

    ax.axhline(75, color="#d62728", linestyle="--", linewidth=1, label="Total test frauds (75)")
    ax.set_xlabel("Review Budget (Top-N alerts reviewed)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Frauds Caught", fontsize=11, fontweight="bold")
    ax.set_title("Test Fraud Triage by Review Capacity", fontsize=12, fontweight="bold")
    ax.set_xticks(REVIEW_BUDGETS)
    ax.set_ylim(0, 80)
    ax.grid(color="#e5e5e5", linewidth=0.8)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.legend(loc="lower right", fontsize=9, frameon=True, facecolor="white", edgecolor="#cccccc")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def run_evaluation(n_bootstrap: int = 1000) -> dict:
    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(SCORES_DIR, exist_ok=True)

    print("=" * 80)
    print("MEMBER 4: FINAL HELD-OUT TEST EVALUATION — CARD FRAUD TRIAGE")
    print("=" * 80)

    # 1. Load test data and metadata
    print("\n[1/5] Loading test partition arrays and verifying split integrity...")
    test_data = load_test_data(PROCESSED_DIR)
    X_test = test_data["X_test"]
    y_test = test_data["y_test"]
    row_ids_test = test_data["row_ids_test"]

    with open(os.path.join(PROCESSED_DIR, "feature_names.json"), "r", encoding="utf-8") as f:
        feature_names = json.load(f)
    with open(os.path.join(PROCESSED_DIR, "preprocessing_metadata.json"), "r", encoding="utf-8") as f:
        prep_meta = json.load(f)

    # Dataset checks
    n_manifest_rows = check_test_manifest_matches(row_ids_test, MANIFEST_PATH)
    validate_labels(y_test)

    n_test_samples = int(len(y_test))
    n_test_fraud = int((y_test == 1).sum())
    n_test_legit = int((y_test == 0).sum())
    prevalence = n_test_fraud / n_test_samples

    expected_total = prep_meta["partitions"]["test"]["total_rows"]
    expected_fraud = prep_meta["partitions"]["test"]["fraud_count"]
    expected_legit = prep_meta["partitions"]["test"]["legitimate_count"]

    if n_test_samples != expected_total:
        raise ValueError(f"Test samples count mismatch: got {n_test_samples}, expected {expected_total}")
    if n_test_fraud != expected_fraud or n_test_legit != expected_legit:
        raise ValueError(
            f"Test class count mismatch: got fraud={n_test_fraud}, legit={n_test_legit}; "
            f"expected fraud={expected_fraud}, legit={expected_legit}"
        )
    if X_test.shape[1] != len(feature_names):
        raise ValueError(f"Feature count mismatch: got {X_test.shape[1]}, expected {len(feature_names)}")
    if not np.isfinite(X_test).all():
        raise ValueError("X_test contains non-finite values (NaN or Inf)")

    print(f"  Test rows: {n_test_samples:,} (Legitimate: {n_test_legit:,}, Fraud: {n_test_fraud})")
    print(f"  Fraud prevalence: {prevalence:.4%}")
    print(f"  Features ({len(feature_names)}): {feature_names[0]}..{feature_names[-1]}")
    print(f"  Row IDs match test manifest ({n_manifest_rows:,} rows) with zero duplicate IDs.")

    # Integrity verification log
    audit_checks = {
        "test_rows": n_test_samples,
        "test_legitimate": n_test_legit,
        "test_fraud": n_test_fraud,
        "fraud_prevalence": prevalence,
        "feature_count": len(feature_names),
        "manifest_match": True,
        "row_ids_unique": bool(len(np.unique(row_ids_test)) == len(row_ids_test)),
        "X_test_finite": bool(np.isfinite(X_test).all()),
        "models": {},
    }

    # 2. Score and evaluate each frozen model
    print("\n[2/5] Generating test scores and applying frozen validation thresholds...")
    model_comparison_rows = []
    threshold_metrics_rows = []
    capacity_rows = []
    val_vs_test_rows = []
    curves = {}

    for model_cfg in EVALUATION_MODELS:
        m_id = model_cfg["model_id"]
        d_name = model_cfg["display_name"]
        print(f"\n--- {d_name} ({m_id}) ---")

        if not os.path.exists(model_cfg["artifact_path"]):
            raise FileNotFoundError(f"Missing frozen artifact: {model_cfg['artifact_path']}")

        # A. Score computation strictly without using test labels
        if model_cfg["model_type"] == "GMM":
            model = joblib.load(model_cfg["artifact_path"])
            if not model.converged_:
                raise RuntimeError(f"GMM {m_id} did not converge!")
            raw_scores = compute_anomaly_scores(model, X_test)
        elif model_cfg["model_type"] == "Autoencoder":
            model, _ = load_autoencoder_model(model_cfg["artifact_path"], model_cfg["metadata_path"])
            raw_scores = compute_autoencoder_scores(model, X_test)
        else:
            raise ValueError(f"Unknown model type: {model_cfg['model_type']}")

        validate_scores(raw_scores, n_test_samples)

        # B. Save test score file
        score_df = create_score_dataframe(row_ids_test, raw_scores, split="test")
        score_file_path = os.path.join(SCORES_DIR, model_cfg["score_file"])
        save_scores(score_df, score_file_path)
        print(f"  Saved test scores to: {score_file_path} ({len(score_df):,} rows)")

        # C. Align scores with test labels by row_id (shuffling score_df first to test alignment robustness)
        shuffled_score_df = score_df.sample(frac=1.0, random_state=42)
        aligned_df = align_scores_with_labels(shuffled_score_df, row_ids_test, y_test)
        aligned_labels = aligned_df["Class"].to_numpy()
        aligned_scores = aligned_df["anomaly_score"].to_numpy()

        # D. Evaluate test metrics at frozen thresholds
        th1 = model_cfg["val_threshold_1pct"]
        th5 = model_cfg["val_threshold_5pct"]

        test_row = evaluate_test_scores(
            model_name=m_id,
            labels=aligned_labels,
            scores=aligned_scores,
            threshold_1pct=th1,
            threshold_5pct=th5,
            n_bootstrap=n_bootstrap,
        )
        test_row["label"] = d_name
        test_row["artifact_path"] = model_cfg["artifact_path"]
        model_comparison_rows.append(test_row)

        # Detailed threshold metrics table
        m_at_1 = evaluate_at_frozen_threshold(aligned_labels, aligned_scores, th1)
        m_at_5 = evaluate_at_frozen_threshold(aligned_labels, aligned_scores, th5)

        threshold_metrics_rows.append({
            "model_name": m_id,
            "label": d_name,
            "operating_point": "1% validation FPR",
            "frozen_threshold": th1,
            "target_val_fpr": 0.01,
            "achieved_val_fpr": model_cfg["val_actual_fpr_1pct"],
            "val_recall": model_cfg["val_recall_1pct"],
            "actual_test_fpr": m_at_1["actual_fpr"],
            "test_alerts": m_at_1["alerts"],
            "test_tp": m_at_1["true_positives"],
            "test_fp": m_at_1["false_positives"],
            "test_recall": m_at_1["recall"],
            "test_precision": m_at_1["precision"],
        })
        threshold_metrics_rows.append({
            "model_name": m_id,
            "label": d_name,
            "operating_point": "5% validation FPR",
            "frozen_threshold": th5,
            "target_val_fpr": 0.05,
            "achieved_val_fpr": model_cfg["val_actual_fpr_5pct"],
            "val_recall": model_cfg["val_recall_5pct"],
            "actual_test_fpr": m_at_5["actual_fpr"],
            "test_alerts": m_at_5["alerts"],
            "test_tp": m_at_5["true_positives"],
            "test_fp": m_at_5["false_positives"],
            "test_recall": m_at_5["recall"],
            "test_precision": m_at_5["precision"],
        })

        # E. Review budget evaluation (Top-N alerts)
        for budget in REVIEW_BUDGETS:
            budget_res = metrics_at_alert_budget(aligned_labels, aligned_scores, budget)
            capacity_rows.append({
                "model_name": m_id,
                "label": d_name,
                "budget": budget,
                "alerts": budget_res["alerts"],
                "frauds_caught": budget_res["frauds_caught"],
                "false_positives": budget_res["false_positives"],
                "precision": budget_res["precision"],
                "recall": budget_res["recall"],
                "fpr": budget_res["fpr"],
                "threshold": budget_res["threshold"],
            })

        # F. Validation vs Test side-by-side row
        val_vs_test_rows.append({
            "model_name": m_id,
            "label": d_name,
            "split": "validation",
            "n_samples": 56957,
            "fraud_count": 57,
            "average_precision": model_cfg["val_ap"],
            "recall_at_1pct": model_cfg["val_recall_1pct"],
            "actual_fpr_1pct": model_cfg["val_actual_fpr_1pct"],
            "recall_at_5pct": model_cfg["val_recall_5pct"],
            "actual_fpr_5pct": model_cfg["val_actual_fpr_5pct"],
        })
        val_vs_test_rows.append({
            "model_name": m_id,
            "label": d_name,
            "split": "test",
            "n_samples": n_test_samples,
            "fraud_count": n_test_fraud,
            "average_precision": test_row["average_precision"],
            "recall_at_1pct": test_row["recall_at_1pct_fpr"],
            "actual_fpr_1pct": test_row["actual_fpr_1pct"],
            "recall_at_5pct": test_row["recall_at_5pct_fpr"],
            "actual_fpr_5pct": test_row["actual_fpr_5pct"],
        })

        # PR curve data
        curve = precision_recall_data(aligned_labels, aligned_scores)
        curves[m_id] = curve

        # Console summary
        print(f"  Test AP: {test_row['average_precision']:.4f} "
              f"(95% CI: [{test_row.get('ap_ci_low', np.nan):.4f}, {test_row.get('ap_ci_high', np.nan):.4f}])")
        print(f"  Test ROC-AUC: {test_row['roc_auc']:.4f}")
        print(f"  Frozen 1% FPR Th ({th1:.4f}): Recall={m_at_1['recall']:.4%} "
              f"({m_at_1['true_positives']}/{n_test_fraud}), FP={m_at_1['false_positives']}, "
              f"Actual FPR={m_at_1['actual_fpr']:.4%}, Precision={m_at_1['precision']:.4%}")
        print(f"  Frozen 5% FPR Th ({th5:.4f}): Recall={m_at_5['recall']:.4%} "
              f"({m_at_5['true_positives']}/{n_test_fraud}), FP={m_at_5['false_positives']}, "
              f"Actual FPR={m_at_5['actual_fpr']:.4%}, Precision={m_at_5['precision']:.4%}")

        audit_checks["models"][m_id] = {
            "artifact_path": model_cfg["artifact_path"],
            "score_finite": bool(np.isfinite(aligned_scores).all()),
            "distinct_scores": int(len(np.unique(aligned_scores))),
            "test_ap": test_row["average_precision"],
            "test_ap_ci": [test_row.get("ap_ci_low"), test_row.get("ap_ci_high")],
            "test_roc_auc": test_row["roc_auc"],
            "frozen_th_1pct": th1,
            "recall_at_1pct": m_at_1["recall"],
            "actual_fpr_1pct": m_at_1["actual_fpr"],
            "tp_1pct": m_at_1["true_positives"],
            "fp_1pct": m_at_1["false_positives"],
            "frozen_th_5pct": th5,
            "recall_at_5pct": m_at_5["recall"],
            "actual_fpr_5pct": m_at_5["actual_fpr"],
            "tp_5pct": m_at_5["true_positives"],
            "fp_5pct": m_at_5["false_positives"],
        }

    # 3. Export Tables
    print("\n[3/5] Saving final test evaluation tables...")
    results_df = pd.DataFrame(model_comparison_rows)
    comparison_cols = [
        "model_name", "label", "artifact_path", "n_samples", "fraud_count", "fraud_prevalence",
        "average_precision", "ap_ci_low", "ap_ci_high", "roc_auc",
        "recall_at_1pct_fpr", "threshold_1pct_fpr", "actual_fpr_1pct", "tp_1pct_fpr", "fp_1pct_fpr",
        "alerts_1pct_fpr", "precision_1pct_fpr",
        "recall_at_5pct_fpr", "threshold_5pct_fpr", "actual_fpr_5pct", "tp_5pct_fpr", "fp_5pct_fpr",
        "alerts_5pct_fpr", "precision_5pct_fpr",
    ]
    results_df = results_df[[c for c in comparison_cols if c in results_df.columns]]
    comparison_path = os.path.join(TABLES_DIR, "final_test_model_comparison.csv")
    results_df.to_csv(comparison_path, index=False)
    print(f"  Saved: {comparison_path}")

    threshold_df = pd.DataFrame(threshold_metrics_rows)
    threshold_path = os.path.join(TABLES_DIR, "final_test_threshold_metrics.csv")
    threshold_df.to_csv(threshold_path, index=False)
    print(f"  Saved: {threshold_path}")

    capacity_df = pd.DataFrame(capacity_rows)
    capacity_path = os.path.join(TABLES_DIR, "final_test_review_capacity.csv")
    capacity_df.to_csv(capacity_path, index=False)
    print(f"  Saved: {capacity_path}")

    val_vs_test_df = pd.DataFrame(val_vs_test_rows)
    val_vs_test_path = os.path.join(TABLES_DIR, "final_test_val_vs_test_comparison.csv")
    val_vs_test_df.to_csv(val_vs_test_path, index=False)
    print(f"  Saved: {val_vs_test_path}")

    checks_path = os.path.join(TABLES_DIR, "final_test_validation_checks.json")
    with open(checks_path, "w", encoding="utf-8") as f:
        json.dump(audit_checks, f, indent=2)
    print(f"  Saved: {checks_path}")

    # 4. Generate and Save Plots
    print("\n[4/5] Generating evaluation figures...")
    pr_curve_fig_path = os.path.join(FIGURES_DIR, "final_test_precision_recall_curve.png")
    plot_test_pr_curves(curves, results_df, prevalence, pr_curve_fig_path)
    print(f"  Saved: {pr_curve_fig_path}")

    capacity_fig_path = os.path.join(FIGURES_DIR, "final_test_review_capacity.png")
    plot_review_capacity_curves(capacity_df, capacity_fig_path)
    print(f"  Saved: {capacity_fig_path}")

    # 5. Summary Printout
    print("\n[5/5] Final Test Evaluation Summary:")
    print("\n=== MODEL COMPARISON ON HELD-OUT TEST PARTITION ===")
    print(results_df[["model_name", "average_precision", "ap_ci_low", "ap_ci_high", "roc_auc",
                      "recall_at_1pct_fpr", "actual_fpr_1pct",
                      "recall_at_5pct_fpr", "actual_fpr_5pct"]].to_string(index=False))

    print("\n=== FIXED REVIEW BUDGETS (TOP-N ALERTS) ===")
    pivot_cap = capacity_df.pivot(index="budget", columns="model_name", values="frauds_caught")
    print("Frauds caught out of 75:")
    print(pivot_cap.to_string())

    print("\nPrecision at review budgets:")
    pivot_prec = capacity_df.pivot(index="budget", columns="model_name", values="precision")
    print((pivot_prec * 100).round(2).to_string())

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETED SUCCESSFULLY")
    print("=" * 80)

    return {
        "results_df": results_df,
        "threshold_df": threshold_df,
        "capacity_df": capacity_df,
        "val_vs_test_df": val_vs_test_df,
        "audit_checks": audit_checks,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Final held-out test evaluation")
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=1000,
        help="Bootstrap resamples for Average Precision 95% CI (default: 1000)",
    )
    args = parser.parse_args()
    run_evaluation(n_bootstrap=args.n_bootstrap)
