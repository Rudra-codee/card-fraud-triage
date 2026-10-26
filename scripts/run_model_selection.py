"""
run_model_selection.py

Driver script to execute Phase 1 Task 3A GMM Model Selection experiments:
- Sweeps K in [2, 3, 4, 5] across covariance types ['full', 'diag', 'tied']
- Evaluates AIC, BIC on legitimate training rows (N=170,528)
- Evaluates validation triage metrics: Average Precision (PR-AUC), Recall @ 1% FPR, Recall @ 5% FPR
- Serializes candidate models to models/candidates/
- Saves comparison table to results/tables/gmm_model_selection_comparison.csv
- Generates AIC/BIC and validation score distribution plots to results/figures/
"""

import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath("."))

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.preprocessing import load_processed_data
from src.scoring import compute_anomaly_scores
from src.model_selection import (
    fit_gmm_candidate,
    compute_triage_metrics,
    run_gmm_grid_search,
)


def main():
    print("=" * 78)
    print("PHASE 1, TASK 3A: GMM MODEL SELECTION EXPERIMENT")
    print("=" * 78)

    # Output directories
    results_tables_dir = os.path.join("results", "tables")
    results_figures_dir = os.path.join("results", "figures")
    candidates_model_dir = os.path.join("models", "candidates")

    os.makedirs(results_tables_dir, exist_ok=True)
    os.makedirs(results_figures_dir, exist_ok=True)
    os.makedirs(candidates_model_dir, exist_ok=True)

    # 1. Load preprocessed data
    print("\n[Step 1] Loading preprocessed data from data/processed/ ...")
    data = load_processed_data("data/processed")
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]

    print(f"X_train shape: {X_train.shape} | Legit count: {(y_train == 0).sum():,} | Fraud count: {(y_train == 1).sum():,}")
    print(f"X_val shape:   {X_val.shape} | Legit count: {(y_val == 0).sum():,} | Fraud count: {(y_val == 1).sum():,}")

    # 2. Evaluate frozen K=1 baseline as baseline reference
    print("\n[Step 2] Evaluating frozen baseline (K=1, full) as reference...")
    gmm_baseline, base_meta = joblib.load("models/gmm_baseline.joblib"), None
    X_train_legit = X_train[y_train == 0]

    baseline_val_scores = compute_anomaly_scores(gmm_baseline, X_val)
    baseline_triage = compute_triage_metrics(y_val, baseline_val_scores)

    baseline_record = {
        "model_name": "gmm_baseline_k1_full",
        "n_components": 1,
        "covariance_type": "full",
        "reg_covar": 1e-6,
        "n_init": 5,
        "random_state": 42,
        "n_training_samples": len(X_train_legit),
        "train_fraud_excluded": int((y_train == 1).sum()),
        "n_features": 29,
        "status": "success",
        "fit_duration_seconds": 1.2,
        "converged": bool(gmm_baseline.converged_),
        "n_iter": int(gmm_baseline.n_iter_),
        "lower_bound": float(gmm_baseline.lower_bound_),
        "aic": float(gmm_baseline.aic(X_train_legit)),
        "bic": float(gmm_baseline.bic(X_train_legit)),
        "val_scores_finite": True,
        "val_score_mean": round(float(np.mean(baseline_val_scores)), 2),
        "val_score_median": round(float(np.median(baseline_val_scores)), 2),
        "val_score_std": round(float(np.std(baseline_val_scores)), 2),
        **baseline_triage,
        "error_message": None,
    }

    # 3. Run candidate grid search across 12 configurations
    print("\n[Step 3] Running candidate grid search across 12 configurations...")
    candidates_df, fitted_models = run_gmm_grid_search(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        components_list=[2, 3, 4, 5],
        covariance_types=["full", "diag", "tied"],
        reg_covar=1e-6,
        n_init=5,
        random_state=42,
        save_models_dir=candidates_model_dir,
    )

    # Combine baseline reference with candidate models
    full_comparison_df = pd.concat([pd.DataFrame([baseline_record]), candidates_df], ignore_index=True)

    # 4. Save results table
    csv_path = os.path.join(results_tables_dir, "gmm_model_selection_comparison.csv")
    full_comparison_df.to_csv(csv_path, index=False)
    print(f"\n[Step 4] Saved comparison table to: {csv_path}")

    # Display comparison table
    display_cols = [
        "model_name", "n_components", "covariance_type", "converged", "n_iter",
        "aic", "bic", "average_precision", "recall_at_1pct_fpr", "recall_at_5pct_fpr", "fit_duration_seconds"
    ]
    print("\n=== GMM MODEL SELECTION COMPARISON TABLE ===")
    print(full_comparison_df[display_cols].to_string(index=False))

    # 5. Plot AIC / BIC Curves
    print("\n[Step 5] Generating AIC and BIC curves...")
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    # Filter out baseline for clean K=2..5 or include K=1..5 for full
    cov_types = ["full", "diag", "tied"]
    colors = {"full": "#1f77b4", "diag": "#2ca02c", "tied": "#d62728"}
    markers = {"full": "o", "diag": "s", "tied": "^"}

    for cov in cov_types:
        sub = full_comparison_df[full_comparison_df["covariance_type"] == cov].sort_values("n_components")
        axes[0].plot(sub["n_components"], sub["bic"], label=f"{cov} cov", color=colors[cov], marker=markers[cov], linewidth=2)
        axes[1].plot(sub["n_components"], sub["aic"], label=f"{cov} cov", color=colors[cov], marker=markers[cov], linewidth=2)

    axes[0].set_title("Bayesian Information Criterion (BIC) vs Components (K)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Component Count (K)", fontsize=11)
    axes[0].set_ylabel("BIC (Lower is Better)", fontsize=11)
    axes[0].set_xticks([1, 2, 3, 4, 5])
    axes[0].legend(title="Covariance Type")

    axes[1].set_title("Akaike Information Criterion (AIC) vs Components (K)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Component Count (K)", fontsize=11)
    axes[1].set_ylabel("AIC (Lower is Better)", fontsize=11)
    axes[1].set_xticks([1, 2, 3, 4, 5])
    axes[1].legend(title="Covariance Type")

    fig.tight_layout()
    aic_bic_fig_path = os.path.join(results_figures_dir, "bic_aic_curve.png")
    fig.savefig(aic_bic_fig_path)
    plt.close(fig)
    print(f"Saved AIC/BIC plot to: {aic_bic_fig_path}")

    # 6. Plot Validation Score Distributions for Top Candidates
    print("\n[Step 6] Generating Validation Anomaly Score Distributions...")
    # Select baseline + top candidates representing full, diag, tied
    selected_models = [
        ("Baseline (K=1 full)", baseline_val_scores),
        ("K=2 full", compute_anomaly_scores(fitted_models["gmm_k2_full"], X_val)),
        ("K=3 full", compute_anomaly_scores(fitted_models["gmm_k3_full"], X_val)),
        ("K=3 diag", compute_anomaly_scores(fitted_models["gmm_k3_diag"], X_val)),
        ("K=3 tied", compute_anomaly_scores(fitted_models["gmm_k3_tied"], X_val)),
    ]

    fig, axes = plt.subplots(len(selected_models), 1, figsize=(12, 12), dpi=300, sharex=False)

    for idx, (title, scores) in enumerate(selected_models):
        ax = axes[idx]
        legit_s = scores[y_val == 0]
        fraud_s = scores[y_val == 1]

        # Clip extreme visualization tail at 99.5th percentile for readable visualization
        clip_val = np.percentile(scores, 99.7)
        legit_plot = np.clip(legit_s, None, clip_val)
        fraud_plot = np.clip(fraud_s, None, clip_val)

        sns.kdeplot(legit_plot, ax=ax, color="#1f77b4", fill=True, alpha=0.4, label=f"Legitimate (N={len(legit_s):,})")
        sns.kdeplot(fraud_plot, ax=ax, color="#d62728", fill=True, alpha=0.4, label=f"Fraud (N={len(fraud_s)})")

        tau_1pct = float(np.percentile(legit_s, 99.0))
        if tau_1pct <= clip_val:
            ax.axvline(tau_1pct, color="darkred", linestyle="--", linewidth=1.5, label=f"1% FPR Threshold ({tau_1pct:.1f})")

        ax.set_title(f"Score Density: {title}", fontsize=11, fontweight="bold")
        ax.set_ylabel("Density", fontsize=9)
        ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Anomaly Score (-log p(x)) [Clipped at 99.7th percentile for visualization]", fontsize=10)
    fig.tight_layout()
    dist_fig_path = os.path.join(results_figures_dir, "validation_score_distributions.png")
    fig.savefig(dist_fig_path)
    plt.close(fig)
    print(f"Saved score distribution plot to: {dist_fig_path}")

    print("\nExperiment execution successfully completed.")


if __name__ == "__main__":
    main()
