"""
run_autoencoder.py

Full training, scoring, artifact generation, and validation-only evaluation
for the Autoencoder anomaly detection model (Task 3B).
"""

import json
import os
import time
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from src.autoencoder_model import (
    DEFAULT_AE_CONFIG,
    fit_autoencoder,
    save_autoencoder_model,
)
from src.autoencoder_scoring import compute_autoencoder_scores
from src.scoring import create_score_dataframe, save_scores, summarize_scores
from src.triage_evaluation import (
    load_validation_data,
    align_scores_with_labels,
    recall_at_fpr,
)


def run_full_autoencoder_pipeline():
    print("=" * 70)
    print("TASK 3B: AUTOENCODER FULL TRAINING & VALIDATION EVALUATION")
    print("=" * 70)

    # 1. Load Training and Validation Data (WITHOUT loading test data)
    print("\n[1/5] Loading preprocessed train and validation datasets...")
    X_train = np.load("data/processed/X_train.npy")
    y_train = np.load("data/processed/y_train.npy")
    row_ids_train = np.load("data/processed/row_ids_train.npy")

    val_dict = load_validation_data("data/processed")
    X_val = val_dict["X_val"]
    y_val = val_dict["y_val"]
    row_ids_val = val_dict["row_ids_val"]

    with open("data/processed/feature_names.json", "r") as f:
        feature_names = json.load(f)

    print(f"  X_train shape: {X_train.shape} (Legit: {(y_train == 0).sum():,}, Fraud: {(y_train == 1).sum():,})")
    print(f"  X_val shape:   {X_val.shape} (Legit: {(y_val == 0).sum():,}, Fraud: {(y_val == 1).sum():,})")
    print(f"  Features ({len(feature_names)}): {feature_names[:3]} ... {feature_names[-1]}")

    # 2. Approved Training Configuration
    config = {
        "input_dim": 29,
        "bottleneck_dim": 5,
        "learning_rate": 0.001,
        "batch_size": 256,
        "epochs": 30,
        "patience": 5,
        "min_delta": 1e-4,
        "random_state": 42,
    }

    print("\n[2/5] Training Autoencoder strictly on legitimate training rows (y_train == 0)...")
    print("  Monitoring MSE reconstruction loss on legitimate validation rows (y_val == 0)...")
    t0 = time.time()
    model, metadata = fit_autoencoder(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        config=config,
        feature_names=feature_names,
    )
    train_duration = time.time() - t0
    metadata["training_duration_seconds"] = train_duration

    print(f"  Training finished in {train_duration:.2f}s")
    print(f"  Epochs trained: {metadata['epochs_trained']} | Best epoch: {metadata['best_epoch']}")
    print(f"  Best legitimate validation MSE: {metadata['best_monitoring_loss']:.6f}")

    # 3. Save Model Artifacts
    print("\n[3/5] Saving model weights and metadata...")
    model_path, meta_path = save_autoencoder_model(
        model=model,
        metadata=metadata,
        output_dir="models",
        model_basename="autoencoder_baseline",
    )
    print(f"  Saved weights:  {model_path}")
    print(f"  Saved metadata: {meta_path}")

    # 4. Generate Anomaly Scores
    print("\n[4/5] Computing MSE reconstruction anomaly scores...")
    train_scores = compute_autoencoder_scores(model, X_train)
    val_scores = compute_autoencoder_scores(model, X_val)

    # Validate finite, non-negative, and shape
    assert len(train_scores) == len(X_train)
    assert len(val_scores) == len(X_val)
    assert np.isfinite(train_scores).all() and (train_scores >= 0.0).all()
    assert np.isfinite(val_scores).all() and (val_scores >= 0.0).all()

    # Package into DataFrames
    df_train_scores = create_score_dataframe(row_ids=row_ids_train, scores=train_scores, split="train", labels=y_train)
    df_val_scores = create_score_dataframe(row_ids=row_ids_val, scores=val_scores, split="val", labels=y_val)

    train_score_path = "data/processed/scores/autoencoder_scores_train.csv"
    val_score_path = "data/processed/scores/autoencoder_scores_val.csv"

    save_scores(df_train_scores, train_score_path)
    save_scores(df_val_scores, val_score_path)
    print(f"  Saved train scores: {train_score_path} ({len(df_train_scores):,} rows)")
    print(f"  Saved val scores:   {val_score_path} ({len(df_val_scores):,} rows)")

    # Diagnostic score statistics
    train_stats = summarize_scores(train_scores, y_train)
    val_stats = summarize_scores(val_scores, y_val)

    print("\n  Score Distribution (Validation):")
    print(f"    Legit mean: {val_stats['legitimate']['mean']:.4f} | median: {val_stats['legitimate']['median']:.4f} | max: {val_stats['legitimate']['max']:.4f}")
    print(f"    Fraud mean: {val_stats['fraud']['mean']:.4f} | median: {val_stats['fraud']['median']:.4f} | max: {val_stats['fraud']['max']:.4f}")


    # 5. Evaluate on Validation Set Only
    print("\n[5/5] Evaluating triage performance on validation partition...")
    aligned_val = align_scores_with_labels(df_val_scores, row_ids_val, y_val)
    ae_scores = aligned_val["anomaly_score"].to_numpy()

    ae_ap = average_precision_score(y_val, ae_scores)
    ae_recall_1pct = recall_at_fpr(y_val, ae_scores, max_fpr=0.01)
    ae_recall_5pct = recall_at_fpr(y_val, ae_scores, max_fpr=0.05)

    print(f"  Autoencoder Average Precision (PR-AUC): {ae_ap:.4f}")
    print(f"  Autoencoder Recall @ 1% FPR:           {ae_recall_1pct['recall'] * 100:.2f}% ({ae_recall_1pct['true_positives']}/{int((y_val == 1).sum())} frauds caught)")
    print(f"    Threshold @ 1% FPR:                  {ae_recall_1pct['threshold']:.6f} (actual FPR: {ae_recall_1pct['actual_fpr']*100:.3f}%)")
    print(f"  Autoencoder Recall @ 5% FPR:           {ae_recall_5pct['recall'] * 100:.2f}% ({ae_recall_5pct['true_positives']}/{int((y_val == 1).sum())} frauds caught)")
    print(f"    Threshold @ 5% FPR:                  {ae_recall_5pct['threshold']:.6f} (actual FPR: {ae_recall_5pct['actual_fpr']*100:.3f}%)")

    # Load frozen GMM baseline validation scores for direct comparison
    gmm_val_df = pd.read_csv("data/processed/scores/gmm_baseline_scores_val.csv")
    aligned_gmm = align_scores_with_labels(gmm_val_df, row_ids_val, y_val)
    gmm_scores = aligned_gmm["anomaly_score"].to_numpy()

    gmm_ap = average_precision_score(y_val, gmm_scores)
    gmm_recall_1pct = recall_at_fpr(y_val, gmm_scores, max_fpr=0.01)
    gmm_recall_5pct = recall_at_fpr(y_val, gmm_scores, max_fpr=0.05)

    print("\n" + "=" * 70)
    print("COMPARISON: AUTOENCODER VS. FROZEN GMM BASELINE (VALIDATION SET)")
    print("=" * 70)
    print(f"{'Metric':<28} | {'Frozen GMM Baseline':<20} | {'Autoencoder Baseline':<20}")
    print("-" * 74)
    print(f"{'PR-AUC (Average Precision)':<28} | {gmm_ap:<20.4f} | {ae_ap:<20.4f}")
    print(f"{'Recall @ 1% FPR':<28} | {gmm_recall_1pct['recall']*100:.2f}% ({gmm_recall_1pct['true_positives']}/57)     | {ae_recall_1pct['recall']*100:.2f}% ({ae_recall_1pct['true_positives']}/57)")
    print(f"{'Recall @ 5% FPR':<28} | {gmm_recall_5pct['recall']*100:.2f}% ({gmm_recall_5pct['true_positives']}/57)     | {ae_recall_5pct['recall']*100:.2f}% ({ae_recall_5pct['true_positives']}/57)")
    print(f"{'Parameters':<28} | {'464 (full cov)':<20} | {'1,929':<20}")
    print("=" * 70)

    # Return summary dict
    return {
        "metadata": metadata,
        "ae_metrics": {
            "ap": ae_ap,
            "recall_1pct": ae_recall_1pct,
            "recall_5pct": ae_recall_5pct,
            "val_stats": val_stats,
            "train_stats": train_stats,
        },
        "gmm_metrics": {
            "ap": gmm_ap,
            "recall_1pct": gmm_recall_1pct,
            "recall_5pct": gmm_recall_5pct,
        },
    }


if __name__ == "__main__":
    run_full_autoencoder_pipeline()
