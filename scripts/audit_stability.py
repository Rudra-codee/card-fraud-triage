"""
audit_stability.py

Comprehensive numerical stability and result-integrity audit for Phase 1 Task 3B:
1. Artifact verification and metric recalculation across all 12 candidate models + baseline.
2. Covariance eigenvalue spectra, condition numbers, and component weight analysis.
3. Deep-dive into gmm_k4_full and gmm_k5_full (component density concentration, log-determinants).
4. Validation AP verification, PR curve inspection, and empirical FPR calculation.
5. Git tracking and file verification.
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))

import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

from src.preprocessing import load_processed_data
from src.scoring import compute_anomaly_scores
from src.model_selection import compute_triage_metrics


def audit_candidate_models():
    print("=" * 80)
    print("PHASE 1, TASK 3B: NUMERICAL STABILITY & RESULT INTEGRITY AUDIT")
    print("=" * 80)

    # 1. Load data
    data = load_processed_data("data/processed")
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    row_ids_val = data["row_ids_val"]

    X_train_legit = X_train[y_train == 0]
    num_train_legit = len(X_train_legit)
    num_val = len(X_val)
    num_val_legit = int((y_val == 0).sum())
    num_val_fraud = int((y_val == 1).sum())

    print(f"Loaded datasets:")
    print(f"  Training legitimate rows: {num_train_legit:,}")
    print(f"  Validation rows:          {num_val:,} ({num_val_legit:,} legit, {num_val_fraud} fraud)")

    # 2. Load comparison table from Task 3A
    csv_path = os.path.join("results", "tables", "gmm_model_selection_comparison.csv")
    if not os.path.exists(csv_path):
        print(f"ERROR: Comparison table not found at {csv_path}")
        return
    reported_df = pd.read_csv(csv_path)

    # 3. Model list
    model_configs = [
        ("gmm_baseline_k1_full", "models/gmm_baseline.joblib"),
    ]
    for k in [2, 3, 4, 5]:
        for cov in ["full", "diag", "tied"]:
            mname = f"gmm_k{k}_{cov}"
            mpath = f"models/candidates/{mname}.joblib"
            model_configs.append((mname, mpath))

    # Audit records
    audit_rows = []
    detailed_components = {}

    print("\n--- TASK 1: RECALCULATING METRICS & TESTING ARTIFACT INTEGRITY ---")

    for mname, mpath in model_configs:
        if not os.path.exists(mpath):
            print(f"ERROR: Model file missing: {mpath}")
            continue

        gmm = joblib.load(mpath)
        k = gmm.n_components
        cov_type = gmm.covariance_type

        # Verify parameter count
        n_features = gmm.n_features_in_
        if cov_type == "full":
            expected_p = (k - 1) + k * n_features + k * n_features * (n_features + 1) // 2
        elif cov_type == "diag":
            expected_p = (k - 1) + k * n_features + k * n_features
        elif cov_type == "tied":
            expected_p = (k - 1) + k * n_features + n_features * (n_features + 1) // 2

        actual_p = gmm._n_parameters()

        # Recalculate training log-likelihood and AIC/BIC
        recalc_train_ll_lower = float(gmm.lower_bound_)
        recalc_aic = float(gmm.aic(X_train_legit))
        recalc_bic = float(gmm.bic(X_train_legit))

        # Recalculate validation anomaly scores
        recalc_val_scores = compute_anomaly_scores(gmm, X_val)
        val_scores_finite = bool(np.isfinite(recalc_val_scores).all())

        # Recalculate validation triage metrics
        legit_val_scores = recalc_val_scores[y_val == 0]
        fraud_val_scores = recalc_val_scores[y_val == 1]

        recalc_ap = float(average_precision_score(y_val, recalc_val_scores))

        tau_1pct = float(np.percentile(legit_val_scores, 99.0))
        tau_5pct = float(np.percentile(legit_val_scores, 95.0))

        # Empirical achieved FPR
        achieved_fpr_1pct = float((legit_val_scores >= tau_1pct).mean())
        achieved_fpr_5pct = float((legit_val_scores >= tau_5pct).mean())

        recalc_r_1pct = float((fraud_val_scores >= tau_1pct).mean())
        recalc_r_5pct = float((fraud_val_scores >= tau_5pct).mean())

        # Covariance analysis
        weights = gmm.weights_
        min_weight = float(weights.min())
        max_weight = float(weights.max())
        min_eff_samples = int(round(min_weight * num_train_legit))

        # Eigenvalues
        eig_min_list = []
        eig_max_list = []
        cond_list = []
        log_det_list = []

        if cov_type == "full":
            for c_idx in range(k):
                cov_c = gmm.covariances_[c_idx]
                eigs = np.linalg.eigvalsh(cov_c)
                e_min = float(eigs.min())
                e_max = float(eigs.max())
                eig_min_list.append(e_min)
                eig_max_list.append(e_max)
                cond_list.append(e_max / e_min if e_min > 0 else np.inf)
                sign, lndet = np.linalg.slogdet(cov_c)
                log_det_list.append(float(lndet))
        elif cov_type == "diag":
            for c_idx in range(k):
                diag_c = gmm.covariances_[c_idx]
                e_min = float(diag_c.min())
                e_max = float(diag_c.max())
                eig_min_list.append(e_min)
                eig_max_list.append(e_max)
                cond_list.append(e_max / e_min if e_min > 0 else np.inf)
                log_det_list.append(float(np.sum(np.log(diag_c))))
        elif cov_type == "tied":
            cov_c = gmm.covariances_
            eigs = np.linalg.eigvalsh(cov_c)
            e_min = float(eigs.min())
            e_max = float(eigs.max())
            eig_min_list.append(e_min)
            eig_max_list.append(e_max)
            cond_list.append(e_max / e_min if e_min > 0 else np.inf)
            sign, lndet = np.linalg.slogdet(cov_c)
            log_det_list.append(float(lndet))

        overall_eig_min = float(np.min(eig_min_list))
        overall_eig_max = float(np.max(eig_max_list))
        overall_cond_max = float(np.max(cond_list))
        overall_log_det_min = float(np.min(log_det_list))
        overall_log_det_max = float(np.max(log_det_list))

        # Check against reported values
        rep_row = reported_df[reported_df["model_name"] == mname].iloc[0]
        aic_diff = abs(recalc_aic - rep_row["aic"])
        bic_diff = abs(recalc_bic - rep_row["bic"])
        ap_diff = abs(recalc_ap - rep_row["average_precision"])
        r1_diff = abs(recalc_r_1pct - rep_row["recall_at_1pct_fpr"])
        r5_diff = abs(recalc_r_5pct - rep_row["recall_at_5pct_fpr"])

        audit_rows.append({
            "model_name": mname,
            "K": k,
            "cov": cov_type,
            "params_expected": expected_p,
            "params_actual": actual_p,
            "param_match": (expected_p == actual_p),
            "recalc_ll_lower": recalc_train_ll_lower,
            "recalc_aic": recalc_aic,
            "recalc_bic": recalc_bic,
            "aic_matches_reported": (aic_diff < 1e-2),
            "bic_matches_reported": (bic_diff < 1e-2),
            "recalc_ap": recalc_ap,
            "ap_matches_reported": (ap_diff < 1e-4),
            "recalc_r1": recalc_r_1pct,
            "r1_matches_reported": (r1_diff < 1e-4),
            "recalc_r5": recalc_r_5pct,
            "r5_matches_reported": (r5_diff < 1e-4),
            "tau_1pct": tau_1pct,
            "achieved_fpr_1pct": achieved_fpr_1pct,
            "tau_5pct": tau_5pct,
            "achieved_fpr_5pct": achieved_fpr_5pct,
            "min_weight": min_weight,
            "max_weight": max_weight,
            "min_eff_samples": min_eff_samples,
            "overall_eig_min": overall_eig_min,
            "ratio_eigmin_regcovar": overall_eig_min / 1e-6,
            "overall_eig_max": overall_eig_max,
            "overall_cond_max": overall_cond_max,
            "log_det_min": overall_log_det_min,
            "log_det_max": overall_log_det_max,
        })

        detailed_components[mname] = {
            "weights": weights.tolist(),
            "eig_min": eig_min_list,
            "eig_max": eig_max_list,
            "cond": cond_list,
            "log_det": log_det_list,
        }

    audit_df = pd.DataFrame(audit_rows)

    # Print Task 1 summary
    print("\n--- METRIC RECALCULATION & CONCORDANCE AUDIT ---")
    chk_cols = [
        "model_name", "param_match", "aic_matches_reported", "bic_matches_reported",
        "ap_matches_reported", "r1_matches_reported", "r5_matches_reported"
    ]
    print(audit_df[chk_cols].to_string(index=False))

    all_concordant = (
        audit_df["param_match"].all() and
        audit_df["aic_matches_reported"].all() and
        audit_df["bic_matches_reported"].all() and
        audit_df["ap_matches_reported"].all() and
        audit_df["r1_matches_reported"].all() and
        audit_df["r5_matches_reported"].all()
    )
    print(f"\nAll recalculated metrics match reported results exactly: {all_concordant}")

    # Print Task 2: Covariance and Eigenvalue Stability
    print("\n--- TASK 2: COVARIANCE STABILITY & CONDITION NUMBERS ---")
    stab_cols = [
        "model_name", "K", "cov", "recalc_ll_lower", "min_weight", "min_eff_samples",
        "overall_eig_min", "ratio_eigmin_regcovar", "overall_cond_max", "log_det_min", "log_det_max"
    ]
    print(audit_df[stab_cols].to_string(index=False))

    # Print Task 1: Empirical FPR breakdown
    print("\n--- EMPIRICAL FPR AT REPORTED THRESHOLDS ---")
    fpr_cols = [
        "model_name", "tau_1pct", "achieved_fpr_1pct", "recalc_r1", "tau_5pct", "achieved_fpr_5pct", "recalc_r5"
    ]
    print(audit_df[fpr_cols].to_string(index=False))

    # Task 3: Deep dive into gmm_k4_full vs gmm_k3_full and baseline
    print("\n--- TASK 3: DEEP DIVE INTO K=4 FULL vs K=3 FULL & BASELINE ---")
    gmm_k4 = joblib.load("models/candidates/gmm_k4_full.joblib")
    gmm_k3 = joblib.load("models/candidates/gmm_k3_full.joblib")
    gmm_k1 = joblib.load("models/gmm_baseline.joblib")

    scores_k4 = compute_anomaly_scores(gmm_k4, X_val)
    scores_k3 = compute_anomaly_scores(gmm_k3, X_val)
    scores_k1 = compute_anomaly_scores(gmm_k1, X_val)

    print("\nDetailed Component Breakdown for gmm_k4_full:")
    for c in range(4):
        print(f"  Component {c}:")
        print(f"    Weight:              {gmm_k4.weights_[c]:.6f} (effective training samples: {gmm_k4.weights_[c]*num_train_legit:,.0f})")
        print(f"    Min Eigenvalue:      {detailed_components['gmm_k4_full']['eig_min'][c]:.6e} (vs reg_covar 1e-6: {detailed_components['gmm_k4_full']['eig_min'][c]/1e-6:.1f}x)")
        print(f"    Max Eigenvalue:      {detailed_components['gmm_k4_full']['eig_max'][c]:.6f}")
        print(f"    Condition Number:    {detailed_components['gmm_k4_full']['cond'][c]:.2f}")
        print(f"    Covariance log-det:  {detailed_components['gmm_k4_full']['log_det'][c]:.2f}")

    print("\nDetailed Component Breakdown for gmm_k5_full:")
    gmm_k5 = joblib.load("models/candidates/gmm_k5_full.joblib")
    for c in range(5):
        print(f"  Component {c}:")
        print(f"    Weight:              {gmm_k5.weights_[c]:.6f} (effective training samples: {gmm_k5.weights_[c]*num_train_legit:,.0f})")
        print(f"    Min Eigenvalue:      {detailed_components['gmm_k5_full']['eig_min'][c]:.6e} (vs reg_covar 1e-6: {detailed_components['gmm_k5_full']['eig_min'][c]/1e-6:.1f}x)")
        print(f"    Max Eigenvalue:      {detailed_components['gmm_k5_full']['eig_max'][c]:.6f}")
        print(f"    Condition Number:    {detailed_components['gmm_k5_full']['cond'][c]:.2f}")
        print(f"    Covariance log-det:  {detailed_components['gmm_k5_full']['log_det'][c]:.2f}")

    # Inspect why AP jumped in K=4 full:
    # Look at Precision-Recall curve
    prec_k4, rec_k4, thresh_k4 = precision_recall_curve(y_val, scores_k4)
    prec_k1, rec_k1, thresh_k1 = precision_recall_curve(y_val, scores_k1)

    print("\nPrecision-Recall curve comparison at key recall points:")
    for target_r in [0.2, 0.4, 0.6, 0.7, 0.75, 0.8]:
        idx_k4 = np.argmin(np.abs(rec_k4 - target_r))
        idx_k1 = np.argmin(np.abs(rec_k1 - target_r))
        print(f"  Recall ~ {target_r:.2f}:")
        print(f"    K=4 full precision: {prec_k4[idx_k4]:.4f} (actual recall: {rec_k4[idx_k4]:.4f}, threshold: {thresh_k4[min(idx_k4, len(thresh_k4)-1)]:.2f})")
        print(f"    K=1 base precision: {prec_k1[idx_k1]:.4f} (actual recall: {rec_k1[idx_k1]:.4f}, threshold: {thresh_k1[min(idx_k1, len(thresh_k1)-1)]:.2f})")

    # Inspect top-ranked transactions for K=4 full vs K=1 baseline
    top_budgets = [50, 100, 200, 500, 570] # 570 is exactly 1% of 56,957 validation transactions
    print("\nTop-N Alert Volume Analysis (Fraud captured in top N highest scores):")
    for top_n in top_budgets:
        top_idx_k4 = np.argsort(scores_k4)[::-1][:top_n]
        top_idx_k1 = np.argsort(scores_k1)[::-1][:top_n]

        fraud_in_top_k4 = int((y_val[top_idx_k4] == 1).sum())
        fraud_in_top_k1 = int((y_val[top_idx_k1] == 1).sum())

        prec_top_k4 = fraud_in_top_k4 / top_n
        prec_top_k1 = fraud_in_top_k1 / top_n

        print(f"  Top {top_n:4d} alerts ({top_n/num_val*100:.2f}% validation volume):")
        print(f"    K=4 full: {fraud_in_top_k4:2d}/57 fraud detected | Precision@{top_n} = {prec_top_k4*100:5.2f}%")
        print(f"    K=1 base: {fraud_in_top_k1:2d}/57 fraud detected | Precision@{top_n} = {prec_top_k1*100:5.2f}%")

    # Score quantiles for K=4 full vs K=1 base
    print("\nAnomaly Score Quantiles (Validation Cohort Breakdown):")
    for name, sc in [("gmm_baseline_k1_full", scores_k1), ("gmm_k4_full", scores_k4)]:
        sc_legit = sc[y_val == 0]
        sc_fraud = sc[y_val == 1]
        print(f"  {name}:")
        print(f"    Legitimate: min={sc_legit.min():.2f}, p25={np.percentile(sc_legit, 25):.2f}, median={np.median(sc_legit):.2f}, p75={np.percentile(sc_legit, 75):.2f}, p95={np.percentile(sc_legit, 95):.2f}, p99={np.percentile(sc_legit, 99):.2f}, max={sc_legit.max():.2f}")
        print(f"    Fraud:      min={sc_fraud.min():.2f}, p25={np.percentile(sc_fraud, 25):.2f}, median={np.median(sc_fraud):.2f}, p75={np.percentile(sc_fraud, 75):.2f}, p95={np.percentile(sc_fraud, 95):.2f}, p99={np.percentile(sc_fraud, 99):.2f}, max={sc_fraud.max():.2f}")

    # Save detailed audit tables to results/tables/
    audit_output_path = os.path.join("results", "tables", "gmm_stability_audit.csv")
    audit_df.to_csv(audit_output_path, index=False)
    print(f"\nSaved stability audit table to: {audit_output_path}")

    # Check Git Tracking
    print("\n--- TASK 4: GIT STATUS & IGNORED FILES CHECK ---")
    with open(".gitignore") as f:
        gitignore_content = f.read()
    print("Relevant .gitignore entries:")
    for line in gitignore_content.splitlines():
        if "results" in line or "venv" in line:
            print(f"  {line}")


if __name__ == "__main__":
    audit_candidate_models()
