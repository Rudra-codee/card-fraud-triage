# Final Held-Out Test Evaluation Report

**Project:** Card Fraud Triage via Gaussian Mixture Density Scoring and Autoencoder Anomaly Detection  
**Report Type:** Final Benchmark & Generalization Evaluation (Task A)  
**Lead:** Member 4 — Anurag, Evaluation & Fraud Triage Lead  
**Status:** Complete, Audited & Fully Reproducible  
**Date:** October 2026  

---

## 1. Executive Summary and Research Integrity

This report documents the final, unbiased evaluation of the frozen anomaly-detection models on the chronological held-out test partition of the European card fraud dataset.

### 1.1 Strict Research-Integrity Protocol

To ensure academic and operational validity, strict research-integrity rules were enforced throughout this benchmark:
1. **Zero Retraining or Refitting:** Neither the preprocessing scaler nor any model artifact was refitted, fine-tuned, or updated using test data.
2. **Frozen Thresholds:** Operating thresholds for the 1% and 5% false positive rate (FPR) targets were derived strictly from the validation partition and frozen prior to examining test labels. Thresholds were **not** adjusted to force the test FPR to equal nominal targets.
3. **Decoupled Scoring:** Anomaly scores were generated strictly from feature matrices $X_{\text{test}}$ without access to target labels (`Class`). Label alignment was performed solely for final metric computation using stable `row_id` keys.
4. **Single-Pass Evaluation:** Models were evaluated once on the test partition. No model selection or hyperparameter tuning was conducted based on test outcomes.

### 1.2 Core Evaluation Candidates

Three frozen candidates were benchmarked:
- **Candidate A — Frozen GMM Baseline (`models/gmm_baseline.joblib`):** Single-component ($K=1$) full-covariance Gaussian density model fitted on 170,528 legitimate training rows (`reg_covar=1e-6`, `random_state=42`). Anomaly score is negative log-likelihood ($-\log p(x)$).
- **Candidate B — Frozen GMM $K=4$ Full (`models/candidates/gmm_k4_full.joblib`):** Shortlisted multi-component full-covariance model (`random_state=42`, `n_init=5`, `reg_covar=1e-6`). Score is negative log-likelihood.
- **Candidate C — Frozen Autoencoder Baseline (`models/autoencoder_baseline.pt`):** Neural reconstruction model ($29 \to 20 \to 10 \to 5 \to 10 \to 20 \to 29$) trained with early stopping on legitimate training rows. Anomaly score is per-transaction mean squared reconstruction error ($\text{MSE}$).

---

## 2. Dataset and Partition Integrity Verification

The dataset partition counts were verified against `data/splits/split_manifest.csv` and `data/processed/preprocessing_metadata.json`:

| Partition | Total Transactions | Legitimate ($y=0$) | Fraud ($y=1$) | Fraud Prevalence | Partition Role |
| :--- | ---: | ---: | ---: | ---: | :--- |
| **Train** | 170,888 | 170,528 | 360 | 0.2107% | Parameter fitting ($y=0$ only) |
| **Validation** | 56,957 | 56,900 | 57 | 0.1001% | Model selection & threshold freezing |
| **Test (Held-Out)** | 56,962 | 56,887 | 75 | 0.1317% | Final one-time unbiased evaluation |
| **Total** | 284,807 | 284,315 | 492 | 0.1728% | Chronological split (60% / 20% / 20%) |

### Data Integrity Audit Checks:
- **Manifest Conformance:** All 56,962 test row IDs match the split manifest exactly with zero missing or duplicate IDs.
- **Partition Disjointness:** Zero overlap exists between train, validation, and test row IDs ($\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$).
- **Feature Matrix:** Exactly 29 features in verified order: $V_1, V_2, \dots, V_{28}$ and $\log(1 + \text{Amount})$ standardized by the training-fitted `StandardScaler`.
- **Finiteness:** All input features and output anomaly scores are strictly finite (no NaN, $+\infty$, or $-\infty$).

---

## 3. Threshold Provenance (Frozen from Validation)

Operating thresholds were determined on the validation partition using `src/triage_evaluation.py::recall_at_fpr` by selecting the lowest threshold keeping legitimate false positive rate $\le \text{target}$. These thresholds were frozen and transferred directly to the test set:

| Model | Operating Point | Frozen Threshold Value | Validation Achieved FPR | Validation Frauds Caught | Score Metric |
| :--- | :--- | ---: | ---: | ---: | :--- |
| **GMM Baseline** | 1% Target FPR | **158.8138** | 0.9947% (566 / 56,900) | 42 / 57 (73.68%) | Negative Log-Likelihood |
| **GMM Baseline** | 5% Target FPR | **60.7258** | 5.0000% (2,845 / 56,900) | 50 / 57 (87.72%) | Negative Log-Likelihood |
| **GMM $K=4$ Full** | 1% Target FPR | **66.4360** | 0.9965% (567 / 56,900) | 43 / 57 (75.44%) | Negative Log-Likelihood |
| **GMM $K=4$ Full** | 5% Target FPR | **50.2414** | 5.0000% (2,845 / 56,900) | 47 / 57 (82.46%) | Negative Log-Likelihood |
| **Autoencoder Baseline** | 1% Target FPR | **2.6511** | 1.0000% (569 / 56,900) | 41 / 57 (71.93%) | Reconstruction MSE |
| **Autoencoder Baseline** | 5% Target FPR | **1.3411** | 5.0000% (2,845 / 56,900) | 50 / 57 (87.72%) | Reconstruction MSE |

*Note:* Thresholds differ across models due to distinct score distributions and units (NLL vs. MSE), emphasizing the necessity of model-specific threshold freezing.

---

## 4. Final Held-Out Test Evaluation Results

### 4.1 Summary Metric Comparison

All metrics were computed on the held-out test partition ($N = 56,962$, 75 frauds):

| Metric | GMM Baseline ($K=1$) | GMM $K=4$ Full | Autoencoder Baseline |
| :--- | :---: | :---: | :---: |
| **Model Artifact** | `models/gmm_baseline.joblib` | `models/candidates/gmm_k4_full.joblib` | `models/autoencoder_baseline.pt` |
| **Average Precision (AP)** | **0.1610** | **0.5770** | **0.1793** |
| └ 95% Bootstrap CI | [0.1208, 0.2209] | [0.4588, 0.6973] | [0.1318, 0.2447] |
| **ROC-AUC** | **0.9622** | **0.9156** | **0.9436** |
| **Recall @ Frozen 1% Val Thresh** | **72.00% (54 / 75)** | **76.00% (57 / 75)** | **73.33% (55 / 75)** |
| └ Actual Test FPR | 0.6645% (378 FP) | 0.5889% (335 FP) | 0.5590% (318 FP) |
| └ Total Alerts & Precision | 432 alerts (12.50%) | 392 alerts (14.54%) | 373 alerts (14.75%) |
| **Recall @ Frozen 5% Val Thresh** | **84.00% (63 / 75)** | **80.00% (60 / 75)** | **80.00% (60 / 75)** |
| └ Actual Test FPR | 4.0132% (2,283 FP) | 3.6845% (2,096 FP) | 3.6019% (2,049 FP) |
| └ Total Alerts & Precision | 2,346 alerts (2.69%) | 2,156 alerts (2.78%) | 2,109 alerts (2.84%) |

### 4.2 Fixed Operational Review Capacity (Top-$N$ Alerts)

When human fraud investigation capacity is constrained to a fixed daily alert quota:

| Review Budget | Metric | GMM Baseline | GMM $K=4$ Full | Autoencoder Baseline |
| :--- | :--- | :---: | :---: | :---: |
| **Top 50 Alerts** | Frauds Caught | 9 / 75 | **39 / 75** | 11 / 75 |
| | Precision | 18.00% | **78.00%** | 22.00% |
| | Recall | 12.00% | **52.00%** | 14.67% |
| | False Positives | 41 | **11** | 39 |
| **Top 100 Alerts** | Frauds Caught | 28 / 75 | **49 / 75** | 31 / 75 |
| | Precision | 28.00% | **49.00%** | 31.00% |
| | Recall | 37.33% | **65.33%** | 41.33% |
| | False Positives | 72 | **51** | 69 |
| **Top 200 Alerts** | Frauds Caught | 38 / 75 | **55 / 75** | 40 / 75 |
| | Precision | 19.00% | **27.50%** | 20.00% |
| | Recall | 50.67% | **73.33%** | 53.33% |
| | False Positives | 162 | **145** | 160 |
| **Top 500 Alerts** | Frauds Caught | 56 / 75 | **57 / 75** | 56 / 75 |
| | Precision | 11.20% | **11.40%** | 11.20% |
| | Recall | 74.67% | **76.00%** | 74.67% |
| | False Positives | 444 | **443** | 444 |

---

## 5. Validation vs. Test Generalization Analysis

### 5.1 Partition Comparison Table

| Model | Split | Fraud Prevalence | Average Precision | Recall @ 1% Thresh | Actual FPR @ 1% | Recall @ 5% Thresh | Actual FPR @ 5% |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **GMM Baseline** | Validation | 0.1001% (57/56,957) | 0.0740 | 73.68% (42/57) | 0.9947% | 87.72% (50/57) | 5.0000% |
| | **Test** | **0.1317% (75/56,962)** | **0.1610** | **72.00% (54/75)** | **0.6645%** | **84.00% (63/75)** | **4.0132%** |
| **GMM $K=4$ Full** | Validation | 0.1001% (57/56,957) | 0.4840 | 75.44% (43/57) | 0.9965% | 82.46% (47/57) | 5.0000% |
| | **Test** | **0.1317% (75/56,962)** | **0.5770** | **76.00% (57/75)** | **0.5889%** | **80.00% (60/75)** | **3.6845%** |
| **Autoencoder** | Validation | 0.1001% (57/56,957) | 0.0765 | 71.93% (41/57) | 1.0000% | 87.72% (50/57) | 5.0000% |
| | **Test** | **0.1317% (75/56,962)** | **0.1793** | **73.33% (55/75)** | **0.5590%** | **80.00% (60/75)** | **3.6019%** |

### 5.2 Key Observations

1. **Increase in Average Precision Across Models:**  
   Average precision was higher on the held-out test partition than on validation across all three models:
   - GMM Baseline: $0.0740 \to 0.1610$ (+117.6% relative)
   - Autoencoder: $0.0765 \to 0.1793$ (+134.4% relative)
   - GMM $K=4$ Full: $0.4840 \to 0.5770$ (+19.2% relative)  
   The test partition also had a higher fraud prevalence (0.1317% vs. 0.1001%, a +31.6% relative increase). While differences in prevalence and the ranking of individual fraud transactions may contribute to the observed change, the available analysis does not establish a causal explanation or confirm formal fraud clustering.

2. **Operating Point Conservatism:**  
   When validation thresholds were applied to the test partition, the actual test false positive rates were consistently lower than nominal targets:
   - At the nominal 1% FPR operating point, actual test FPRs were **0.56% to 0.66%** (~35–44% fewer false alerts).
   - At the nominal 5% FPR operating point, actual test FPRs were **3.60% to 4.01%** (~20–28% fewer false alerts).  
   This confirms that validation-derived thresholds provided safe, non-exploding alert rates on this unseen test partition.

3. **Did the GMM $K=4$ Advantage Hold?**  
   **Yes, for this specific frozen candidate artifact.** The saved seed-42 $K=4$ model maintained a dominant AP advantage ($0.5770$ vs. $0.1610$ for baseline and $0.1793$ for AE). Under tight alert constraints (Top 50 alerts), $K=4$ achieved **78.00% precision (39/50)** compared to 18.00% (9/50) for the baseline and 22.00% (11/50) for the autoencoder.  
   *Critical Research Integrity Caveat:* As documented in the model-selection stability audit (`reports/gmm_model_selection.md` and `results/tables/triage_stability_check.csv`), multi-component full-covariance GMMs exhibit substantial seed sensitivity due to non-convex EM likelihood surfaces. For $K=4$ full covariance, alternative random seeds 0–4 produced validation AP values from approximately 0.069 to 0.102, while the saved seed-42 artifact achieved validation AP of approximately 0.484. The broader range of approximately 0.045–0.059 observed during model selection applies to diagonal-covariance candidates rather than the $K=4$ full-covariance seed comparison. While the specific saved seed-42 artifact generalized effectively on this test partition (0.5770 AP), its performance is sensitive to the specific covariance mode identified by seed 42, and retraining with a different seed should not be assumed to yield equivalent performance.

4. **Baseline Recall at Wider Operating Budgets:**  
   At the frozen 5% FPR threshold (~2,100–2,300 alerts), the single-component GMM Baseline intercepted **63 of 75 frauds (84.00%)**, outperforming both $K=4$ (60/75, 80.00%) and the Autoencoder (60/75, 80.00%). The unimodal baseline's smooth quadratic distance penalty ensures that severe multivariate outliers are flagged, avoiding component-weight trade-offs that can assign intermediate likelihoods to unusual profiles.

5. **Autoencoder vs. GMM Baseline:**  
   The Autoencoder achieved a slightly higher test AP than the GMM baseline (0.1793 vs. 0.1610) and caught more frauds at Top 50 (11 vs. 9) and Top 100 (31 vs. 28). However, their 95% bootstrap confidence intervals overlap heavily ([0.1318, 0.2447] vs. [0.1208, 0.2209]). The GMM baseline requires only 464 parameters and trains in ~1.2 s, compared to 1,929 parameters and ~10.2 s for the Autoencoder.

---

## 6. Operational Recommendations for Triage

Based on the empirical test results across operating budgets, a practical operational deployment distinguishes between **dynamic review capacity policies** and **fixed score threshold policies**:

```
[Incoming Real-Time Transactions]
           │
           ▼
┌────────────────────────────────────────────────────────────────────────┐
│  Tier 1: High-Precision Rapid Interception                             │
│  Model: Frozen GMM K=4 Full Covariance Artifact                        │
│  Policy Type: Dynamic Review Capacity (Top 50–100 alerts per cycle)    │
│  Validation Calibration Reference: τ₁₀₀ = 135.78 (Top-100 cut-off)     │
│  Observed Test Set Cutoff at Rank 100: Score = 143.26 (Descriptive)    │
│  Test Yield: 49.00% to 78.00% Precision (39 to 49 frauds caught)       │
│  Action: Immediate challenge / step-up cardholder verification         │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│  Tier 2: Broad High-Recall Safety Net                                  │
│  Model: Frozen GMM Baseline (K=1 full covariance)                      │
│  Policy Type: Fixed Validation-Derived Threshold                       │
│  Operating Threshold: Score >= 60.726 (Frozen 5% validation-FPR point) │
│  Test Yield: 84.00% Total Fraud Recall (63/75 frauds intercepted)      │
│  Realized Test FPR: 4.01% (Safe operational analyst workload)          │
│  Action: Queued for secondary offline analyst review                   │
└────────────────────────────────────────────────────────────────────────┘
```

> **Threshold Provenance Note:**  
> The score $143.26$ represents the empirical score of the 100th-ranked transaction observed on the held-out test partition (`final_test_review_capacity.csv`). It is a retrospective descriptive statistic of that test ranking, not a pre-selected operational threshold. The prospective validation-derived Top-100 threshold was $\tau_{100} = 135.78$ (`reports/triage_evaluation.md`). For production deployment, Tier 1 is designed as a capacity-constrained dynamic ranking queue (Top-N alerts per period) rather than a static score cutoff.

---

## 7. Limitations and Scope

1. **Single Chronological Holdout:** The test set represents the subsequent chronological period in a single European card dataset. While chronological holdout prevents temporal data leakage, it does not evaluate multi-issuer or cross-border distribution drift.
2. **Small Fraud Sample ($N=75$):** Although the test partition contains 56,962 transactions, only 75 are fraudulent. A difference of 3 caught frauds changes recall by 4.0 percentage points.
3. **Pure Anomaly Detection Paradigm:** All models were trained purely on legitimate transactions without fraud supervision. While this satisfies the unsupervised triage setting, supervised or semi-supervised models would attain higher precision if historical fraud labels were available at training time.

---

## 8. Artifacts Generated and Verified

The following deliverables have been produced and verified:
- **Evaluation Script:** `scripts/run_final_test_evaluation.py`
- **Generated Score Files:**
  - `data/processed/scores/gmm_baseline_scores_test.csv` (56,962 rows)
  - `data/processed/scores/gmm_k4_full_scores_test.csv` (56,962 rows)
  - `data/processed/scores/autoencoder_scores_test.csv` (56,962 rows)
- **Result Tables:**
  - `results/tables/final_test_model_comparison.csv`
  - `results/tables/final_test_threshold_metrics.csv`
  - `results/tables/final_test_review_capacity.csv`
  - `results/tables/final_test_val_vs_test_comparison.csv`
  - `results/tables/final_test_validation_checks.json`
- **Result Figures:**
  - `results/figures/final_test_precision_recall_curve.png`
  - `results/figures/final_test_review_capacity.png`
- **Unit & Regression Tests:** `tests/test_final_test_evaluation.py` (61/61 passing across full test suite)
