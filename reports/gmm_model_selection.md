# Phase 1: Gaussian Mixture Model (GMM) Model Selection Report

**Project:** Card Fraud Triage via Gaussian Mixture Density Scoring and Autoencoder Anomaly Detection  
**Phase:** Phase 1 — Gaussian Mixture Model (GMM) Anomaly Detection  
**Task:** Task 3 — GMM Model Selection, Numerical Stability Audit, and Candidate Handoff  
**Lead:** Member 3 — GMM Model Selection Lead (Kaustubh)  
**Status:** Completed & Audited — Ready for Member 4 (Triage Evaluation)  
**Date:** October 2026  

---

## 1. Executive Summary & Experiment Scope

This report details the systematic model-selection experiment for the Card Fraud Triage project. Building upon the frozen single-component baseline ($K=1$, full covariance), we evaluated 12 candidate Gaussian Mixture Models across:
* **Component Counts ($K$):** $K \in \{2, 3, 4, 5\}$
* **Covariance Structures:** `'full'` (unconstrained pairwise covariance), `'diag'` (axis-aligned independent variances), and `'tied'` (equal covariance matrix shared across all mixture components).

Every candidate was trained strictly on **legitimate transactions** ($y_{\text{train}} == 0$) from the chronological training partition. We evaluated candidates using a multi-dimensional protocol:
1. **Unsupervised Density Estimation Quality:** Akaike Information Criterion (AIC) and Bayesian Information Criterion (BIC) on legitimate training transactions.
2. **Prospective Validation Triage Ranking:** Average Precision (PR-AUC), Recall @ 1% False Positive Rate (FPR), and Recall @ 5% False Positive Rate (FPR) on unseen validation transactions ($N = 56,957$, containing 57 fraud cases).
3. **Numerical Conditioning & Stability Audit:** Eigenvalue spectra, covariance condition numbers, component weights, and regularization sensitivity across $10^{-6} \le \text{reg\_covar} \le 10^{-1}$.
4. **Data Hygiene & Test Set Isolation:** Zero test set leakage. Test features and labels were strictly never loaded, evaluated, or referenced.

---

## 2. Dataset Partitioning & Leak-Free Fitting Protocol

All experiments operate on the frozen, leak-free preprocessed data generated in Phase 0:

### 1. Chronological Strategy B Partitioning
* **Train Partition (60%):** 170,888 total transactions (170,528 legitimate, 360 fraud).
* **Validation Partition (20%):** 56,957 total transactions (56,900 legitimate, 57 fraud).
* **Test Partition (20%):** 56,962 total transactions (56,887 legitimate, 75 fraud) — **Completely Isolated**.
* **Total Transactions:** 284,807 transactions (preserving all 492 fraud cases).

### 2. Feature Schema & Scaling
* **29 Model Features:** $V_1$ through $V_{28}$ (PCA components) plus transaction `Amount`.
* **Amount Transformation:** $x' = \ln(1 + x)$ (`np.log1p(Amount.astype(float))`).
* **Strict Exclusions:** `Time` (non-stationary time counter), `Class` (ground-truth label), and `row_id` (tracking index) are excluded from all model feature matrices.
* **Leak-Free StandardScaler:** Fitted strictly on the 170,528 legitimate training rows (`y_train == 0`). Scaler was never refitted.
* **Candidate Fitting Subset:** Exactly 170,528 rows. All 360 training fraud transactions were strictly filtered out before calling `gmm.fit()`.

---

## 3. Official 12-Model Grid Results & Comparison Table

All 12 candidate models and the baseline reference were trained with standardized hyperparameters:
$$\text{reg\_covar} = 10^{-6}, \quad n\_\text{init} = 5, \quad \text{random\_state} = 42$$

Validation anomaly scores are defined as negative log-likelihood: $\mathrm{Score}(x) = -\ln p(x \mid \hat{\theta}) = -\mathrm{gmm.score\_samples}(X_{\mathrm{val}})$. Higher scores denote higher anomaly severity.

| Model Identifier | $K$ | Covariance Type | Free Parameters ($p$) | AIC | BIC | Average Precision (PR-AUC) | Recall @ 1% FPR | Recall @ 5% FPR | $\tau_{1\%}$ Threshold | $\tau_{5\%}$ Threshold | Fit Time (s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `gmm_baseline_k1_full` | 1 | `full` | 464 | 13,747,068.7 | 13,751,730.4 | 0.0740 | 73.68% (42/57) | 87.72% (50/57) | 158.73 | 60.72 | 1.20 |
| `gmm_k2_full` | 2 | `full` | 929 | 5,963,565.2 | 5,972,898.6 | 0.0750 | 75.44% (43/57) | **89.47% (51/57)** | 88.69 | 55.42 | 5.73 |
| `gmm_k2_diag` | 2 | `diag` | 88 | 11,967,086.1 | 11,968,261.6 | 0.0455 | 57.89% (33/57) | 85.96% (49/57) | 78.42 | 55.21 | 1.38 |
| `gmm_k2_tied` | 2 | `tied` | 494 | 13,405,853.0 | 13,410,816.0 | 0.0742 | 73.68% (42/57) | 85.96% (49/57) | 157.41 | 60.33 | 1.87 |
| `gmm_k3_full` | 3 | `full` | 1,394 | 3,907,436.8 | 3,921,441.8 | 0.0828 | 75.44% (43/57) | 87.72% (50/57) | 76.19 | 55.94 | 11.38 |
| `gmm_k3_diag` | 3 | `diag` | 147 | 11,271,995.2 | 11,273,763.4 | 0.0533 | 66.67% (38/57) | 84.21% (48/57) | 74.14 | 55.69 | 3.34 |
| `gmm_k3_tied` | 3 | `tied` | 524 | 11,452,288.1 | 11,457,552.6 | 0.0681 | 71.93% (41/57) | 85.96% (49/57) | 157.86 | 54.88 | 2.81 |
| **`gmm_k4_full`** | **4** | **`full`** | **1,859** | **-4,843,306.4** | **-4,824,629.7** | **0.4840** | **75.44% (43/57)** | **82.46% (47/57)** | **66.42** | **50.24** | **11.50** |
| `gmm_k4_diag` | 4 | `diag` | 206 | 10,922,772.5 | 10,925,133.4 | 0.0591 | 68.42% (39/57) | 84.21% (48/57) | 72.00 | 56.05 | 5.01 |
| `gmm_k4_tied` | 4 | `tied` | 554 | 11,810,241.2 | 11,815,807.0 | 0.0723 | 75.44% (43/57) | 84.21% (48/57) | 153.76 | 58.32 | 7.53 |
| `gmm_k5_full` | 5 | `full` | 2,324 | -4,778,899.9 | -4,755,551.4 | 0.2192 | 73.68% (42/57) | 85.96% (49/57) | 61.90 | 46.20 | 88.76 |
| `gmm_k5_diag` | 5 | `diag` | 265 | 10,760,323.8 | 10,763,277.5 | 0.0510 | 66.67% (38/57) | 82.46% (47/57) | 72.64 | 56.04 | 7.69 |
| **`gmm_k5_tied`** | **5** | **`tied`** | **584** | **12,018,802.8** | **12,024,670.1** | **0.1770** | **75.44% (43/57)** | **85.96% (49/57)** | **150.41** | **55.09** | **7.68** |

*Artifact Location:* Saved as CSV at [`results/tables/gmm_model_selection_comparison.csv`](results/tables/gmm_model_selection_comparison.csv).

---

## 4. Goodness-of-Fit Analysis (AIC & BIC)

AIC and BIC evaluate density goodness-of-fit under legitimate transaction behavior:
$$\mathrm{AIC} = 2p - 2\ln(\hat{L}), \qquad \mathrm{BIC} = p\ln(N) - 2\ln(\hat{L})$$

### Observations from Model Curves
* **Full Covariance Dominance:** Moving from $K=1$ full covariance ($\text{BIC} = 13.75\text{M}$) to $K=2$ full covariance ($\text{BIC} = 5.97\text{M}$) drops BIC by nearly 8 million points. Moving to $K=4$ full covariance further drops BIC to $-4.82\text{M}$.
* **Failure of Diagonal Structure:** Diagonal models (`diag`) exhibit flat, uncompetitive AIC/BIC trajectories ($10.7\text{M}$ to $11.9\text{M}$) and low Average Precision (~0.045 to 0.059). Assuming features are conditionally independent discards critical pairwise correlation between PCA dimensions and transaction Amount.
* **Why Negative AIC/BIC Occurs in Continuous Density Estimation:** In continuous multivariate density models, $p(x)$ can exceed 1 locally in concentrated regions of feature space. When $\ln \hat{L} > 0$, the $-2\ln \hat{L}$ term becomes negative and dominates the positive parameter penalty term. Lower (more negative) values indicate tighter probability density fit.
* **Diagnostic Plot:** Visualized in [`results/figures/bic_aic_curve.png`](results/figures/bic_aic_curve.png).

---

## 5. Numerical Stability & Regularization Sensitivity Audit

### 1. Covariance Eigenvalue Spectra & Condition Numbers
Under the pinned baseline configuration (`reg_covar = 1e-6`):
* `gmm_baseline_k1_full`: $\lambda_{\min} = 0.2699$, $\lambda_{\max} = 1.7478$, $\kappa = 6.47$ (exceptionally well-conditioned).
* `gmm_k4_full`:
  * **Component 0 ($\pi_0 = 0.0650$, 11,076 samples):** $\lambda_{\min} = 1.09 \times 10^{-3}$, $\kappa = 4,281$.
  * **Component 1 ($\pi_1 = 0.3720$, 63,440 samples):** 4 eigenvalues clamped at $1.00 \times 10^{-6}$, $\kappa = 1.93 \times 10^6$.
  * **Component 2 ($\pi_2 = 0.4676$, 79,734 samples):** 12 eigenvalues clamped at $1.00 \times 10^{-6}$, $\kappa = 1.82 \times 10^6$.
  * **Component 3 ($\pi_3 = 0.0955$, 16,277 samples):** $\lambda_{\min} = 4.49 \times 10^{-5}$, $\kappa = 2.45 \times 10^5$.

### 2. Explanation of Positive Training Lower Bounds (+14.21)
* **No Classical Singularity:** In pathological GMM collapse, a component captures 1 or 2 outlier points ($\pi_k \to 0, \sigma_k \to 0$). Here, Component 2 contains **79,734 training transactions** (46.8% of legitimate data).
* **Manifold Geometry:** The legitimate cardholder population naturally concentrates along an approximately 17-dimensional subspace in $\mathbb{R}^{29}$. In directions orthogonal to this manifold, transaction variance is nearly zero and hits the $10^{-6}$ regularization floor, driving the log-determinant $\ln|\Sigma_2|$ to $-166.93$ and shifting density normalization to $+83.47$.

### 3. Regularization Sensitivity Sweep (Supplemental Grid)
To prove the triage AP jump is genuine and not an artifact of $10^{-6}$ regularization, we evaluated $K=4$ full across varying `reg_covar`:

| `reg_covar` | Training Lower Bound | $\lambda_{\min}$ | Condition Number ($\kappa_{\max}$) | Average Precision | Recall @ 1% FPR | Recall @ 5% FPR | Analytical Finding |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`1e-06`** | **+14.21** | $1.0 \times 10^{-6}$ | $1.9 \times 10^6$ | **0.4840** | **75.44%** | **82.46%** | Baseline grid parameter |
| `1e-05` | +6.50 | $1.0 \times 10^{-5}$ | $2.0 \times 10^5$ | 0.4871 | 75.44% | 84.21% | Stable performance |
| `1e-04` | -0.95 | $1.0 \times 10^{-4}$ | $6.7 \times 10^4$ | 0.4837 | 77.19% | 85.96% | Stable performance |
| **`1e-03`** | **-10.06** | **$1.0 \times 10^{-3}$** | **$5.9 \times 10^3$** | **0.5223** | **77.19%** | **85.96%** | **Optimal balance: $\kappa < 6,000$ & highest AP** |
| `1e-02` | -18.68 | $1.0 \times 10^{-2}$ | $5.9 \times 10^2$ | 0.2843 | 77.19% | 87.72% | Regularization begins diluting density |
| `1e-01` | -27.68 | $1.0 \times 10^{-1}$ | $1.7 \times 10^2$ | 0.1488 | 73.68% | 84.21% | Over-smoothed density |

*Diagnostic Plot:* Score separation across cohorts visualized in [`results/figures/validation_score_distributions.png`](results/figures/validation_score_distributions.png).

---

## 6. Candidate Recommendations & Trade-Off Analysis

Rather than prematurely declaring an absolute winner on AIC/BIC alone, we provide three distinct candidate profiles for Member 4's triage evaluation:

### 1. Primary Precision Champion: `gmm_k4_full` ($K=4$, Full Covariance)
* **Average Precision:** **0.4840** (6.5x improvement over baseline).
* **Alert Quality (Top 100 alerts):** Catches **38 of 57 validation fraud cases** with **38.00% precision** (vs. 4 fraud cases / 4.00% precision for baseline).
* **Catch Rate @ 1% FPR:** 75.44% (43 / 57 fraud cases).
* **Trade-off:** Condition number is $1.9 \times 10^6$ under `reg_covar=1e-6` (reducible to $5.9 \times 10^3$ under `1e-3` without loss of triage power).

### 2. High-Sensitivity Champion: `gmm_k2_full` ($K=2$, Full Covariance)
* **Recall @ 5% FPR:** **89.47%** (catches 51 of 57 validation fraud cases, highest in the grid).
* **Recall @ 1% FPR:** 75.44% (43 / 57 fraud cases).
* **Average Precision:** 0.0750.
* **Trade-off:** Lower precision at tight alert budgets (only 11 fraud cases in top 100 alerts).

### 3. Lightweight / Low-Parameter Champion: `gmm_k5_tied` ($K=5$, Tied Covariance)
* **Average Precision:** **0.1770** (strong triage power with only 584 parameters).
* **Recall @ 1% FPR / 5% FPR:** **75.44% / 85.96%**.
* **Fit Duration:** 7.68s; single shared covariance matrix.
* **Trade-off:** Slightly lower AP than full covariance models, but highly resistant to overfitting.

---

## 7. Setup, Dependencies, and Reproducibility

### Pinned Environment
* Python 3.13.6 (Virtual environment: `.venv`)
* `scikit-learn==1.9.1`, `numpy==2.5.3`, `pandas==3.0.6`, `scipy==1.18.1`, `joblib==1.6.0`, `matplotlib==3.11.2`, `seaborn==0.13.2`

### Exact Reproduction Commands
```bash
# 1. Run the candidate model-selection experiment
./.venv/bin/python3 scripts/run_model_selection.py

# 2. Run the numerical stability audit
./.venv/bin/python3 scripts/audit_stability.py

# 3. Execute the unit test suite (13 tests)
./.venv/bin/python3 -m unittest discover -s tests -v
```

---

## 8. Handoff Protocol & Final Winner Status

* **Status of Final Winner:** **The final winning model remains strictly undecided.**
* **Next Steps for Member 4 (Anurag):** Member 4 will evaluate candidate models against operational criteria:
  1. Precision-Recall operating curves and threshold calibration.
  2. Cost-utility trade-offs (operational investigator capacity vs. financial fraud exposure).
  3. Final recommendation of the champion model for Phase 2 Autoencoder benchmarking.
* **Preservation Confirmation:** All baseline artifacts, preprocessing pipelines, chronological split manifests, and the untouched test set are preserved intact.
