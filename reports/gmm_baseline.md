# Phase 1: Gaussian Mixture Model (GMM) Baseline Freeze & Handoff Report

**Project:** Card Fraud Triage via Gaussian Mixture Density Scoring and Autoencoder Anomaly Detection  
**Phase:** Phase 1 — Gaussian Mixture Model (GMM) Anomaly Detection  
**Task:** Baseline Model Implementation, Verification, and Freeze  
**Status:** Frozen & Audited — Ready for Member 3 (Model Selection)  
**Date:** October 2026  

---

## 1. Executive Summary

This report establishes and freezes the benchmark Gaussian Mixture Model (GMM) baseline for the Card Fraud Triage anomaly detection pipeline. The model provides an unsupervised density estimation of normal credit card transaction behavior against which future multi-component GMMs and deep autoencoders will be benchmarked.

The baseline model has been trained strictly on legitimate transactions, verified against data leakage, mathematically audited for numerical stability, and confirmed 100% reproducible.

---

## 2. Frozen Model Architecture & Hyperparameters

In accordance with Phase 1 Step 2 specifications, the baseline model is a single-component full-covariance Gaussian Mixture Model:

```python
GaussianMixture(
    n_components=1,
    covariance_type="full",
    reg_covar=1e-6,
    n_init=5,
    random_state=42
)
```

### Parameter Specifications:
* **Components ($K$):** `1` (single unimodal multivariate Gaussian modeling legitimate transaction density).
* **Covariance Type:** `'full'` (full $29 \times 29$ covariance matrix capturing pairwise feature correlations).
* **Regularization Covariance (`reg_covar`):** `1e-6` (added to covariance diagonal to prevent singularity).
* **Initializations (`n_init`):** `5` (best run selected by log-likelihood).
* **Random State:** `42` (guarantees deterministic reproducibility).
* **Dimensionality ($D$):** `29` input features (`V1` to `V28` plus `log1p(Amount)`).

---

## 3. Training Protocol & Leak-Free Design

* **Training Subset:** Filtered strictly using `y_train == 0` from `data/processed/X_train.npy`.
* **Fitted Sample Count:** Exactly **`170,528`** legitimate transactions.
* **Excluded Fraud Cases:** Exactly **`360`** training fraud transactions (`Class == 1`) were excluded to prevent target contamination of normal behavior density.
* **Validation & Test Isolation:** Exactly **`0`** validation or test observations were seen during model fitting.
* **Convergence:** The EM algorithm converged in **`2`** iterations with lower bound log-likelihood of **`-40.3046`**.

---

## 4. Anomaly Scoring Definition

Anomaly scores are mathematically defined as the negative log-likelihood under the fitted GMM probability density function:

$$\text{anomaly\_score}(x) = -\ln p(x \mid \hat{\mu}, \hat{\Sigma}) = -\text{gmm.score\_samples}(x)$$

### Interpretation:
* **Higher scores** denote transactions that have lower probability density under the learned legitimate distribution, flagging them as potential anomalies / triage candidates.
* **Row-ID Tracking:** Scores are saved in structured DataFrames alongside their original `row_id` from `data/splits/split_manifest.csv` to ensure unambiguous join semantics without relying on array positions.

---

## 5. Baseline Score Distribution & Diagnostics

Summary of negative log-likelihood score distributions ($-\ln p(x)$):

### Training Partition (Diagnostics Only — 170,888 transactions):
| Cohort | Count | Mean | Std | Min | 25% | Median | 75% | 95% | Max |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Overall** | 170,888 | 43.26 | 111.69 | 27.90 | 32.09 | 35.05 | 39.07 | 55.79 | 10,227.38 |
| **Legitimate (`Class = 0`)** | 170,528 | 40.30 | 61.17 | 27.90 | 32.08 | 35.03 | 39.04 | 55.30 | 10,227.38 |
| **Fraud (`Class = 1`)** | 360 | 1,441.40 | 1,479.89 | 31.50 | 253.52 | 722.86 | 2,801.44 | 3,800.68 | 6,210.06 |

### Validation Partition (Baseline Reference — 56,957 transactions):
| Cohort | Count | Mean | Std | Min | 25% | Median | 75% | 95% | Max |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Overall** | 56,957 | 45.30 | 72.86 | 29.15 | 35.01 | 38.07 | 43.04 | 61.01 | 5,217.94 |
| **Legitimate (`Class = 0`)** | 56,900 | 45.04 | 72.08 | 29.15 | 35.01 | 38.07 | 43.02 | 60.72 | 5,217.94 |
| **Fraud (`Class = 1`)** | 57 | 308.21 | 220.01 | 31.95 | 151.10 | 273.53 | 501.94 | 531.25 | 1,254.60 |

*(Note: These distributions demonstrate natural separation in density space. In accordance with Phase 1 Step 2 restrictions, no decision thresholds or F1 claims have been established yet).*

---

## 6. Numerical Stability & Quality Audit

The fitted full covariance matrix $\hat{\Sigma} \in \mathbb{R}^{29 \times 29}$ was numerically evaluated:
1. **Symmetry:** Exact symmetry verified ($\max \vert \hat{\Sigma} - \hat{\Sigma}^T \vert = 0.00\text{e}+00$).
2. **Eigenvalue Spectrum:** All eigenvalues are strictly positive ($\lambda_{\min} = 0.2699$, $\lambda_{\max} = 1.7478$).
3. **Condition Number:** $\kappa(\hat{\Sigma}) = \lambda_{\max} / \lambda_{\min} = 6.475$, confirming the covariance matrix is exceptionally well-conditioned and far from ill-conditioning.
4. **Cholesky Factorization:** Successfully factored into lower triangular matrix with diagonal elements $\ge 1.00$.
5. **Finite Values:** Zero NaN, null, or infinite values in features or generated scores.

---

## 7. Artifact Inventory & File Specifications

| Artifact Path | Description | Format / Size | Schema / Key Attributes |
| :--- | :--- | :--- | :--- |
| [`models/gmm_baseline.joblib`](file:///Users/rudra/Downloads/card-fraud-triage-main/models/gmm_baseline.joblib) | Serialized fitted GMM model | Joblib (21.5 KB) | `GaussianMixture` instance |
| [`models/gmm_baseline_metadata.json`](file:///Users/rudra/Downloads/card-fraud-triage-main/models/gmm_baseline_metadata.json) | Model configuration & fit metadata | JSON (794 B) | Hyperparameters, sample counts, feature order, sklearn version |
| [`data/processed/scores/gmm_baseline_scores_train.csv`](file:///Users/rudra/Downloads/card-fraud-triage-main/data/processed/scores/gmm_baseline_scores_train.csv) | Full training set anomaly scores | CSV (5.0 MB) | `row_id` (int), `anomaly_score` (float), `split` (str) |
| [`data/processed/scores/gmm_baseline_scores_val.csv`](file:///Users/rudra/Downloads/card-fraud-triage-main/data/processed/scores/gmm_baseline_scores_val.csv) | Validation set anomaly scores | CSV (1.97 MB) | `row_id` (int), `anomaly_score` (float), `split` (str) |
| [`data/processed/scores/gmm_baseline_scores_metadata.json`](file:///Users/rudra/Downloads/card-fraud-triage-main/data/processed/scores/gmm_baseline_scores_metadata.json) | Statistical score diagnostics | JSON (2.2 KB) | Cohort quantile breakdowns for train and validation sets |

---

## 8. Handoff Instructions for Member 3 (Model Selection)

Member 3 can immediately load the frozen baseline and reproduce scores using the reusable modules:

```python
import os
from src.preprocessing import load_processed_data
from src.gmm_model import load_gmm_model
from src.scoring import compute_anomaly_scores

# 1. Load preprocessed features
data = load_processed_data("data/processed")
X_val, y_val = data["X_val"], data["y_val"]

# 2. Load frozen GMM baseline
gmm, metadata = load_gmm_model("models/gmm_baseline.joblib")

# 3. Score validation set
val_scores = compute_anomaly_scores(gmm, X_val)
print(f"Loaded GMM (K={gmm.n_components}). Validation scores shape: {val_scores.shape}")
```

### Reproducing Baseline Unit Tests:
```bash
./.venv/bin/python3 -m unittest discover -s tests -v
```

---

## 9. Baseline Limitations & Talking Points

1. **Unimodal Assumption:** A 1-component GMM models the legitimate distribution as a single ellipsoidal cluster. Real-world card transactions often span multiple behavioral profiles (e.g. low-value daily groceries vs. high-value travel bookings). Multi-component GMM exploration ($K \in \{2, 3, 4, 5\}$) in Phase 1 Step 3 will test whether mixture densities improve triage precision.
2. **Full Covariance Scaling:** While $K=1$ full covariance has only 435 parameters ($29 \times 30 / 2$), full covariance scaling grows quadratically with components ($K \cdot D(D+1)/2$). Diagonal or tied covariance structures may offer better sample efficiency for higher component counts.
3. **Threshold-Agnostic Baseline:** This step deliberately avoids premature threshold selection. Operating curves (Precision-Recall AUC, Cost-Utility Curves) will be constructed during model selection.
