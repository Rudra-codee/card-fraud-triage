# Card Fraud Triage via Gaussian Mixture Density Scoring and Autoencoder Anomaly Detection

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn 1.9.1](https://img.shields.io/badge/scikit--learn-1.9.1-orange.svg)](https://scikit-learn.org/)
[![Status: Baseline Frozen](https://img.shields.io/badge/Phase%201-GMM%20Baseline%20Frozen-success.svg)](#current-milestone-status)
[![Unit Tests: 11 Passing](https://img.shields.io/badge/tests-11%2F11%20passing-brightgreen.svg)](#6-verification-and-test-results)

An academic and research prototype for prioritizing suspicious credit card transactions using semi-supervised / unsupervised anomaly detection. Normal transaction behavior is modeled using density estimation on legitimate transactions, generating negative log-likelihood triage scores without fitting on fraudulent transactions or snooping into evaluation sets.

---

## Current Milestone Status

* **Completed:** Dataset verification, exploratory data analysis, chronological train/validation/test splitting (Strategy B: Timestamp Boundary Alignment), leak-free preprocessing, GMM baseline implementation ($K=1$, full covariance), negative log-likelihood scoring, automated unit testing (11/11 passing), numerical stability audit, and baseline freeze.
* **Current Stage:** **Phase 1 — GMM Baseline Frozen and Handed Off.**
* **Next Stage:** **Phase 1 — Model Selection (Member 3):** Component tuning ($K \in \{2, 3, 4, 5\}$), covariance structures (`full`, `diag`, `tied`), AIC/BIC analysis, and validation evaluation.
* **Disclaimer:** *This repository contains a research prototype for decision-support triage. It is not a production-deployed payment security system and does not claim commercial fraud-detection effectiveness.*

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Dataset Specification](#2-dataset-specification)
3. [Data Verification and Preprocessing](#3-data-verification-and-preprocessing)
4. [Repository Architecture](#4-repository-architecture)
5. [GMM Baseline Implementation](#5-gmm-baseline-implementation)
6. [Anomaly Scoring Pipeline](#6-anomaly-scoring-pipeline)
7. [Verification and Test Results](#7-verification-and-test-results)
8. [Setup and Reproducibility](#8-setup-and-reproducibility)
9. [Current Results and Limitations](#9-current-results-and-limitations)
10. [Team Responsibilities](#10-team-responsibilities)
11. [Project Roadmap](#11-project-roadmap)

---

## 1. Project Overview

### The Problem
Credit card fraud detection is characterized by **extreme class imbalance** (typically $<0.2\%$ of transactions are fraudulent) and **non-stationary fraud patterns**. In financial operations, human fraud analysts face alert fatigue from high false-positive rates. A fraud triage system must rank incoming transactions by anomaly severity to surface the most suspicious transactions for human review.

### The Modeling Principle
Traditional supervised classifiers can overfit to observed fraud patterns and fail to detect novel attack mechanisms. Our triage approach models the probability density distribution of **normal customer spending behavior**:

1. Fit density estimators strictly on **legitimate transactions** from the training partition (`Class == 0`).
2. Evaluate unseen transactions against the learned legitimate distribution.
3. Quantify anomaly severity using **negative log-likelihood (NLL)**:

$$\text{anomaly\_score}(x) = -\ln p(x \mid \hat{\theta}_{\text{GMM}}) = -\text{gmm.score\_samples}(x)$$

### Interpretation
* **Higher scores** denote transactions that are improbable under the learned normal spending profile, marking them as higher priority for triage.
* **Lower scores** represent high-density normal behavior.
* **Important Caveat:** A high anomaly score is an indicator of statistical rarity under the model, **not definitive proof of fraud**. High-value legitimate transactions, unusual international spending, or rare merchant categories can also produce high anomaly scores.

```
+---------------------------------------------------------------------------------------+
|                                DATA FLOW & TRIAGE PIPELINE                            |
+---------------------------------------------------------------------------------------+
|                                                                                       |
|   Raw Data (creditcard.csv)                                                           |
|       │                                                                               |
|       ▼                                                                               |
|   Chronological Split (60% Train / 20% Val / 20% Test) ──► split_manifest.csv         |
|       │                                                                               |
|       ▼                                                                               |
|   Amount Log-Transform: np.log1p(Amount)                                              |
|       │                                                                               |
|       ▼                                                                               |
|   StandardScaler (Fitted ONLY on Train Legit: Class == 0, N = 170,528)                |
|       │                                                                               |
|       ▼                                                                               |
|   Processed Arrays (X_train, X_val, X_test) [29 features: V1–V28 + log1p(Amount)]     |
|       │                                                                               |
|       ├─────────────────────────────────┐                                             |
|       ▼ (y_train == 0 ONLY)             ▼ (All Rows: Diagnostics / Validation)        |
|   GaussianMixture (K=1, full cov)   compute_anomaly_scores: -gmm.score_samples(X)     |
|       │                                 │                                             |
|       ▼                                 ▼                                             |
|   models/gmm_baseline.joblib        data/processed/scores/gmm_baseline_scores_*.csv   |
|                                                                                       |
+---------------------------------------------------------------------------------------+
```

---

## 2. Dataset Specification

All data specifications are verified against the raw dataset and split manifest:

* **Source Context:** Credit card transactions made by European cardholders in September 2013 over a two-day period (48 hours).
* **Total Transactions:** `284,807` records.
* **Total Fraud Cases:** `492` verified fraudulent transactions (`0.1727%` class balance).
* **Total Legitimate Cases:** `284,315` transactions (`99.8273%`).
* **Original Columns (31 total):**
  * `Time`: Seconds elapsed between each transaction and the first transaction in the dataset (range: $0.0\text{s}$ to $172,792.0\text{s}$).
  * `V1` through `V28`: 28 principal components obtained via Principal Component Analysis (PCA) to protect cardholder privacy.
  * `Amount`: Transaction value in Euros (range: €$0.00$ to €$25,691.16$).
  * `Class`: Ground-truth binary target (`0` = legitimate, `1` = fraudulent).
* **Storage Location:** `data/raw/creditcard.csv` (150,828,752 bytes).
* **Git LFS Requirement:** The raw CSV is tracked via Git Large File Storage (`git-lfs`). When cloning the repository, execute `git lfs pull` to populate the raw CSV if it appears as a 134-byte pointer file.

---

## 3. Data Verification and Preprocessing

The preprocessing pipeline ([`src/preprocessing.py`](src/preprocessing.py)) implements rigorous data hygiene to prevent leakage:

### 1. Chronological Partitioning (Strategy B)
Transactions arrive sequentially in time. Random train/test splits cause time-travel data leakage (training on future patterns to predict past transactions). We implement **Strategy B: Chronological Split with Timestamp Boundary Alignment**:

* **Train Partition (60%):** Sorted indices `0` to `170,887` ($0.0\text{s}$ to $120,396.0\text{s}$, ~33.44 hrs).
  * *Boundary Adjustment:* The nominal 60% integer cutoff ($170,884$) fell in the middle of a single second where 6 distinct transactions shared `Time = 120,396.0s`. The boundary was shifted by 4 rows to index $170,888$ so that all 6 transactions remain together in the training partition.
* **Validation Partition (20%):** Sorted indices `170,888` to `227,844` ($120,397.0\text{s}$ to $145,247.0\text{s}$, ~6.90 hrs).
* **Test Partition (20%):** Sorted indices `227,845` to `284,806` ($145,248.0\text{s}$ to $172,792.0\text{s}$, ~7.65 hrs).
* **Chronological Monotonicity:** $\max(\text{Time}_{\text{train}}) = 120,396.0\text{s} < \min(\text{Time}_{\text{val}}) = 120,397.0\text{s} < \min(\text{Time}_{\text{test}}) = 145,248.0\text{s}$.
* **Zero Duplicate Leakage:** All 773 identical duplicate transaction clusters occur strictly within a single partition; zero duplicate groups cross split boundaries.
* **Split Manifest:** Saved at `data/splits/split_manifest.csv` (`row_id`, `split`) to preserve immutability without duplicating raw files.

### 2. Confirmed Partition Dimensions
| Partition | Total Rows | Legitimate (`Class == 0`) | Fraud (`Class == 1`) | Fraud Rate | Split % | Model Fitting Usage |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | **170,888** | **170,528** | **360** | 0.2107% | 60.001% | **170,528 legit rows ONLY** |
| **Validation** | **56,957** | **56,900** | **57** | 0.1001% | 19.998% | **Unused for fitting** (evaluation only) |
| **Test** | **56,962** | **56,887** | **75** | 0.1317% | 20.000% | **Unused for fitting** (frozen benchmark) |
| **Total** | **284,807** | **284,315** | **492** | 0.1727% | 100.000% | Preserves all 492 fraud cases |

### 3. Feature Selection & Transformation
* **Baseline Feature Count:** Exactly **29 model features** (`V1` through `V28` + `Amount`).
* **Amount Compression:** Applied `np.log1p(Amount.astype(float))` to compress extreme right-skew (kurtosis reduced from 700+ to near-normal bounds). Zero-dollar verification pings map safely to $\ln(1+0) = 0$. The feature name remains `"Amount"` in metadata.
* **Excluded Columns:** `Time` (non-stationary temporal index), `Class` (target label), and `row_id` (tracking index) are strictly excluded from feature arrays.
* **Leak-Free StandardScaler:** Fitted **strictly on the 170,528 legitimate training rows** (`y_train == 0`). Fraud patterns, validation data, and test data were never exposed to the scaler.
  * Verified centering: $\text{mean}(X_{\text{train, legit}}) \approx 0.0$ (max abs error $5.51 \times 10^{-15}$).
  * Verified variance: $\text{var}(X_{\text{train, legit}}) \approx 1.0$ (range $[0.99999999999997, 1.00000000000011]$).
* **Row-ID Traceability:** Every preprocessed split retains its original 0-indexed row number in `row_ids_train.npy`, `row_ids_val.npy`, and `row_ids_test.npy`.

---

## 4. Repository Architecture

```text
card-fraud-triage-main/
├── .gitattributes                           # Git LFS specification for data/raw/creditcard.csv
├── .gitignore                              # Excludes .venv/, caches, temporary figures
├── requirements.txt                        # Pinned dependencies (scikit-learn==1.9.1, numpy==2.5.3)
├── README.md                               # Comprehensive project documentation
│
├── data/
│   ├── raw/
│   │   └── creditcard.csv                  # Raw Kaggle European cardholders dataset (150.8 MB)
│   ├── splits/
│   │   └── split_manifest.csv              # 284,807-row chronological split map (row_id, split)
│   └── processed/
│       ├── X_train.npy, X_val.npy, X_test.npy # Scaled 29-dimensional feature matrices
│       ├── y_train.npy, y_val.npy, y_test.npy # Target labels (0 = legit, 1 = fraud)
│       ├── row_ids_train.npy, ...          # Row-ID traceability tracking arrays
│       ├── scaler.joblib                   # StandardScaler fitted strictly on Class == 0 train rows
│       ├── feature_names.json              # List of 29 model feature names
│       ├── preprocessing_metadata.json     # Preprocessing configuration and sample counts
│       └── scores/
│           ├── gmm_baseline_scores_train.csv # 170,888 train scores (row_id, anomaly_score, split)
│           ├── gmm_baseline_scores_val.csv   # 56,957 val scores (row_id, anomaly_score, split)
│           └── gmm_baseline_scores_metadata.json # Baseline score summary statistics
│
├── models/
│   ├── gmm_baseline.joblib                 # Serialized fitted 1-component GMM (21.5 KB)
│   └── gmm_baseline_metadata.json          # GMM configuration, convergence, and sample metadata
│
├── notebooks/
│   ├── 01_dataset_verification_eda.ipynb   # Raw data audit, distributions, outlier analysis
│   ├── 02_preprocessing.ipynb              # Pipeline execution, scaling, and output validation
│   └── 03_gmm_model_selection.ipynb        # Data verification, GMM baseline training & audit
│
├── reports/
│   ├── data_cleaning_decisions.md          # Policy: retain duplicates to protect 19 fraud cases
│   ├── split_strategy.md                   # Chronological 60/20/20 Strategy B boundary details
│   └── gmm_baseline.md                     # GMM baseline freeze, stability audit, and handoff
│
├── src/
│   ├── create_split.py                     # Deterministic chronological split manifest generator
│   ├── preprocessing.py                    # Reusable leak-free preprocessing and data loader
│   ├── gmm_model.py                        # GMM baseline configuration, fitting, and persistence
│   └── scoring.py                          # Negative log-likelihood scoring and score export
│
└── tests/
    └── test_gmm_baseline.py                # 11 unit tests covering all Phase 1 Step 2 constraints
```

---

## 5. GMM Baseline Implementation

### Conceptual Overview: Gaussian Mixture Models & EM
A **Gaussian Mixture Model (GMM)** is a probabilistic density model that assumes data points are generated from a mixture of a finite number of Gaussian distributions with unknown parameters:

$$p(x) = \sum_{k=1}^{K} \pi_k \, \mathcal{N}(x \mid \mu_k, \Sigma_k)$$

Where $\pi_k$ are mixture weights ($\sum \pi_k = 1$), $\mu_k \in \mathbb{R}^D$ are component means, and $\Sigma_k \in \mathbb{R}^{D \times D}$ are covariance matrices.

Parameters are iteratively estimated using the **Expectation-Maximization (EM)** algorithm:
1. **E-step (Expectation):** Compute the posterior probability (responsibility) that component $k$ generated observation $x_i$.
2. **M-step (Maximization):** Update $\pi_k$, $\mu_k$, and $\Sigma_k$ to maximize the expected log-likelihood.

For our baseline ($K=1$), the model reduces to a single multivariate Gaussian $\mathcal{N}(\hat{\mu}, \hat{\Sigma})$. The EM algorithm completes in 2 iterations (initialization + convergence confirmation), yielding maximum likelihood sample estimates regularized by $\text{reg\_covar} \cdot I$.

### Frozen Baseline Configuration
Audited from [`src/gmm_model.py`](src/gmm_model.py) and [`models/gmm_baseline_metadata.json`](models/gmm_baseline_metadata.json):

```python
GaussianMixture(
    n_components=1,
    covariance_type="full",
    reg_covar=1e-6,
    n_init=5,
    random_state=42
)
```

| Hyperparameter | Value | Technical Meaning |
| :--- | :--- | :--- |
| `n_components` | `1` | Single unimodal Gaussian component modeling the primary legitimate transaction density. |
| `covariance_type` | `'full'` | Full $29 \times 29$ symmetric positive-definite covariance matrix modeling all pairwise feature correlations. |
| `reg_covar` | `1e-06` | Non-negative regularization constant added to covariance diagonals ($\Sigma + \epsilon I$) to prevent singularity. |
| `n_init` | `5` | Performs 5 independent EM initializations and selects the model with the highest log-likelihood lower bound. |
| `random_state` | `42` | Pinned random seed guaranteeing bit-for-bit deterministic initialization and reproducibility. |

### Fit Diagnostics
* **Fitted Observations:** `170,528` legitimate training rows (`Class == 0` exclusively).
* **Excluded Fraud Observations:** `360` training fraud rows (`Class == 1` strictly filtered out).
* **Convergence Status:** **`True`** (converged in `2` iterations).
* **EM Log-Likelihood Lower Bound:** **`-40.3046`**.
* **Parameter Count:** 435 covariance parameters ($29 \times 30 / 2$) + 29 mean parameters = 464 parameters.

---

## 6. Anomaly Scoring Pipeline

### Score Generation
Anomaly scores are calculated via [`src/scoring.py`](src/scoring.py) using negative log-likelihood:

$$\text{anomaly\_score}(x) = -\text{gmm.score\_samples}(x)$$

Because $p(x)$ evaluates the density under normal behavior, an observation deep in the tails of the legitimate distribution will have $p(x) \to 0$, causing $-\ln p(x) \to +\infty$.

### Score Artifacts & Traceability
Scores are exported to `data/processed/scores/`:
* `gmm_baseline_scores_train.csv` (170,888 rows, 5.0 MB)
* `gmm_baseline_scores_val.csv` (56,957 rows, 1.97 MB)

**CSV Schema:**
| Column | Type | Description |
| :--- | :--- | :--- |
| `row_id` | `int64` | Stable 0-indexed row identifier mapping directly to `split_manifest.csv` and raw data. |
| `anomaly_score` | `float64` | Negative log-likelihood score under the fitted GMM. |
| `split` | `string` | Split designation (`train` or `validation`). |

*Joining Rule:* Scores must **always** be joined with labels or transaction metadata using `row_id`. Positional array indexing must never be assumed.

### Descriptive Cohort Distributions
From [`data/processed/scores/gmm_baseline_scores_metadata.json`](data/processed/scores/gmm_baseline_scores_metadata.json):

#### Training Partition (Diagnostics Only):
| Cohort | Count | Mean | Std | Min | 25% | Median | 75% | 95% | Max |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Overall** | 170,888 | 43.26 | 111.69 | 27.90 | 32.09 | 35.05 | 39.07 | 55.79 | 10,227.38 |
| **Legitimate (`Class = 0`)** | 170,528 | 40.30 | 61.17 | 27.90 | 32.08 | 35.03 | 39.04 | 55.30 | 10,227.38 |
| **Fraud (`Class = 1`)** | 360 | 1,441.40 | 1,479.89 | 31.50 | 253.52 | 722.86 | 2,801.44 | 3,800.68 | 6,210.06 |

#### Validation Partition (Baseline Benchmark):
| Cohort | Count | Mean | Std | Min | 25% | Median | 75% | 95% | Max |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Overall** | 56,957 | 45.30 | 72.86 | 29.15 | 35.01 | 38.07 | 43.04 | 61.01 | 5,217.94 |
| **Legitimate (`Class = 0`)** | 56,900 | 45.04 | 72.08 | 29.15 | 35.01 | 38.07 | 43.02 | 60.72 | 5,217.94 |
| **Fraud (`Class = 1`)** | 57 | 308.21 | 220.01 | 31.95 | 151.10 | 273.53 | 501.94 | 531.25 | 1,254.60 |

*(Note: Cohort statistics are recorded for descriptive observation only. Ground-truth labels were never used to train the GMM).*

---

## 7. Verification and Test Results

### Automated Unit Test Suite
The test suite in [`tests/test_gmm_baseline.py`](tests/test_gmm_baseline.py) executes 11 assertions validating every Phase 1 Step 2 requirement:

```bash
./.venv/bin/python3 -m unittest discover -s tests -v
```

```text
test_01_fitting_subset_is_strictly_legitimate ... ok
test_02_fitted_observations_count_matches     ... ok
test_03_feature_count_and_order_consistent    ... ok
test_04_convergence_inspected                ... ok
test_05_scores_length_matches                 ... ok
test_06_scores_are_finite                     ... ok
test_07_score_artifacts_preserve_row_ids      ... ok
test_08_save_and_reload_reproduces_scores     ... ok
test_09_reproducibility_with_seed             ... ok
test_10_no_validation_or_test_leakage_in_model... ok
test_11_no_test_labels_used                   ... ok

----------------------------------------------------------------------
Ran 11 tests in 2.260s

OK
```

### Numerical Stability Audit
Audited in [`reports/gmm_baseline.md`](reports/gmm_baseline.md):
1. **Symmetry:** Exact symmetry confirmed ($\max \vert \Sigma - \Sigma^T \vert = 0.00\text{e}+00$).
2. **Eigenvalues:** $\lambda_{\min} = 0.2699$, $\lambda_{\max} = 1.7478$ (all $\lambda_i > 0$).
3. **Condition Number:** $\kappa(\Sigma) = \lambda_{\max} / \lambda_{\min} = 6.475$ (exceptionally well-conditioned, bounded away from singularity).
4. **Cholesky Factorization:** Successfully factored into lower-triangular matrix with strictly positive diagonal elements ($\ge 1.00$).
5. **Score Reproducibility:** Loading `models/gmm_baseline.joblib` and scoring validation samples yields a maximum absolute difference of **`0.00e+00`** compared to stored scores.

---

## 8. Setup and Reproducibility

### 1. Prerequisites
* Python 3.13 (or compatible Python 3.10+)
* Git with Git LFS installed (`git lfs --version`)

### 2. Environment Setup (macOS / Linux)
```bash
# 1. Clone repository
git clone https://github.com/Rudra-codee/card-fraud-triage.git
cd card-fraud-triage

# 2. Pull raw dataset via Git LFS (if needed)
git lfs pull

# 3. Create isolated virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 4. Install exact pinned dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note on Scikit-Learn Versioning:**  
> The project artifacts (`scaler.joblib`, `gmm_baseline.joblib`) were fitted and serialized under **`scikit-learn==1.9.1`** as pinned in `requirements.txt`. Running under older versions (e.g. 1.8.0) will cause scikit-learn to emit an `InconsistentVersionWarning`. Always activate `.venv` with `scikit-learn==1.9.1` to ensure warning-free deserialization.

### 3. Run Verification Tests
```bash
python3 -m unittest discover -s tests -v
```

### 4. Load Model and Generate Anomaly Scores
```python
from src.preprocessing import load_processed_data
from src.gmm_model import load_gmm_model
from src.scoring import compute_anomaly_scores

# Load preprocessed validation features
data = load_processed_data("data/processed")
X_val = data["X_val"]
row_ids_val = data["row_ids_val"]

# Load frozen baseline model
gmm, metadata = load_gmm_model("models/gmm_baseline.joblib")

# Compute anomaly scores (-score_samples)
val_scores = compute_anomaly_scores(gmm, X_val)
print(f"Loaded GMM baseline (K={gmm.n_components}).")
print(f"Validation anomaly scores shape: {val_scores.shape}")
print(f"Sample score: row_id {row_ids_val[0]} -> anomaly_score {val_scores[0]:.4f}")
```

---

## 9. Current Results and Limitations

### Established Findings
* The one-component GMM baseline is computationally lightweight, fitting on 170,528 29-dimensional transactions in ~1.2 seconds.
* The covariance matrix is well-conditioned ($\kappa = 6.475$), preventing numerical degeneracy.
* The negative log-likelihood score distribution separates the cohorts: median validation score is $38.07$ for legitimate transactions versus $273.53$ for fraudulent transactions.

### Limitations & Caveats
* **Unimodal Assumption:** A single Gaussian component assumes legitimate transaction behavior follows a single ellipsoidal cluster. In reality, credit card transactions exhibit multiple behavioral modes (e.g., micro-transactions, subscription rebills, high-value discretionary spend). Multi-component GMMs ($K \in \{2, 3, 4, 5\}$) will test whether modeling multiple clusters improves triage accuracy.
* **Score Overlap:** Because the baseline is unsupervised, outlier legitimate transactions (e.g., legitimate luxury purchases) overlap with fraud in the upper score tails ($95\text{th}$ percentile legitimate score is $60.72$, while minimum fraud score is $31.95$).
* **Threshold Independence:** No classification decision threshold has been established yet. Selecting thresholds on validation data or the test set at this stage would violate modeling integrity.
* **Academic Prototype:** This pipeline is intended for research and coursework evaluation. Real-world fraud detection requires latency SLAs ($<50\text{ms}$ per swipe), streaming graph features, merchant risk profiles, and active feedback loops.

---

## 10. Team Responsibilities

| Team Member | Role & Responsibilities | Current Status |
| :--- | :--- | :--- |
| **Rudraksh** (Member 1) | GMM baseline implementation, legitimate-only training protocol, negative log-likelihood anomaly scoring, baseline correctness testing, reproducibility audit, technical review, and viva defense. | **Completed & Frozen** (Phase 1 Steps 1–2) |
| **Bhargav** (Member 2) | Raw dataset verification, exploratory data analysis, chronological split strategy (Strategy B), feature engineering (`log1p(Amount)`), leak-free `StandardScaler`. | **Completed** (Data Phase) |
| **Kaustubh** (Member 3) | GMM model selection: component counts ($K \in \{2, 3, 4, 5\}$), covariance structures (`full`, `diag`, `tied`), AIC/BIC analysis, seed stability, experiment tracking. | **Next Milestone** (Phase 1 Step 3) |
| **Anurag** (Member 4) | Anomaly triage evaluation: Average Precision / PR-AUC, Recall at fixed false-positive rates (e.g. Recall @ 1% FPR, Recall @ 5% FPR), Precision@$K$ for triage capacity. | **Upcoming** (Phase 1 Step 4) |
| **Pratyush** (Member 5) | Cross-phase integration, comparative literature review, project documentation, final paper/report drafting, slide presentation, viva coordination. | **In Progress** |

---

## 11. Project Roadmap

- [x] **Milestone 0: Dataset Audit & Preprocessing**
  - [x] Exploratory data analysis and missing/infinite value audit ([`reports/data_cleaning_decisions.md`](reports/data_cleaning_decisions.md))
  - [x] Chronological split with timestamp boundary alignment ([`reports/split_strategy.md`](reports/split_strategy.md))
  - [x] Reusable preprocessing module ([`src/preprocessing.py`](src/preprocessing.py))
- [x] **Milestone 1: GMM Baseline Implementation & Freeze**
  - [x] GMM baseline training on legitimate training rows only ([`src/gmm_model.py`](src/gmm_model.py))
  - [x] Negative log-likelihood anomaly scoring ([`src/scoring.py`](src/scoring.py))
  - [x] 11-point automated unit test suite ([`tests/test_gmm_baseline.py`](tests/test_gmm_baseline.py))
  - [x] Numerical stability and covariance audit ([`reports/gmm_baseline.md`](reports/gmm_baseline.md))
  - [x] Baseline model and score artifact persistence
- [ ] **Milestone 2: GMM Model Selection (Member 3)**
  - [ ] Component tuning ($K \in \{2, 3, 4, 5\}$)
  - [ ] Covariance structure comparison (`full`, `diag`, `tied`, `spherical`)
  - [ ] AIC and BIC density goodness-of-fit evaluation
  - [ ] Validation set triage performance evaluation
  - [ ] Selection of champion GMM candidate
- [ ] **Milestone 3: Deep Autoencoder Anomaly Detection (Phase 2)**
  - [ ] Reconstruction-error based anomaly scoring
  - [ ] Architecture design and bottleneck compression tuning
- [ ] **Milestone 4: Final Evaluation & Handoff (Member 4 & 5)**
  - [ ] Precision-Recall AUC and Recall@FPR benchmarking on test set
  - [ ] Comparison between GMM and Autoencoder triage capabilities
  - [ ] Final academic report, presentation slides, and viva preparation
