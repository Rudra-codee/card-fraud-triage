# Card Fraud Triage via Gaussian Mixture Density Scoring and Autoencoder Anomaly Detection

[![Python 3.14 / 3.10+](https://img.shields.io/badge/python-3.14%20%7C%203.10%2B-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn 1.9.1](https://img.shields.io/badge/scikit--learn-1.9.1-orange.svg)](https://scikit-learn.org/)
[![PyTorch 2.14.1](https://img.shields.io/badge/PyTorch-2.14.1-red.svg)](https://pytorch.org/)
[![Status: Final Test Evaluation Completed](https://img.shields.io/badge/Phase%203-Final%20Test%20Evaluation%20Completed-success.svg)](#2-current-milestone-project-status)
[![Unit Tests: 61 Passing](https://img.shields.io/badge/tests-61%2F61%20passing-brightgreen.svg)](#12-verification-and-automated-test-results)

An academic and research prototype for prioritizing suspicious credit card transactions using semi-supervised / unsupervised anomaly detection. Normal transaction behavior is modeled strictly on legitimate transactions using two distinct paradigms:
1. **Gaussian Mixture Model (GMM) Density Estimation:** Quantifying statistical rarity via negative log-likelihood (NLL).
2. **Deep Autoencoder Neural Reconstruction:** Quantifying feature abnormality via mean squared reconstruction error (MSE).

The primary objective is **fraud triage**—ranking transactions to optimize analyst review capacity under stringent operational constraints (e.g., $\le 1$% false positive rate)—without fitting on fraudulent transactions or snooping into the held-out evaluation partitions.

---

## Table of Contents

1. [Project Overview & Scope](#1-project-overview-scope)
2. [Current Milestone & Project Status](#2-current-milestone-project-status)
3. [Dataset Specification](#3-dataset-specification)
4. [Data Verification and Leak-Free Preprocessing](#4-data-verification-and-leak-free-preprocessing)
5. [Repository Architecture](#5-repository-architecture)
6. [Frozen GMM Baseline Implementation](#6-frozen-gmm-baseline-implementation)
7. [GMM Model Selection Experiments](#7-gmm-model-selection-experiments)
8. [Deep Autoencoder Anomaly Detection](#8-deep-autoencoder-anomaly-detection)
9. [GMM Baseline vs. Autoencoder Validation Comparison](#9-gmm-baseline-vs-autoencoder-validation-comparison)
10. [Anomaly Scoring, Triage Evaluation & Operating Points](#10-anomaly-scoring-triage-evaluation-operating-points)
11. [Final Held-Out Test Evaluation Results](#11-final-held-out-test-evaluation-results)
12. [Verification and Automated Test Results](#12-verification-and-automated-test-results)
13. [Setup and Reproducibility](#13-setup-and-reproducibility)
14. [Limitations & Research Integrity](#14-limitations-research-integrity)
15. [Team Responsibilities](#15-team-responsibilities)
16. [Project Roadmap & Completion Checklist](#16-project-roadmap-completion-checklist)

---

## 1. Project Overview & Scope

### The Problem
Credit card fraud detection is characterized by **extreme class imbalance** (<0.2% fraud prevalence) and **non-stationary, adversarial fraud evolution**. In commercial fraud operations, human investigation teams experience severe alert fatigue from high false-positive volumes. A fraud triage system must reliably rank incoming transactions by anomaly severity to surface the highest-risk events within a limited analyst review budget.

### The Modeling Principle
Supervised classification models can overfit to historical fraud topologies and struggle to identify novel fraud mechanisms. Our triage framework models the distribution of **normal cardholder spending behavior**:

1. Fit estimators strictly on **legitimate transactions** from the training partition (`Class == 0`).
2. Evaluate unseen transactions against the learned legitimate representation.
3. Score anomaly severity:
   * **GMM:** Negative log-likelihood (NLL), $-\ln p(x \mid \hat{\theta}_{\mathrm{GMM}})$.
   * **Autoencoder:** Per-sample Mean Squared Error (MSE), $\frac{1}{D} \sum_{j=1}^D (x_j - \hat{x}_j)^2$.

```
+---------------------------------------------------------------------------------------------------+
|                                 DUAL-MODEL TRIAGE DATA FLOW                                       |
+---------------------------------------------------------------------------------------------------+
|                                                                                                   |
|   Raw Data (data/raw/creditcard.csv) [284,807 rows]                                               |
|       │                                                                                           |
|       ▼                                                                                           |
|   Chronological Split (Strategy B: Timestamp Boundary Alignment) ──► split_manifest.csv           |
|       ├─ Train Split (60%): 170,888 rows (170,528 legit, 360 fraud)                              |
|       ├─ Val Split   (20%):  56,957 rows ( 56,900 legit,  57 fraud)                              |
|       └─ Test Split  (20%):  56,962 rows ( 56,887 legit,  75 fraud) [Held-out for evaluation]    |
|       │                                                                                           |
|       ▼                                                                                           |
|   Feature Engineering & Scaling:                                                                  |
|       • Amount Compression: np.log1p(Amount)                                                      |
|       • StandardScaler fitted strictly on Train Legit (Class == 0, N = 170,528)                   |
|       • 29 Features: V1–V28 + transformed Amount (Time, Class, row_id excluded)                   |
|       │                                                                                           |
|       ├───────────────────────────────────────────┬───────────────────────────────────────────┤   |
|       ▼                                           ▼                                           |   |
|   PARADIGM 1: Density Estimation              PARADIGM 2: Neural Reconstruction               |   |
|   Gaussian Mixture Models (GMM)               Deep Autoencoder (PyTorch)                      |   |
|   • Baseline: K=1, full covariance            • Architecture: 29→20→10→5→10→20→29             |   |
|   • Candidates: K∈{2,3,4,5}, full/diag/tied   • Trained strictly on Train Legit (N=170,528)   |   |
|   • Score: -ln p(x | θ)                       • Val Legit early stopping (N=56,900)           |   |
|       │                                       • Score: mean((x - x̂)²)                         |   |
|       ▼                                           │                                           |   |
|   data/processed/scores/gmm_baseline_*.csv        ▼                                           |   |
|                                               data/processed/scores/autoencoder_*.csv         |   |
|       │                                           │                                           |   |
|       └─────────────────────┬─────────────────────┘                                           |   |
|                             ▼                                                                 |   |
|   Validation Triage Evaluation & Model Comparison (src/triage_evaluation.py)                  |   |
|   • Average Precision (PR-AUC), Recall @ 1% FPR, Recall @ 5% FPR, Top-N Alert Capacity       |   |
|   • reports/gmm_vs_autoencoder.md & reports/triage_evaluation.md                              |   |
|                             │                                                                 |   |
|                             ▼ (Thresholds and model artifacts frozen)                         |   |
|   Final Held-Out Test Evaluation (scripts/run_final_test_evaluation.py)                       |   |
|   • Unbiased benchmark on 56,962 test transactions (75 frauds)                                |   |
|   • reports/final_test_evaluation.md & results/tables/final_test_*.csv                        |   |
|                                                                                                   |
+---------------------------------------------------------------------------------------------------+
```

### Scope & Decision Support Distinction
* **Ranking, Not Classification:** The goal is prioritizing transactions for analyst investigation, not declaring that every high-scoring transaction is fraud.
* **Evaluation Independence:** Models are fitted on legitimate training data only; candidate operating thresholds are estimated on the validation partition only; the held-out test partition was strictly preserved during development and evaluated once for the final unbiased benchmark.
* **Academic Research Prototype:** This repository is an academic coursework implementation. It is not a commercial, production-deployed payment security system and does not claim production latency or operational guarantees.

---

## 2. Current Milestone & Project Status

### Executive Summary

| Category | Status | Summary |
| :--- | :---: | :--- |
| **Data Hygiene & Preprocessing** | **Completed** | Chronological 60/20/20 partitioning (Strategy B), `log1p(Amount)` compression, leak-free `StandardScaler` fitted solely on legitimate training data. |
| **GMM Baseline Model** | **Completed & Frozen** | 1-component full-covariance model fitted on 170,528 legitimate training rows; condition number $\kappa=6.475$; negative log-likelihood scores exported. |
| **GMM Model Selection** | **Completed** | 12 candidate grid ($K \in \{2, 3, 4, 5\}$, `full`, `diag`, `tied`), AIC/BIC analysis, seed sensitivity analysis, and candidate serialization. |
| **Triage Evaluation Framework** | **Completed** | Reusable validation evaluation module (`src/triage_evaluation.py`) with Recall@FPR, Average Precision, and Top-N review-capacity analysis. |
| **Autoencoder Model** | **Completed** | PyTorch symmetric autoencoder ($29 \to 20 \to 10 \to 5 \to 10 \to 20 \to 29$) trained on legitimate training data with legitimate validation monitoring. |
| **Cross-Model Comparison** | **Completed & Audited** | Direct validation comparison between GMM baseline and Autoencoder baseline documented in [`reports/gmm_vs_autoencoder.md`](reports/gmm_vs_autoencoder.md). |
| **Automated Test Suite** | **61 / 61 Passing** | Full unit test suite passing in restored `.venv` (`.venv/bin/python -m unittest discover tests`). |
| **Final Test Set Evaluation** | **Completed & Audited** | Final, unbiased evaluation of frozen GMM and Autoencoder models on 56,962 test transactions using frozen validation thresholds; documented in [`reports/final_test_evaluation.md`](reports/final_test_evaluation.md). |
| **Presentation & Final Viva** | **In Progress** | Final report consolidation and presentation slide deck in preparation. |

---

## 3. Dataset Specification

Verified against the raw dataset and split manifest:

* **Source:** European cardholder credit card transactions (September 2013, 48-hour window).
* **Total Transactions:** `284,807` records.
* **Fraud Distribution:** `492` fraud transactions (`0.1727%` prevalence) and `284,315` legitimate transactions (`99.8273%`).
* **Original Features (31 columns):**
  * `Time`: Seconds elapsed from first transaction (`0.0s` to `172,792.0s`).
  * `V1`–`V28`: PCA-transformed numerical features preserving confidentiality.
  * `Amount`: Transaction value in Euros (`€0.00` to `€25,691.16`).
  * `Class`: Target ground truth (`0` = legitimate, `1` = fraud).
* **Location:** `data/raw/creditcard.csv` (150.8 MB).
* **Git LFS Requirement:** Managed via Git Large File Storage (`.gitattributes`). Run `git lfs pull` after cloning if the file appears as a pointer.

### Chronological Partitioning (Strategy B: Timestamp Boundary Alignment)
Random splits cause future-to-past data leakage. We implement a strict chronological split:

| Partition | Total Rows | Legitimate (`Class == 0`) | Fraud (`Class == 1`) | Fraud Rate | Time Window (s) | Model Usage |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| **Train** | **170,888** | **170,528** | **360** | 0.2107% | 0.0s – 120,396.0s (~33.4h) | **Legitimate rows ONLY** used for parameter fitting |
| **Validation** | **56,957** | **56,900** | **57** | 0.1001% | 120,397.0s – 145,247.0s (~6.9h) | Early stopping & triage threshold evaluation |
| **Test** | **56,962** | **56,887** | **75** | 0.1317% | 145,248.0s – 172,792.0s (~7.6h) | **Held out during development; evaluated in final benchmark** |
| **Total** | **284,807** | **284,315** | **492** | 0.1727% | 0.0s – 172,792.0s (48.0h) | Full dataset preserved |

* **Boundary Alignment:** The nominal 60% index boundary fell within a 1-second burst sharing `Time = 120,396.0s`. The boundary was shifted by 4 rows to index 170,888 to ensure all simultaneous transactions remained in the training split.
* **Traceability:** Mapped immutably in [`data/splits/split_manifest.csv`](data/splits/split_manifest.csv) with unique, 0-indexed `row_id` values.

---

## 4. Data Verification and Leak-Free Preprocessing

The preprocessing pipeline ([`src/preprocessing.py`](src/preprocessing.py)) enforces rigorous data hygiene:

1. **Feature Selection (29 Model Features):**
   * Features: `V1` through `V28` plus transformed `Amount`.
   * In metadata ([`data/processed/feature_names.json`](data/processed/feature_names.json)), the transformed amount feature is labeled `"Amount"`.
   * Strict exclusions: `Time` (non-stationary temporal index), `Class` (target label), and `row_id` (record tracker) are never passed to model features.
2. **Amount Transformation:**
   * Compressed via $x' = \ln(1 + x)$ (`np.log1p(Amount.astype(float))`), reducing kurtosis from 700+ to near-normal bounds while mapping zero amounts safely to $\ln(1+0) = 0$.
3. **Leak-Free StandardScaler:**
   * Fitted strictly on the **170,528 legitimate training rows** (`y_train == 0`).
   * Scaler centering: $\mathrm{mean}(X_{\mathrm{train, legit}}) \approx 0.0$ (max error $5.51 \times 10^{-15}$).
   * Scaler variance: $\mathrm{var}(X_{\mathrm{train, legit}}) \approx 1.0$.
   * Frozen scaler saved at [`data/processed/scaler.joblib`](data/processed/scaler.joblib) and applied without refitting to validation and test features.
4. **Legitimate-Only Fitting Principle:**
   * Fraud labels are used strictly for post-hoc validation triage evaluation.
   * For the Autoencoder, early stopping monitors reconstruction loss solely on the **56,900 legitimate validation transactions** (`y_val == 0`).

---

## 5. Repository Architecture

```text
card-fraud-triage/
├── .gitattributes                           # Git LFS tracking for data/raw/creditcard.csv
├── .gitignore                              # Ignores .venv/, caches, temporary scratch files
├── requirements.txt                        # 114 pinned dependencies (scikit-learn==1.9.1, torch==2.14.1)
├── README.md                               # Comprehensive project documentation
│
├── data/
│   ├── raw/
│   │   └── creditcard.csv                  # Raw Kaggle European cardholders dataset (150.8 MB)
│   ├── splits/
│   │   └── split_manifest.csv              # 284,807-row chronological split manifest (row_id, split)
│   └── processed/
│       ├── X_train.npy, X_val.npy, X_test.npy # Scaled 29-dimensional feature arrays
│       ├── y_train.npy, y_val.npy, y_test.npy # Binary labels (0 = legit, 1 = fraud)
│       ├── row_ids_train.npy, ...          # Deterministic row-ID tracking arrays
│       ├── scaler.joblib                   # StandardScaler fitted on legitimate train rows only
│       ├── feature_names.json              # List of 29 feature names (V1-V28, Amount)
│       ├── preprocessing_metadata.json     # Split and transformation metadata
│       └── scores/
│           ├── gmm_baseline_scores_train.csv # 170,888 train scores (row_id, anomaly_score, split)
│           ├── gmm_baseline_scores_val.csv   # 56,957 val scores (row_id, anomaly_score, split)
│           ├── gmm_baseline_scores_test.csv  # 56,962 test scores (row_id, anomaly_score, split)
│           ├── gmm_k4_full_scores_test.csv   # 56,962 test scores for K=4 full GMM
│           ├── gmm_baseline_scores_metadata.json # GMM score distribution statistics
│           ├── autoencoder_scores_train.csv  # 170,888 train reconstruction error scores
│           ├── autoencoder_scores_val.csv    # 56,957 val reconstruction error scores
│           └── autoencoder_scores_test.csv   # 56,962 test reconstruction error scores
│
├── models/
│   ├── gmm_baseline.joblib                 # Frozen 1-component full-covariance GMM (21.5 KB)
│   ├── gmm_baseline_metadata.json          # GMM configuration, convergence, and fit diagnostics
│   ├── autoencoder_baseline.pt             # Best Autoencoder checkpoint weights (epoch 30)
│   ├── autoencoder_baseline_metadata.json  # Autoencoder loss history, config, sample counts
│   └── candidates/                         # 12 serialized GMM candidate models and metadata
│       ├── gmm_k2_full.joblib, ...         # Multi-component GMM candidates (K=2..5, full/diag/tied)
│       └── gmm_k4_full.joblib              # Primary multi-component candidate
│
├── notebooks/
│   ├── 01_dataset_verification_eda.ipynb   # Raw data audit and exploratory distribution analysis
│   ├── 02_preprocessing.ipynb              # Pipeline execution, scaling, and validation checks
│   ├── 03_gmm_model_selection.ipynb        # Initial GMM baseline training and numerical audit
│   └── 04_gmm_model_selection_experiments.ipynb # 12-model grid search, AIC/BIC, and triage plots
│
├── reports/
│   ├── data_cleaning_decisions.md          # Policy documenting retention of duplicate rows
│   ├── split_strategy.md                   # Chronological 60/20/20 Strategy B boundary details
│   ├── gmm_baseline.md                     # Baseline freeze, numerical stability, and handoff
│   ├── gmm_model_selection.md              # 12-model GMM grid, AIC/BIC analysis, seed audit
│   ├── triage_evaluation.md                # Validation triage metrics, FPR thresholds, review budgets
│   ├── autoencoder_baseline.md             # Autoencoder training, architecture, and score audit
│   ├── gmm_vs_autoencoder.md               # Cross-model validation comparison and trade-offs
│   └── final_test_evaluation.md            # Final held-out test evaluation benchmark report
│
├── results/
│   ├── figures/
│   │   ├── triage_precision_recall_curve.png # Validation precision-recall curves
│   │   ├── final_test_precision_recall_curve.png # Final test precision-recall curves
│   │   └── final_test_review_capacity.png    # Test fraud triage by review capacity
│   └── tables/
│       ├── triage_model_comparison.csv     # Validation model comparison across AP, Recall@FPR
│       ├── triage_fpr_thresholds.csv       # Validation FPR-constrained threshold operating points
│       ├── triage_review_capacity.csv      # Validation top-N review budget operating points
│       ├── triage_stability_check.csv      # Seed and regularization sensitivity evaluation
│       ├── triage_validation_checks.json   # Automated validation integrity check outputs
│       ├── final_test_model_comparison.csv # Final test model comparison (AP, Recall@FPR, precision)
│       ├── final_test_threshold_metrics.csv # Final test metrics at frozen validation thresholds
│       ├── final_test_review_capacity.csv  # Final test top-N review capacity (Top 50..500)
│       ├── final_test_val_vs_test_comparison.csv # Side-by-side validation vs test comparison
│       └── final_test_validation_checks.json # Final test integrity audit and checks
│
├── scripts/
│   ├── audit_stability.py                  # Numerical stability and condition number audit
│   ├── generate_notebook.py                # Notebook generation utility
│   ├── run_model_selection.py              # Executes 12-candidate GMM grid search and exports models
│   ├── run_triage_evaluation.py            # Executes validation triage evaluation suite
│   ├── run_autoencoder.py                  # Trains Autoencoder, exports scores, logs metadata
│   └── run_final_test_evaluation.py        # Executes final held-out test evaluation benchmark
│
├── src/
│   ├── create_split.py                     # Deterministic chronological split generator
│   ├── preprocessing.py                    # Reusable leak-free preprocessing and data loader
│   ├── gmm_model.py                        # GMM baseline configuration, fitting, and persistence
│   ├── scoring.py                          # Negative log-likelihood scoring and CSV export
│   ├── model_selection.py                  # GMM grid training, AIC/BIC, candidate evaluation
│   ├── triage_evaluation.py                # Reusable triage metrics, frozen threshold evaluation
│   ├── autoencoder_model.py                # PyTorch Autoencoder architecture and training loop
│   └── autoencoder_scoring.py              # Autoencoder MSE scoring and score CSV export
│
└── tests/
    ├── test_gmm_baseline.py                # 11 tests: GMM baseline training, leakage, stability
    ├── test_model_selection.py             # 2 tests: Candidate fitting and triage metrics
    ├── test_triage_evaluation.py           # 27 tests: AP, Recall@FPR, ties, row alignment, budgets
    ├── test_autoencoder.py                 # 8 tests: AE architecture, training, MSE, early stopping
    └── test_final_test_evaluation.py       # 13 tests: Test integrity, frozen thresholds, zero refit
```

---

## 6. Frozen GMM Baseline Implementation

### Probabilistic Density Modeling
A Gaussian Mixture Model represents normal transaction probability density as:
$$p(x) = \sum_{k=1}^K \pi_k \, \mathcal{N}(x \mid \mu_k, \Sigma_k)$$
where $\pi_k$ are mixture weights, $\mu_k \in \mathbb{R}^{29}$, and $\Sigma_k \in \mathbb{R}^{29 \times 29}$.

For the baseline ($K=1$), the model reduces to a single multivariate Gaussian $\mathcal{N}(\hat{\mu}, \hat{\Sigma})$. Anomaly scores follow the negative log-likelihood convention:
$$\mathrm{Score}(x) = -\ln p(x \mid \hat{\theta}_{\mathrm{GMM}})$$
In code, this is evaluated directly using the model log-density:
```python
anomaly_score = -gmm.score_samples(X)
```

### Frozen Baseline Configuration
Audited from [`models/gmm_baseline_metadata.json`](models/gmm_baseline_metadata.json):
* `n_components = 1`
* `covariance_type = "full"`
* `reg_covar = 1e-6`
* `n_init = 5`
* `random_state = 42`

### Fit Diagnostics & Stability
* **Fitted Observations:** `170,528` legitimate training rows (`Class == 0`).
* **Excluded Fraud Observations:** `360` training fraud rows strictly filtered out.
* **Convergence:** Converged in `2` iterations; lower-bound log-likelihood: `-40.3046`.
* **Parameter Count:** 435 covariance parameters + 29 mean parameters = **464 parameters**.
* **Numerical Conditioning:** $\lambda_{\min} = 0.2699$, $\lambda_{\max} = 1.7478$; condition number $\kappa(\Sigma) = 6.475$ (exceptionally well-conditioned, bounded away from singularity).
* **Score Reproducibility:** Rescoring validation rows reproduces saved scores with maximum absolute difference of `0.00e+00`.

---

## 7. GMM Model Selection Experiments

To investigate whether modeling multi-modal transaction patterns improves fraud triage, a systematic 12-model grid search was conducted in Phase 1 Task 3 ([`reports/gmm_model_selection.md`](reports/gmm_model_selection.md)):
* **Component Counts ($K$):** $K \in \{2, 3, 4, 5\}$
* **Covariance Structures:** `'full'`, `'diag'`, and `'tied'`
* **Fixed Hyperparameters:** `reg_covar = 1e-6`, `n_init = 5`, `random_state = 42`

### 12-Candidate Grid Results
Evaluated on legitimate training fit (AIC/BIC) and unseen validation transactions ($N = 56,957$, 57 frauds):

| Model Identifier | $K$ | Covariance Type | Free Parameters ($p$) | AIC | BIC | Average Precision (PR-AUC) | Recall @ 1% FPR | Recall @ 5% FPR | Fit Time (s) |
| :--- | :---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `gmm_baseline_k1_full` | 1 | `full` | 464 | 13,747,068.7 | 13,751,730.4 | 0.0740 | 73.68% (42/57) | 87.72% (50/57) | 1.20 |
| `gmm_k2_full` | 2 | `full` | 929 | 5,963,565.2 | 5,972,898.6 | 0.0750 | 75.44% (43/57) | **89.47% (51/57)** | 5.73 |
| `gmm_k2_diag` | 2 | `diag` | 88 | 11,967,086.1 | 11,968,261.6 | 0.0455 | 57.89% (33/57) | 85.96% (49/57) | 1.38 |
| `gmm_k2_tied` | 2 | `tied` | 494 | 13,405,853.0 | 13,410,816.0 | 0.0742 | 73.68% (42/57) | 85.96% (49/57) | 1.87 |
| `gmm_k3_full` | 3 | `full` | 1,394 | 3,907,436.8 | 3,921,441.8 | 0.0828 | 75.44% (43/57) | 87.72% (50/57) | 11.38 |
| `gmm_k3_diag` | 3 | `diag` | 147 | 11,271,995.2 | 11,273,763.4 | 0.0533 | 66.67% (38/57) | 84.21% (48/57) | 3.34 |
| `gmm_k3_tied` | 3 | `tied` | 524 | 11,452,288.1 | 11,457,552.6 | 0.0681 | 71.93% (41/57) | 85.96% (49/57) | 2.81 |
| **`gmm_k4_full`** | **4** | **`full`** | **1,859** | **-4,843,306.4** | **-4,824,629.7** | **0.4840** | **75.44% (43/57)** | **82.46% (47/57)** | **11.50** |
| `gmm_k4_diag` | 4 | `diag` | 206 | 10,922,772.5 | 10,925,133.4 | 0.0591 | 68.42% (39/57) | 84.21% (48/57) | 5.01 |
| `gmm_k4_tied` | 4 | `tied` | 554 | 11,810,241.2 | 11,815,807.0 | 0.0723 | 75.44% (43/57) | 84.21% (48/57) | 7.53 |
| `gmm_k5_full` | 5 | `full` | 2,324 | -4,778,899.9 | -4,755,551.4 | 0.2192 | 73.68% (42/57) | 85.96% (49/57) | 88.76 |
| `gmm_k5_diag` | 5 | `diag` | 265 | 10,760,323.8 | 10,763,277.5 | 0.0510 | 66.67% (38/57) | 82.46% (47/57) | 7.69 |
| `gmm_k5_tied` | 5 | `tied` | 584 | 12,018,802.8 | 12,024,670.1 | 0.1770 | 75.44% (43/57) | 85.96% (49/57) | 7.68 |

### Goodness-of-Fit Analysis (AIC & BIC)
* **Full Covariance Dominance:** AIC/BIC improved drastically with unconstrained covariance ($K=1$ BIC: $13.75\text{M} \to K=4$ BIC: $-4.82\text{M}$).
* **Failure of Diagonal Structures:** Diagonal models (`diag`) exhibited poor density fit ($\text{BIC} \ge 10.7\text{M}$) and poor triage AP (~0.045 to 0.059), demonstrating that ignoring feature correlations in credit data destroys ranking utility.
* **Negative Information Criteria:** In continuous probability densities, $p(x)$ can exceed 1 in dense clusters. When $\ln \hat{L} > 0$, the $-2\ln \hat{L}$ term becomes negative and dominates parameter penalties; more negative values represent tighter probability density fit.

### Seed Sensitivity and Candidate Recommendation
* **The $K=4$ Full Finding:** Under seed 42, `gmm_k4_full` achieved an AP of **0.4840** and captured 38 of 57 validation frauds within the top 100 alerts (0.11% FPR).
* **Seed Sensitivity Audit:** For $K=4$ full covariance, alternative random seeds 0–4 produced validation AP values from approximately 0.069 to 0.102, while the saved seed-42 artifact achieved validation AP of approximately 0.484. This indicates substantial seed sensitivity, and the seed-42 result should not be assumed to be representative of typical runs. The broader range of approximately 0.045–0.059 observed during model selection applies to diagonal-covariance candidates rather than the $K=4$ full-covariance seed comparison.
* **Triage Recommendation ([`reports/triage_evaluation.md`](reports/triage_evaluation.md)):** Advance the frozen `gmm_k4_full.joblib` artifact to final testing alongside the frozen `gmm_baseline_k1_full.joblib` as reference. If $K=4$'s advantage in top-budget alerts does not hold on the held-out test set, fall back to the simpler, uniquely determined $K=1$ baseline.

---

## 8. Deep Autoencoder Anomaly Detection

### Neural Reconstruction Architecture
Implemented in PyTorch ([`src/autoencoder_model.py`](src/autoencoder_model.py)), the Autoencoder compresses input transactions into a low-dimensional bottleneck and reconstructs them:

$$\text{Encoder: } 29 \to 20 \to 10 \to 5 \qquad \text{Decoder: } 5 \to 10 \to 20 \to 29$$

* **Layer Activations:** `ReLU` on hidden layers, `Linear` on the output layer.
* **Total Parameters:** **1,929 trainable parameters** (weights and biases).
* **Loss Function:** Mean Squared Error ($\text{MSELoss}$).
* **Optimization:** Adam optimizer, learning rate $10^{-3}$, batch size $256$, seed $42$.
* **Execution:** CPU-based execution (~10.24 seconds total training time).

```python
Autoencoder(
  (encoder): Sequential(
    (0): Linear(in_features=29, out_features=20, bias=True)
    (1): ReLU()
    (2): Linear(in_features=20, out_features=10, bias=True)
    (3): ReLU()
    (4): Linear(in_features=10, out_features=5, bias=True)
  )
  (decoder): Sequential(
    (0): Linear(in_features=5, out_features=10, bias=True)
    (1): ReLU()
    (2): Linear(in_features=10, out_features=20, bias=True)
    (3): ReLU()
    (4): Linear(in_features=20, out_features=29, bias=True)
  )
)
```

### Legitimate-Only Training & Early Stopping
* **Training Data:** Fitted on **170,528 legitimate training rows** (`y_train == 0`); all 360 fraud cases excluded.
* **Validation Monitoring:** Early stopping monitored reconstruction error strictly on the **56,900 legitimate validation rows** (`y_val == 0`); 57 validation fraud cases excluded from loss calculation.
* **Saved Checkpoint:** Saved at [`models/autoencoder_baseline.pt`](models/autoencoder_baseline.pt) with metadata [`models/autoencoder_baseline_metadata.json`](models/autoencoder_baseline_metadata.json):
  * Best Epoch: Epoch 30 (best validation loss: `0.70142`).
  * Final Training Loss: `0.46780`.
  * Training Duration: `10.24 s`.

### Reconstruction Error Scoring
Implemented in [`src/autoencoder_scoring.py`](src/autoencoder_scoring.py), per-transaction anomaly scores measure the Mean Squared Error across all 29 features:

$$\mathrm{Score}_{\mathrm{AE}}(x) = \frac{1}{D} \sum_{j=1}^{D} (x_j - \hat{x}_j)^2$$

where $D = 29$, $x$ is the input transaction vector, and $\hat{x}$ is the reconstructed vector. Transactions that deviate from normal spending manifolds incur higher reconstruction error and are prioritized for investigation.

Exported score artifacts:
* `data/processed/scores/autoencoder_scores_train.csv` (170,888 rows, finite, $\ge 0$).
* `data/processed/scores/autoencoder_scores_val.csv` (56,957 rows, finite, $\ge 0$).

---

## 9. GMM Baseline vs. Autoencoder Validation Comparison

Both models were audited and compared on the **identical validation partition** ($N = 56,957$, 57 frauds) using the shared evaluation module [`src/triage_evaluation.py`](src/triage_evaluation.py). Documented in [`reports/gmm_vs_autoencoder.md`](reports/gmm_vs_autoencoder.md):

### Validation Comparison Table

| Metric | Frozen GMM Baseline | Autoencoder Baseline | Comparison / Trade-off |
| :--- | :--- | :--- | :--- |
| **Model Type** | Probabilistic Density Estimation | Neural Compression & Reconstruction | Generative density vs. deterministic reconstruction |
| **Score Definition** | Negative Log-Likelihood: $-\ln p(x)$ | Per-sample MSE: $\frac{1}{29} \sum (x_j - \hat{x}_j)^2$ | Different score scales; thresholded independently |
| **Trainable Parameters** | 464 parameters | 1,929 parameters | Autoencoder has ~4.2× more parameters |
| **Training Duration (CPU)**| ~1.20 s | ~10.24 s | GMM fits ~8.5× faster |
| **Average Precision (PR-AUC)**| **0.0740** | **0.0765** | AE leads by +0.0025 (+3.4% relative) |
| **Recall @ 1% FPR** | **73.68% (42 / 57 frauds)** | **71.93% (41 / 57 frauds)** | GMM catches **1 additional fraud** |
| └ Operating Threshold | 158.814 | 2.651 | Model-specific score threshold |
| └ Legitimate Flagged (FP) | 566 / 56,900 (`0.995%`) | 569 / 56,900 (`1.000%`) | Both strictly satisfy $\le 1.0$% FPR |
| **Recall @ 5% FPR** | **87.72% (50 / 57 frauds)** | **87.72% (50 / 57 frauds)** | **Identical recall (50 / 57 frauds caught)** |
| └ Operating Threshold | 60.726 | 1.341 | Model-specific score threshold |
| └ Legitimate Flagged (FP) | 2,845 / 56,900 (`5.000%`) | 2,845 / 56,900 (`5.000%`) | Exactly 2,845 false positives flagged |

### Technical Interpretation
1. **Average Precision:** The Autoencoder's AP (`0.0765`) exceeds the GMM baseline (`0.0740`) by 0.0025 points. Both are ~7–8× higher than random guessing ($\approx 0.0010$). With only 57 fraud cases, this margin is within natural sampling variance and does not establish overall superiority.
2. **Tight Operating Budget (1% FPR):** The GMM baseline catches 42 of 57 frauds versus 41 of 57 for the Autoencoder, providing a slight advantage when analyst investigation budgets are tightest.
3. **Moderate Operating Budget (5% FPR):** Both models achieve identical recall, capturing 50 of 57 frauds and flagging exactly 2,845 false positives. The 7 uncaught frauds overlap heavily with legitimate data under both representations.
4. **Generalization Note:** These initial comparison figures were measured on the validation partition only. The subsequent Section 11 presents the final, unbiased evaluation on the held-out test partition.

---

## 10. Anomaly Scoring, Triage Evaluation & Operating Points

The triage evaluation framework in [`src/triage_evaluation.py`](src/triage_evaluation.py) turns raw anomaly scores into operational triage metrics:

### 1. Score Alignment by Record ID
Scores are stored with their `row_id`. In evaluation, rows are joined back to ground-truth labels using a one-to-one merge on `row_id` (`align_scores_with_labels`). Positional array alignment is never assumed, preventing subtle label-permutation bugs.

### 2. Operating Point Constraints
* **Fixed False Positive Rate (FPR $\le \alpha$):**
  * Denominator: strictly legitimate rows ($N = 56,900$ in validation, $N = 56,887$ in test).
  * Threshold selection: `roc_curve(drop_intermediate=False)` is used to identify distinct score thresholds. The algorithm selects the **lowest threshold that keeps FPR $\le \alpha$**.
  * **Tie-Handling Rule:** If including a group of tied scores causes FPR to exceed the budget, the entire group remains unflagged. Achieved FPR is guaranteed to stay $\le \alpha$.
* **Fixed Analyst Review Budget (Top-N Alerts):**
  * Flags the top $N$ highest-scoring transactions ($N \in \{50, 100, 200, 500\}$).
  * Evaluates true positives (frauds caught) and achieved precision under fixed operational review capacity.

---

## 11. Final Held-Out Test Evaluation Results

The final, unbiased evaluation was executed on the chronological held-out test partition ($N = 56,962$ transactions, $75$ fraud cases, 0.1317% fraud prevalence) using the frozen model artifacts and frozen validation-derived thresholds. Full documentation and diagnostics are available in [`reports/final_test_evaluation.md`](reports/final_test_evaluation.md).

### 11.1 Methodological Rigor & Threshold Provenance
* **Zero Test Retraining:** Preprocessing parameters (`StandardScaler`) and model parameters were frozen.
* **Frozen Threshold Rule:** Operating thresholds were selected strictly on the validation partition to meet 1% and 5% FPR targets, then applied directly to test anomaly scores. **No threshold tuning was performed on test labels.**
* **Decoupled Scoring:** Anomaly scores were generated from $X_{\text{test}}$ alone without access to target labels.
* **Verified Threshold Values:**
  * **GMM Baseline:** 1% FPR threshold = `158.814`, 5% FPR threshold = `60.726` (NLL units).
  * **GMM $K=4$ Full:** 1% FPR threshold = `66.436`, 5% FPR threshold = `50.241` (NLL units).
  * **Autoencoder Baseline:** 1% FPR threshold = `2.651`, 5% FPR threshold = `1.341` (MSE units).

### 11.2 Comprehensive Model Comparison on Held-Out Test Partition

Evaluated across the 56,962 test transactions ($56,887$ legitimate, $75$ fraud):

| Metric | Frozen GMM Baseline | Frozen GMM $K=4$ Full | Frozen Autoencoder Baseline |
| :--- | :---: | :---: | :---: |
| **Model Artifact** | `models/gmm_baseline.joblib` | `models/candidates/gmm_k4_full.joblib` | `models/autoencoder_baseline.pt` |
| **Model Architecture** | $K=1$, full covariance (464 params) | $K=4$, full covariance (1,859 params) | $29 \to 20 \to 10 \to 5 \to 10 \to 20 \to 29$ (1,929 params) |
| **Average Precision (AP)** | **0.1610** | **0.5770** | **0.1793** |
| └ 95% Bootstrap CI | [0.1208, 0.2209] | [0.4588, 0.6973] | [0.1318, 0.2447] |
| **ROC-AUC** | **0.9622** | **0.9156** | **0.9436** |
| **Recall @ Frozen 1% Val Thresh** | **72.00% (54 / 75)** | **76.00% (57 / 75)** | **73.33% (55 / 75)** |
| └ Operating Threshold | 158.814 | 66.436 | 2.651 |
| └ Actual Test FPR | 0.6645% (378 FP) | 0.5889% (335 FP) | 0.5590% (318 FP) |
| └ Alerts & Precision | 432 alerts (12.50%) | 392 alerts (14.54%) | 373 alerts (14.75%) |
| **Recall @ Frozen 5% Val Thresh** | **84.00% (63 / 75)** | **80.00% (60 / 75)** | **80.00% (60 / 75)** |
| └ Operating Threshold | 60.726 | 50.241 | 1.341 |
| └ Actual Test FPR | 4.0132% (2,283 FP) | 3.6845% (2,096 FP) | 3.6019% (2,049 FP) |
| └ Alerts & Precision | 2,346 alerts (2.69%) | 2,156 alerts (2.78%) | 2,109 alerts (2.84%) |

*Serialized outputs:* [`results/tables/final_test_model_comparison.csv`](results/tables/final_test_model_comparison.csv) and [`results/tables/final_test_threshold_metrics.csv`](results/tables/final_test_threshold_metrics.csv).

### 11.3 Fixed Operational Review Capacity (Top-$N$ Alerts)

Measuring triage utility when investigation capacity is constrained to top-$N$ ranked alerts:

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

*Serialized output:* [`results/tables/final_test_review_capacity.csv`](results/tables/final_test_review_capacity.csv).
*Figure:* [`results/figures/final_test_review_capacity.png`](results/figures/final_test_review_capacity.png).

### 11.4 Validation vs. Test Generalization

| Model | Split | Transactions | Fraud Count | AP | Recall @ 1% Thresh | Actual FPR @ 1% | Recall @ 5% Thresh | Actual FPR @ 5% |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **GMM Baseline** | Validation | 56,957 | 57 | 0.0740 | 73.68% (42/57) | 0.9947% | 87.72% (50/57) | 5.0000% |
| | **Test** | **56,962** | **75** | **0.1610** | **72.00% (54/75)** | **0.6645%** | **84.00% (63/75)** | **4.0132%** |
| **GMM $K=4$ Full** | Validation | 56,957 | 57 | 0.4840 | 75.44% (43/57) | 0.9965% | 82.46% (47/57) | 5.0000% |
| | **Test** | **56,962** | **75** | **0.5770** | **76.00% (57/75)** | **0.5889%** | **80.00% (60/75)** | **3.6845%** |
| **Autoencoder** | Validation | 56,957 | 57 | 0.0765 | 71.93% (41/57) | 1.0000% | 87.72% (50/57) | 5.0000% |
| | **Test** | **56,962** | **75** | **0.1793** | **73.33% (55/75)** | **0.5590%** | **80.00% (60/75)** | **3.6019%** |

*Serialized comparison:* [`results/tables/final_test_val_vs_test_comparison.csv`](results/tables/final_test_val_vs_test_comparison.csv).

### 11.5 Detailed Discussion & Technical Synthesis

1. **Did the GMM $K=4$ Advantage Hold?**
   * **Yes, on this frozen artifact:** The saved seed-42 $K=4$ full-covariance candidate maintained a decisive advantage in Average Precision ($0.5770$ vs. $0.1610$ for baseline and $0.1793$ for AE).
   * **Tight Review Budget Dominance:** For investigations capped at 50 alerts, $K=4$ captured **39 of 75 frauds with 78.00% precision**, compared to 9 frauds (18.00%) for the GMM baseline and 11 frauds (22.00%) for the Autoencoder.
   * **Essential Seed-Sensitivity Caveat:** As established during model selection ([`reports/gmm_model_selection.md`](reports/gmm_model_selection.md)), full-covariance GMMs with $K \ge 2$ suffer from non-convex EM optimization and seed sensitivity: alternative random seeds 0–4 produced validation AP values from approximately 0.069 to 0.102, while the saved seed-42 artifact achieved validation AP of approximately 0.484. The broader range of approximately 0.045–0.059 observed during model selection applies to diagonal-covariance candidates. While the saved seed-42 artifact generalized effectively on this test partition (0.5770 AP), retraining with a different seed should not be assumed to yield equivalent performance without explicit validation-based checkpoint selection.

2. **Heavy-Tail Coverage at Wider Operating Budgets:**
   * At the 5% FPR threshold, the simpler single-component **GMM Baseline achieved the highest overall recall: 84.00% (63 of 75 frauds)**, intercepting 3 more frauds than $K=4$ (60/75, 80.00%) and the Autoencoder (60/75, 80.00%).
   * Unimodal full-covariance scoring applies a smooth Mahalanobis quadratic distance penalty that flags severe multi-feature outliers reliably, whereas mixture components can occasionally assign intermediate likelihoods to unusual spending profiles.

3. **Autoencoder Generalization:**
   * The Autoencoder achieved a slightly higher test AP than the GMM baseline ($0.1793$ vs. $0.1610$), with overlapping 95% bootstrap confidence intervals ([0.1318, 0.2447] vs. [0.1208, 0.2209]).
   * It caught slightly more frauds at Top 50 (11 vs. 9) and Top 100 (31 vs. 28), but required ~4.2× more parameters and ~8.5× longer training time.

4. **Operating Point Stability:**
   * Thresholds selected on validation data proved conservative on test data: actual test FPR was **0.56% to 0.66%** at the nominal 1% target, and **3.60% to 4.01%** at the nominal 5% target.
   * This confirms that validation-frozen thresholds do not produce runaway false alarm rates on unseen chronological data.

---

## 12. Verification and Automated Test Results

The repository features an automated test suite verifying data hygiene, model convergence, score finiteness, and evaluation correctness:

```bash
.venv/bin/python -m unittest discover tests
```

### Test Suite Execution Summary (61 Passing Tests)

```text
.............................................................
----------------------------------------------------------------------
Ran 61 tests in 7.541s

OK
```

### Verified Test Breakdown
* **[`tests/test_gmm_baseline.py`](tests/test_gmm_baseline.py) (11 tests):**
  * Legitimate-only training filtering (`Class == 0`).
  * Feature count (29) and feature order invariance.
  * Model convergence, finite score outputs, row-ID preservation.
  * Seed reproducibility and zero leakage of test labels.
* **[`tests/test_autoencoder.py`](tests/test_autoencoder.py) (8 tests):**
  * Symmetric encoder-decoder architecture ($29 \to 20 \to 10 \to 5 \to 10 \to 20 \to 29$).
  * Finite reconstruction loss and exclusion of training frauds.
  * Deterministic training with fixed seed.
  * Early stopping and best checkpoint restoration.
  * Non-negative, finite MSE scores and row-ID alignment.
* **[`tests/test_model_selection.py`](tests/test_model_selection.py) (2 tests):**
  * Synthetic triage metrics computation.
  * Candidate model legitimate filtering and score consistency.
* **[`tests/test_triage_evaluation.py`](tests/test_triage_evaluation.py) (27 tests):**
  * Input validation (rejection of NaNs, infs, 2D scores, invalid labels).
  * Average Precision known-value calculations and PR curve bounds.
  * Recall@FPR calculation with exact tie-handling compliance.
  * Top-N analyst alert budget boundary tests.
  * Row-ID shuffle resistance and score-label alignment.
  * Frozen baseline rescore reproducibility.
* **[`tests/test_final_test_evaluation.py`](tests/test_final_test_evaluation.py) (13 tests):**
  * Missing, duplicate, and extraneous row-ID detection.
  * Score and label alignment invariance after row shuffling.
  * Rejection of non-finite scores (NaN, $\pm\infty$).
  * Frozen threshold evaluation without retuning from test labels.
  * Legitimate-only denominator verification for test FPR computation.
  * Alert budget compliance and deterministic tie-dropping policy.
  * Zero model-fitting or scaler-fitting execution during evaluation.
  * Backward compatibility of validation evaluation functions.

---

## 13. Setup and Reproducibility

### Prerequisites
* macOS (Apple Silicon ARM64 verified) or Linux
* Python 3.14 (or compatible Python 3.10+)
* Git with Git LFS installed (`git lfs --version`)

### Quick-Start Environment Setup
```bash
# 1. Clone repository
git clone https://github.com/Rudra-codee/card-fraud-triage.git
cd card-fraud-triage

# 2. Pull raw dataset via Git LFS
git lfs install
git lfs pull

# 3. Create isolated virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 4. Install exact pinned dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 5. Run complete test suite (61 tests)
python -m unittest discover tests
```

### Re-running Pipeline Workflows
All artifacts and scores are checked into the repository for instant reproduction. To re-execute specific stages:

* **Run GMM Model Selection Experiments (12 candidates):**
  ```bash
  python scripts/run_model_selection.py
  ```
* **Run Validation Triage Evaluation & Review-Capacity Analysis:**
  ```bash
  python scripts/run_triage_evaluation.py
  ```
* **Train Autoencoder & Generate Scores:**
  ```bash
  python scripts/run_autoencoder.py
  ```
* **Run Final Held-Out Test Evaluation Benchmark:**
  ```bash
  python scripts/run_final_test_evaluation.py
  ```

---

## 14. Limitations & Research Integrity

1. **Extreme Class Imbalance:** Fraud represents only 0.1727% of transactions overall, 0.1001% (57 cases) in validation, and 0.1317% (75 cases) in test. A shift of 3 caught frauds alters test recall by 4.0 percentage points.
2. **Anomaly Rarity vs. Fraudulent Intent:** Statistical anomaly detection identifies transactions that deviate from normal spending distributions. High-value legitimate transactions, unusual travel expenses, or rare merchant codes can produce high anomaly scores without being fraudulent.
3. **Incompatible Score Scales:** GMM negative log-likelihood scores ($-\ln p(x)$, range $\approx 28$ to $5,200+$) cannot be directly compared in numerical magnitude to Autoencoder MSE scores ($\approx 0.06$ to $170+$). They can only be compared through unified operating constraints (such as fixed FPR thresholds or review budgets).
4. **Seed Sensitivity in Multi-Component GMMs:** Multi-component GMMs ($K \ge 2$) have non-convex likelihood surfaces. While $K=4$ under seed 42 yielded high test performance (0.5770 AP), alternative random seeds 0–4 produced validation APs in the 0.069–0.102 range. Retraining with a different seed should not be assumed to yield equivalent performance without validation-based checkpoint selection.
5. **Validation vs. Test Generalization:** Operating thresholds were determined on validation data and evaluated once on the test set. While generalization held well on this 48-hour time series, results from a single dataset do not guarantee performance across different financial institutions or fraud seasons.
6. **Prototype Status:** This system is an academic research prototype. It does not provide sub-50ms latency guarantees, merchant graph profiling, or streaming state handling necessary for production payment rails.

---

## 15. Team Responsibilities

| Member | Name | Primary Project Responsibilities | Verified Deliverable Status |
| :---: | :--- | :--- | :--- |
| **Member 1** | **Rudraksh** | GMM baseline implementation, legitimate-only training protocol, anomaly scoring, Autoencoder core model implementation, technical integration, and test suite. | **Completed & Verified** (GMM baseline frozen, Autoencoder implementation in `src/autoencoder_model.py`, passing tests). |
| **Member 2** | **Bhargav** | Dataset verification, exploratory data analysis, chronological split strategy (Strategy B), feature engineering (`log1p(Amount)`), and leak-free `StandardScaler`. | **Completed & Verified** (EDA notebook, `split_manifest.csv`, `src/preprocessing.py`). |
| **Member 3** | **Kaustubh** | GMM model selection: component counts ($K \in \{2, 3, 4, 5\}$), covariance structures (`full`, `diag`, `tied`), AIC/BIC analysis, and seed stability audit. | **Completed & Verified** (12 candidate models, `reports/gmm_model_selection.md`, `run_model_selection.py`). |
| **Member 4** | **Anurag** | Triage evaluation: Average Precision (PR-AUC), fixed-FPR recall, Top-N review capacity, threshold tie-handling, cross-model comparison, and final held-out test evaluation. | **Completed & Verified** (`src/triage_evaluation.py`, `scripts/run_final_test_evaluation.py`, `reports/final_test_evaluation.md`, `results/tables/final_test_*.csv`, `tests/test_final_test_evaluation.py`). |
| **Member 5** | **Pratyush** | Autoencoder validation evaluation and comparison, project documentation consolidation, final academic report synthesis, and presentation slide deck. | **Completed & In Progress** (`reports/gmm_vs_autoencoder.md` verified; final report & presentation pending confirmation). |

---

## 16. Project Roadmap & Completion Checklist

- [x] **Milestone 0: Dataset Audit & Preprocessing**
  - [x] Exploratory data analysis and missing value audit ([`reports/data_cleaning_decisions.md`](reports/data_cleaning_decisions.md))
  - [x] Chronological split with timestamp boundary alignment ([`reports/split_strategy.md`](reports/split_strategy.md))
  - [x] Reusable leak-free preprocessing module ([`src/preprocessing.py`](src/preprocessing.py))
- [x] **Milestone 1: GMM Baseline Implementation & Freeze**
  - [x] Baseline fitted on legitimate training rows only ([`src/gmm_model.py`](src/gmm_model.py))
  - [x] Negative log-likelihood anomaly scoring ([`src/scoring.py`](src/scoring.py))
  - [x] Baseline numerical stability and covariance condition audit ([`reports/gmm_baseline.md`](reports/gmm_baseline.md))
- [x] **Milestone 2: GMM Model Selection Experiments**
  - [x] 12-model grid search across $K \in \{2, 3, 4, 5\}$ and `full`, `diag`, `tied` covariances
  - [x] AIC and BIC goodness-of-fit evaluation
  - [x] Candidate serialization under `models/candidates/` and report ([`reports/gmm_model_selection.md`](reports/gmm_model_selection.md))
  - [x] Seed stability and regularization audit
- [x] **Milestone 3: Validation Triage Evaluation Framework**
  - [x] Implementation of `src/triage_evaluation.py` (Recall@FPR, Average Precision, Top-N budgets)
  - [x] Rigorous tie handling and label-alignment verification
  - [x] Comprehensive evaluation report and candidate recommendation ([`reports/triage_evaluation.md`](reports/triage_evaluation.md))
- [x] **Milestone 4: Deep Autoencoder Anomaly Detection**
  - [x] Symmetric neural network implementation ([`src/autoencoder_model.py`](src/autoencoder_model.py))
  - [x] Legitimate-only training with legitimate validation early stopping
  - [x] Mean squared reconstruction error scoring pipeline ([`src/autoencoder_scoring.py`](src/autoencoder_scoring.py))
  - [x] Checkpoint persistence and metadata export ([`reports/autoencoder_baseline.md`](reports/autoencoder_baseline.md))
- [x] **Milestone 5: Cross-Model Validation Comparison**
  - [x] Recomputed validation metrics across GMM baseline and Autoencoder baseline
  - [x] Operating point analysis at 1% FPR and 5% FPR
  - [x] Audited comparison report ([`reports/gmm_vs_autoencoder.md`](reports/gmm_vs_autoencoder.md))
- [x] **Milestone 6: Verification & Automated Test Suite**
  - [x] Comprehensive 61-test automated suite across all 5 test modules (`tests/`)
  - [x] Clean pass verification (`.venv/bin/python -m unittest discover tests`)
- [x] **Milestone 7: Final Held-Out Test Set Evaluation**
  - [x] One-time unbiased evaluation of frozen candidate models on the 56,962-row test partition
  - [x] Final generalization benchmarking against validation operating points
  - [x] Exported test scores, comparison tables, review capacity metrics, and plots
  - [x] Final evaluation report documented in [`reports/final_test_evaluation.md`](reports/final_test_evaluation.md)
- [ ] **Milestone 8: Final Academic Report & Presentation**
  - [ ] Synthesis of consolidated project viva report
  - [ ] Final presentation slide deck preparation and defense coordination
