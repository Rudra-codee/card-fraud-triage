# Project Report: Card Fraud Triage via Gaussian Mixture Density Scoring and Autoencoder Anomaly Detection

**Academic Coursework & Research Prototype Evaluation**  
**Repository Working Directory:** `card-fraud-triage`  
**Current Milestone:** Milestone 7 (Final Held-Out Test Evaluation Benchmark) Completed & Audited  
**Test Suite Status:** 61 / 61 Unit & Regression Tests Passing  

---

## 1. Executive Summary & Project Purpose

### 1.1 The Operational Problem in Commercial Card Fraud
Credit card fraud detection is one of the most critical challenges in financial technology and payment security. In real-world banking rails, detection systems must contend with three major structural hurdles:
1. **Extreme Class Imbalance:** Fraudulent transactions account for less than $0.2\%$ of total transaction traffic (in this benchmark, $492$ out of $284,807$ records, or $0.1727\%$). Standard machine learning classification accuracy is completely ineffective: a naive model predicting 100% legitimate transactions achieves $>99.8\%$ accuracy while failing to intercept any fraud.
2. **Severe Alert Fatigue on Human Triage Teams:** Commercial card fraud operations rely on human fraud analysts (triage desks) who review flagged alerts. These teams operate under finite daily investigation budgets (e.g., $50$ to $500$ alerts per shift). When automated systems trigger high false alarm rates, analysts become overwhelmed, leading to missed fraud and costly delays.
3. **Adversarial Non-Stationarity & Novel Attack Vectors:** Fraudsters continually adapt their behavior to bypass static business rules and supervised models that overfit to historical fraud patterns. Supervised models often struggle to identify zero-day attack vectors that do not match past training labels.

### 1.2 The Modeling Principle: Normal-Behavior Anomaly Triage
Rather than treating fraud detection as a standard supervised classification problem, this project implements a **pure normal-behavior anomaly triage framework**:
* **Legitimate-Only Training:** Model estimators are fitted strictly on **legitimate transactions** (`Class == 0`) from the chronological training partition. Fraud transactions are never exposed during model fitting or parameter optimization.
* **Dual-Paradigm Anomaly Scoring:**
  1. **Probabilistic Density Estimation (GMM):** Quantifies statistical rarity via Negative Log-Likelihood ($\mathrm{NLL} = -\ln p(x \mid \hat{\theta}_{\mathrm{GMM}})$). Transactions falling in low-density tail regions receive high anomaly scores.
  2. **Deep Neural Reconstruction (Autoencoder):** Quantifies feature abnormality via per-sample Mean Squared Reconstruction Error ($\mathrm{MSE} = \frac{1}{29} \sum_{j=1}^{29} (x_j - \hat{x}_j)^2$). Transactions deviating from the learned normal spending manifold incur high reconstruction error.
* **Triage Prioritization:** Incoming prospective transactions are ranked by anomaly severity to maximize fraud interception within strict analyst review capacity constraints (e.g., Top 50 alerts or False Positive Rate $\le 1\%$).

---

## 2. Dataset Specification & Data Quality Policy

### 2.1 Raw Dataset Specification
The project is built and benchmarked on the canonical European cardholder credit card transaction dataset (`creditcard.csv`):
* **Source:** Transactions made by European cardholders in September 2013 across a 48-hour continuous window.
* **Total Transactions:** `284,807` records ($150.8\text{ MB}$, tracked via Git LFS).
* **Ground-Truth Distribution:** `284,315` legitimate transactions ($99.8273\%$) and `492` fraudulent transactions ($0.1727\%$).
* **Raw Features (31 columns):**
  * `Time`: Seconds elapsed between each transaction and the first transaction in the dataset ($0.0\text{s}$ to $172,792.0\text{s}$).
  * `V1`–`V28`: Numerical feature projections obtained via Principal Component Analysis (PCA) to preserve cardholder privacy.
  * `Amount`: Transaction amount in Euros ($€0.00$ to $€25,691.16$).
  * `Class`: Target ground truth ($0$ = legitimate, $1$ = fraud).

### 2.2 Data Quality & Cleaning Decisions (`reports/data_cleaning_decisions.md`)
A formal data quality audit established the preprocessing rules for the raw dataset:

| Issue / Check | Measured Count | Decision Taken | Rationale & Impact |
| :--- | :--- | :--- | :--- |
| **Missing Values** | `0` null values across all 31 columns | Take no action | Dataset is complete; no imputation required. |
| **Infinite Values** | `0` positive or negative infinite values | Take no action | All numeric features are finite and well-bounded. |
| **Data Types** | 30 columns `float64`, 1 column `int64` (`Class`) | Retain original types | All columns are already correctly typed. |
| **Zero-Dollar Transactions** | 1,825 transactions are `$0.00` | **Retain 100% as valid** | Zero amounts represent card authorization pings / account verification pings; account for $5.49\%$ of frauds vs $0.63\%$ of legitimate transactions. $\ln(1+0) = 0$ is mathematically safe. |
| **Extreme Outliers** | Max Amount: $€25,691.16$; long tails on $V_1..V_{28}$ | **Retain all outliers** | Real financial transactions naturally exhibit heavy tails; clipping or removing legitimate variance would bias anomaly detection. |
| **Exact Duplicate Rows** | 1,081 redundant duplicate copies (1,854 total rows in duplicate groups) | **Option 3 Approved: Retain all duplicate rows** | Exact duplicates represent automated carding bot bursts or gateway retries; deleting them would destroy 19 rare fraud cases ($3.86\%$ of all positives). All duplicate clusters fall cleanly within single split partitions. |

---

## 3. Chronological Partitioning & Preprocessing Pipeline

### 3.1 Chronological Split Strategy (Strategy B: Timestamp Boundary Alignment)
Random splits (`train_test_split`) create future-to-past data leakage. The project implements a strict chronological split into Train (60%), Validation (20%), and Test (20%):

```
[0.0s ────────────────────────────── 120,396.0s] [120,397.0s ───────── 145,247.0s] [145,248.0s ──────── 172,792.0s]
          TRAIN SPLIT (60%)                             VALIDATION SPLIT (20%)              TEST SPLIT (20%)
  170,888 rows (170,528 legit, 360 fraud)        56,957 rows (56,900 legit, 57 fraud) 56,962 rows (56,887 legit, 75 fraud)
```

* **Boundary Burst Alignment:** The nominal 60% integer cutoff fell at index 170,884, where 6 transactions occurred in the same second (`Time = 120,396.0s`). The cutoff was shifted by 4 rows to index **170,888**, keeping all simultaneous transactions together in the training partition.
* **Partition Disjointness & Traceability:** All partitions are mutually disjoint ($\text{Train} \cap \text{Val} = \emptyset, \text{Train} \cap \text{Test} = \emptyset, \text{Val} \cap \text{Test} = \emptyset$). Mapped immutably in `data/splits/split_manifest.csv` with unique, 0-indexed `row_id` values.
* **Zero Duplicate Leakage:** All 773 duplicate groups share identical timestamps ($t_{\min} = t_{\max}$) and fall 100% within a single partition.

### 3.2 Leak-Free Preprocessing Pipeline (`src/preprocessing.py`)
1. **Feature Selection (29 Model Features):** $V_1$ through $V_{28}$ plus transformed `Amount`. Non-stationary `Time`, label `Class`, and tracking `row_id` are strictly excluded from model feature matrices.
2. **Amount Compression:** Transformed via $x' = \ln(1 + x)$ (`np.log1p(Amount.astype(float))`), reducing skewness and mapping zero amounts safely to $0.0$.
3. **Legitimate-Only Scaling:** `StandardScaler` is fitted strictly on the **170,528 legitimate training rows** (`y_train == 0`). The frozen scaler parameters are saved at `data/processed/scaler.joblib` and applied without refitting to validation and test features:
   $$\mathrm{mean}(X_{\mathrm{train, legit}}) \approx 0.0, \quad \mathrm{var}(X_{\mathrm{train, legit}}) \approx 1.0$$

---

## 4. Dual-Paradigm Anomaly Detection Implementations

```
                                  INPUT TRANSACTION x ∈ ℝ²⁹
                 [ V1, V2, ..., V28, ln(1 + Amount) - Standardized ]
                                        │
                    ┌───────────────────┴───────────────────┐
                    ▼                                       ▼
        [ PARADIGM 1: GMM DENSITY ]            [ PARADIGM 2: AUTOENCODER ]
        Single / Multi-Component              Symmetric Neural Bottleneck
        Gaussian Mixture Model                 (29 → 20 → 10 → 5 → 10 → 20 → 29)
                    │                                       │
                    ▼                                       ▼
          Negative Log-Likelihood                 Reconstruction Error
           Score = -ln p(x | θ)                   Score = 1/29 Σ (x_j - x̂_j)²
```

### 4.1 Paradigm 1: Gaussian Mixture Models (`src/gmm_model.py`, `src/model_selection.py`)
A Gaussian Mixture Model parameterizes normal cardholder spending as a linear combination of $K$ multivariate Gaussian components:
$$p(x \mid \theta) = \sum_{k=1}^K \pi_k \, \mathcal{N}(x \mid \mu_k, \Sigma_k), \quad \sum_{k=1}^K \pi_k = 1$$
Anomaly severity follows the negative log-likelihood convention:
$$\mathrm{Score}_{\mathrm{GMM}}(x) = -\ln p(x \mid \hat{\theta}_{\mathrm{GMM}})$$

#### 4.1.1 Frozen Single-Component Baseline (`models/gmmbaseline.joblib`)
* **Configuration:** $K=1$, full covariance ($29 \times 29$), $\text{reg\_covar} = 10^{-6}, n\_\text{init} = 5, \text{random\_state} = 42$.
* **Training Subset:** $170,528$ legitimate training rows (all 360 training frauds excluded).
* **Parameters:** $464$ free parameters (29 means + 435 unique covariance elements).
* **Numerical Stability:** Eigenvalues $\lambda_{\min} = 0.2699, \lambda_{\max} = 1.7478$; condition number $\kappa = 6.475$ (exceptionally well-conditioned, globally convex Mahalanobis metric).
* **Convergence:** Converged in 2 iterations with lower-bound log-likelihood of $-40.3046$.

#### 4.1.2 12-Model GMM Selection Grid (`reports/gmm_model_selection.md`)
Evaluated $K \in \{2, 3, 4, 5\}$ across `full`, `diag`, and `tied` covariance structures on the validation set ($56,957$ rows, $57$ frauds):

| Model Identifier | $K$ | Covariance Type | Free Parameters | AIC | BIC | Validation AP | Recall @ 1% FPR | Recall @ 5% FPR |
| :--- | :---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `gmm_baseline_k1_full` | 1 | `full` | 464 | $13,747,069$ | $13,751,730$ | 0.0740 | 73.68% (42/57) | 87.72% (50/57) |
| `gmm_k2_full` | 2 | `full` | 929 | $5,963,565$ | $5,972,899$ | 0.0750 | 75.44% (43/57) | **89.47% (51/57)** |
| `gmm_k2_diag` | 2 | `diag` | 88 | $11,967,086$ | $11,968,262$ | 0.0455 | 57.89% (33/57) | 85.96% (49/57) |
| `gmm_k3_full` | 3 | `full` | 1,394 | $3,907,437$ | $3,921,442$ | 0.0828 | 75.44% (43/57) | 87.72% (50/57) |
| **`gmm_k4_full`** | **4** | **`full`** | **1,859** | **-4,843,306** | **-4,824,630** | **0.4840** | **75.44% (43/57)** | **82.46% (47/57)** |
| `gmm_k5_full` | 5 | `full` | 2,324 | -4,778,900 | -4,755,551 | 0.2192 | 73.68% (42/57) | 85.96% (49/57) |
| `gmm_k5_tied` | 5 | `tied` | 584 | $12,018,803$ | $12,024,670$ | 0.1770 | 75.44% (43/57) | 85.96% (49/57) |

#### 4.1.3 Theoretical Findings from Model Selection
1. **Dominance of Full Covariance:** Discarding feature correlations (`diag`) degraded BIC ($\ge 10.7\text{M}$) and collapsed AP to $\approx 0.045 - 0.059$. Financial transactions exhibit strong correlations across PCA dimensions and Amount that cannot be assumed independent.
2. **Continuous Negative BIC:** In continuous multivariate density models, probability density $p(x)$ can exceed 1 in concentrated subspaces. When likelihood $\hat{L} > 1$, $-2\ln\hat{L}$ becomes negative and dominates parameter penalties; lower (more negative) values indicate tighter probability density fit.
3. **Subspace Flatness & Seed Sensitivity:** In `gmm_k4_full`, the legitimate cardholder population concentrates on an $\approx 17$-dimensional subspace in $\mathbb{R}^{29}$, causing orthogonal variance to hit the `reg_covar = 1e-6` floor. Across alternative random seeds (0–4), validation AP ranged from approximately **$0.069$ to $0.102$**, while the saved seed-42 artifact achieved **$0.4840$**. The broader range of $0.045–0.059$ applies to diagonal models.

---

### 4.2 Paradigm 2: Deep Autoencoder Neural Reconstruction (`src/autoencoder_model.py`)
The deep autoencoder learns a non-linear manifold mapping that compresses transactions through a low-dimensional bottleneck and reconstructs them:
* **Architecture:** Symmetric Multi-Layer Perceptron (PyTorch):
  $$\text{Encoder: } 29 \to 20 \to 10 \to 5 \quad \mid \quad \text{Decoder: } 5 \to 10 \to 20 \to 29$$
* **Layer Activations:** `ReLU` on hidden layers; **Linear** output layer (essential because standardized Z-score input features span unbounded real domains $[-\infty, +\infty]$).
* **Trainable Parameters:** $1,929$ weights and biases.
* **Loss Function:** Mean Squared Error ($\text{MSELoss}$).
* **Optimization:** Adam optimizer ($\text{lr} = 10^{-3}$, batch size $256$, CPU-based deterministic training, seed $42$).
* **Legitimate-Only Training & Early Stopping:**
  * Trained strictly on $170,528$ legitimate training rows (30 epochs, duration $10.24\text{s}$ on CPU).
  * Early stopping and best checkpoint restoration monitored reconstruction loss strictly on the **$56,900$ legitimate validation rows** (`y_val == 0`). Validation fraud cases were never exposed during loss monitoring. Best validation loss: $0.701421$.
* **Anomaly Scoring Formula:**
  $$\mathrm{Score}_{\mathrm{AE}}(x) = \frac{1}{29} \sum_{j=1}^{29} (x_j - \hat{x}_j)^2$$

---

## 5. Operational Triage Evaluation Framework

The evaluation module (`src/triage_evaluation.py`) translates continuous anomaly scores into operational triage metrics:

### 5.1 Strict Record Alignment by `row_id`
Scores are joined to ground-truth labels using a one-to-one merge on stable `row_id` keys (`align_scores_with_labels`). Positional array alignment is never assumed, preventing label-permutation errors.

### 5.2 Two Distinct Operational Operating Modes
```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        OPERATIONAL TRIAGE MODES IN PRACTICE                           │
├────────────────────────────────────────────────────┬───────────────────────────────────┤
│ Mode 1: False Positive Rate Constraint (FPR ≤ α)   │ Mode 2: Fixed Review Budget (Top-N)│
├────────────────────────────────────────────────────┼───────────────────────────────────┤
│ • Limits cardholder friction                       │ • Matches investigator staffing   │
│ • Denominator: strictly legitimate rows            │ • Dynamic queue: top N scores     │
│ • Threshold: lowest score keeping FPR ≤ α          │ • Alert volume: exactly N alerts  │
│ • Alert count: varies with incoming fraud density  │ • FPR: varies with fraud density  │
│ • Conservative tie-handling rule: tied groups that │ • Guarantees human investigators  │
│   exceed budget remain unflagged                   │   are never overwhelmed           │
└────────────────────────────────────────────────────┴───────────────────────────────────┘
```

### 5.3 Why Accuracy and F1-Score are Misleading
* **Accuracy:** A model predicting zero fraud achieves $99.87\%$ accuracy on test data while catching zero fraud cases.
* **F1-Score:** Standard F1 assumes an arbitrary $0.5$ decision threshold that has no meaning for unbounded continuous anomaly scores ($-\ln p(x) \in [28, 5200+]$, $\text{MSE} \in [0.05, 190+]$).
* **Selected Primary Metrics:**
  * **Average Precision (AP / PR-AUC):** Area under the Precision-Recall curve across all operational thresholds (random baseline $\approx 0.0013$).
  * **Recall @ Fixed FPR ($\le 1\%, \le 5\%$):** Fraud catch rate when false alarms are strictly capped.
  * **Top-N Alert Capacity (N = 50, 100, 200, 500):** Realized precision and fraud yield under strict workforce constraints.

---

## 6. Final Held-Out Test Evaluation Benchmark Results

The final benchmark evaluated frozen candidate models on the chronological held-out test partition ($56,962$ transactions, $75$ fraud cases, $0.1317\%$ prevalence) using **frozen operating thresholds derived from the validation set**.

### 6.1 Primary Test Benchmark Comparison Table
*Source:* `results/tables/final_test_model_comparison.csv` and `results/tables/final_test_threshold_metrics.csv`

| Metric / Dimension | Frozen GMM Baseline | Frozen GMM $K=4$ Full | Frozen Autoencoder Baseline |
| :--- | :---: | :---: | :---: |
| **Model Artifact** | `models/gmm_baseline.joblib` | `models/candidates/gmm_k4_full.joblib` | `models/autoencoder_baseline.pt` |
| **Model Type** | Unimodal Parametric Density | Multi-Component Density | Deep Neural Reconstruction |
| **Trainable Parameters** | **464** | 1,859 | 1,929 |
| **Training Time (CPU)** | **~1.20 s** | ~11.50 s | ~10.24 s |
| **Average Precision (AP)** | **0.1610** | **0.5770** | **0.1793** |
| └ 95% Bootstrap CI | [0.1208, 0.2209] | [0.4588, 0.6973] | [0.1318, 0.2447] |
| **ROC-AUC** | **0.9622** | 0.9156 | 0.9436 |
| **Recall @ Frozen 1% Val Threshold** | **72.00% (54 / 75)** | **76.00% (57 / 75)** | **73.33% (55 / 75)** |
| └ Frozen Threshold Value ($\tau_{1\%}$) | 158.814 (NLL) | 66.436 (NLL) | 2.651 (MSE) |
| └ Realized Test FPR | 0.6645% (378 FP) | 0.5889% (335 FP) | 0.5590% (318 FP) |
| └ Total Alerts & Precision | 432 alerts (12.50%) | 392 alerts (14.54%) | 373 alerts (14.75%) |
| **Recall @ Frozen 5% Val Threshold** | **84.00% (63 / 75)** | **80.00% (60 / 75)** | **80.00% (60 / 75)** |
| └ Frozen Threshold Value ($\tau_{5\%}$) | 60.726 (NLL) | 50.241 (NLL) | 1.341 (MSE) |
| └ Realized Test FPR | 4.0132% (2,283 FP) | 3.6845% (2,096 FP) | 3.6019% (2,049 FP) |
| └ Total Alerts & Precision | 2,346 alerts (2.69%) | 2,156 alerts (2.78%) | 2,109 alerts (2.84%) |

### 6.2 Top-N Analyst Review Capacity Performance
*Source:* `results/tables/final_test_review_capacity.csv`

| Analyst Review Budget | Evaluation Metric | GMM Baseline ($K=1$) | GMM $K=4$ Full | Autoencoder Baseline |
| :--- | :--- | :---: | :---: | :---: |
| **Top 50 Alerts** | Frauds Intercepted | 9 / 75 (12.00%) | **39 / 75 (52.00%)** | 11 / 75 (14.67%) |
| | Realized Precision | 18.00% | **78.00%** | 22.00% |
| | False Positive Count | 41 | **11** | 39 |
| **Top 100 Alerts** | Frauds Intercepted | 28 / 75 (37.33%) | **49 / 75 (65.33%)** | 31 / 75 (41.33%) |
| | Realized Precision | 28.00% | **49.00%** | 31.00% |
| | False Positive Count | 72 | **51** | 69 |
| **Top 200 Alerts** | Frauds Intercepted | 38 / 75 (50.67%) | **55 / 75 (73.33%)** | 40 / 75 (53.33%) |
| | Realized Precision | 19.00% | **27.50%** | 20.00% |
| | False Positive Count | 162 | **145** | 160 |
| **Top 500 Alerts** | Frauds Intercepted | 56 / 75 (74.67%) | **57 / 75 (76.00%)** | 56 / 75 (74.67%) |
| | Realized Precision | 11.20% | **11.40%** | 11.20% |
| | False Positive Count | 444 | **443** | 444 |

### 6.3 Validation vs. Test Generalization Findings
*Source:* `results/tables/final_test_val_vs_test_comparison.csv`

1. **Increase in Average Precision Across Models:** Average precision increased across all models on test data ($0.074 \to 0.161$ for baseline, $0.076 \to 0.179$ for AE, $0.484 \to 0.577$ for $K=4$). The test partition had a higher fraud prevalence ($0.1317\%$ vs. $0.1001\%$, a $+31.6\%$ relative increase). While differences in prevalence and the ranking of individual fraud transactions may contribute to the observed change, the available analysis does not establish a causal explanation or confirm formal fraud clustering.
2. **Threshold Conservatism (Zero Runaway False Positives):** Freezing validation thresholds proved safe and conservative: actual test FPRs were **$0.56\%$ to $0.66\%$** at the 1% target, and **$3.60\%$ to $4.01\%$** at the 5% target.
3. **The GMM $K=4$ Dominance at Tight Budgets:** At Top 50 alerts, $K=4$ captured **$52.00\%$ of all test fraud with $78.00\%$ precision**, intercepting nearly $4\times$ more fraud than the baseline and Autoencoder.
4. **Baseline Recall at Wide Budgets:** At the 5% FPR operating point, the single-component GMM Baseline captured **$84.00\%$ of frauds (63/75)**, surpassing both $K=4$ (60/75, 80.00%) and the Autoencoder (60/75, 80.00%). Unimodal quadratic distance flags multi-feature outliers without mixture-weight dilution.

---

## 7. Recommended Tiered Production Deployment Architecture

Combining empirical findings yields a practical **two-tiered hybrid triage pipeline**:

```
                       INCOMING REAL-TIME TRANSACTIONS
                                      │
                                      ▼
             [ PREPROCESSING: log1p(Amount) + Frozen StandardScaler ]
                                      │
                                      ▼
      ┌───────────────────────────────────────────────────────────────┐
      │  TIER 1: RAPID HIGH-PRECISION QUEUE                           │
      │  Model: Frozen GMM K=4 Full Covariance Artifact               │
      │  Policy Type: Dynamic Review Capacity (Top 50–100 alerts)     │
      │  Validation Calibration Reference: τ₁₀₀ = 135.78              │
      │  Observed Test Set Cutoff at Rank 100: Score = 143.26         │
      │  Test Yield: 49% to 78% Precision (Intercepts 52%–65% Frauds) │
      │  Action: Immediate challenge / step-up card verification      │
      └───────────────────────────────┬───────────────────────────────┘
                                      │
                                      ▼
      ┌───────────────────────────────────────────────────────────────┐
      │  TIER 2: COMPREHENSIVE HIGH-RECALL SAFETY NET                 │
      │  Model: Frozen GMM Baseline (K=1 Full Covariance)             │
      │  Policy Type: Fixed Validation-Derived Threshold              │
      │  Operating Threshold: Score ≥ 60.726 (Frozen 5% Val FPR)      │
      │  Test Yield: 84.00% Overall Fraud Recall (Intercepts 63/75)  │
      │  Realized Test FPR: 4.01% (Safe operational workload)         │
      │  Action: Queued for secondary offline batch analyst review    │
      └───────────────────────────────────────────────────────────────┘
```

> **Operational Threshold Provenance Note:**  
> The score threshold $143.26$ represents the empirical anomaly score of the 100th-ranked transaction observed on the held-out test partition (`final_test_review_capacity.csv`). It is a retrospective descriptive statistic of that test ranking, not a pre-selected operational threshold. The prospective validation-derived Top-100 threshold was $\tau_{100} = 135.78$ (`reports/triage_evaluation.md`). In practice, Tier 1 is implemented as a capacity-constrained dynamic ranking queue (Top-N alerts per period) rather than a static score cutoff.

---

## 8. Software Architecture & Verification

### 8.1 Repository Layout
```text
card-fraud-triage/
├── data/
│   ├── raw/creditcard.csv                  # 284,807 transactions (Git LFS)
│   ├── splits/split_manifest.csv           # Chronological Strategy B mapping
│   └── processed/                          # Scaled arrays, scaler.joblib, scores
├── models/
│   ├── gmm_baseline.joblib                 # Frozen K=1 full GMM (464 params)
│   ├── autoencoder_baseline.pt             # PyTorch MLP weights (1,929 params)
│   └── candidates/gmm_k4_full.joblib       # Primary multi-component GMM (1,859 params)
├── notebooks/                              # 01_eda, 02_preproc, 03_gmm, 04_experiments
├── reports/                                # 8 rigorous technical markdown audits
├── results/                                # Serialized tables (.csv/.json) and figures (.png)
├── scripts/                                # End-to-end execution runners
├── src/                                    # Reusable modular ML engineering libraries
└── tests/                                  # 61 automated unit and regression tests
```

### 8.2 Automated Test Suite Verification (61 / 61 Passing)
Verified with command:
```bash
.venv/bin/python -m unittest discover tests
```
* `tests/test_gmm_baseline.py` (11 tests): Legitimate-only training, feature order invariance, convergence, reproducibility.
* `tests/test_autoencoder.py` (8 tests): Symmetric PyTorch MLP, MSE scoring, early stopping, exclusion of training fraud.
* `tests/test_model_selection.py` (2 tests): Candidate fitting, synthetic triage metrics.
* `tests/test_triage_evaluation.py` (27 tests): Exact tie-handling, PR curves, alert budgets, shuffle resistance.
* `tests/test_final_test_evaluation.py` (13 tests): Test manifest conformance, frozen thresholds, zero scaler/model refit.
* **Output:** `Ran 61 tests in 14.993s. OK.`

---

## 9. College Viva & Technical Defense Master Guide

### Comprehensive Q&A for Technical Examiners

#### Q1: "Why did you use unsupervised anomaly detection instead of a supervised classifier like XGBoost or Random Forest?"
> **Examiner Answer:**
> *"Supervised classifiers require balanced ground truth and overfit heavily to known historical fraud patterns. In credit cards, fraud is adversarial and continually mutating: fraudsters invent new attack vectors that do not resemble past fraud. By modeling normal cardholder behavior strictly on legitimate transactions (`Class == 0`), our GMM and Autoencoder flag anything that departs from legitimate norms, making them inherently capable of detecting zero-day, unobserved fraud mechanisms without target contamination."*

#### Q2: "How did you prevent data leakage during preprocessing and model training?"
> **Examiner Answer:**
> *"We enforced four strict anti-leakage barriers:*
> *1. **Chronological Splitting (Strategy B):** We split data strictly by transaction timestamp (60% Train / 20% Val / 20% Test) rather than random splitting, preventing future-to-past time travel leakage.*
> *2. **Boundary Burst Alignment:** We adjusted the train cutoff by 4 rows to index 170,888 to ensure a burst of transactions sharing identical timestamps (`120,396.0s`) was kept together.*
> *3. **Legitimate-Only Fitting:** The `StandardScaler`, the GMM, and the Autoencoder were fitted strictly on legitimate training rows (`y_train == 0`). All 360 training frauds were filtered out.*
> *4. **Frozen Validation Thresholds:** Operating thresholds were selected strictly on the validation set and frozen before evaluating test data once with zero test refitting."*

#### Q3: "Why did you retain duplicate rows instead of dropping them?"
> **Examiner Answer:**
> *"Exact row duplicates in transaction logs originate from automated bot attack bursts (rapid-fire carding) or payment gateway retry logging. Without server network telemetry, dropping duplicate rows would delete 19 of our 492 fraud cases (nearly 4% of our positive fraud data). Furthermore, our chronological split analysis verified that all 773 duplicate groups share identical timestamps and fall 100% within a single split boundary. Retaining them preserves rare fraud signals while causing zero train-test leakage."*

#### Q4: "Why did AIC and BIC become negative during GMM model selection?"
> **Examiner Answer:**
> *"In discrete probability models, probabilities are bounded in $[0, 1]$, so $\ln L \le 0$ and $-2\ln L \ge 0$. However, in continuous multivariate Gaussian distributions, probability density $p(x)$ can exceed 1 in tight feature clusters. When likelihood $\hat{L} > 1$, the $-2\ln\hat{L}$ term turns negative and outweighs the positive parameter penalty term. Lower (more negative) BIC values still indicate superior goodness-of-fit."*

#### Q5: "Why did you evaluate Recall at 1% FPR and Top-N alert capacity instead of Accuracy or F1-Score?"
> **Examiner Answer:**
> *"With 0.13% fraud prevalence, a dummy model predicting zero fraud achieves 99.87% accuracy. Similarly, standard F1-score assumes an arbitrary 0.5 decision threshold that has no meaning for unbounded continuous anomaly scores. In a commercial fraud operations center, human analysts can only investigate a fixed number of alerts per shift. Evaluating Recall @ 1% FPR and Top-N alert capacity measures real operational utility: exactly how much fraud is intercepted within the bank's human investigation budget."*

#### Q6: "Why did GMM $K=4$ perform so well under seed 42, and what are its caveats?"
> **Examiner Answer:**
> *"In $K=4$ full covariance, the legitimate transaction distribution concentrates on an $\approx 17$-dimensional manifold in $\mathbb{R}^{29}$. Orthogonal directions hit the `reg_covar = 1e-6` floor, creating a massive penalty for anomalies leaving this subspace. However, because multi-component GMMs have non-convex EM optimization surfaces, alternative random seeds 0–4 converged to different local modes with lower validation AP ($0.069$ to $0.102$). Thus, while the specific frozen seed-42 artifact generalized effectively on the test set (0.5770 AP, 78% precision at Top 50), the baseline single-component GMM remains essential as a robust safety net because $K=1$ has a globally convex, well-conditioned formulation."*

---

## 10. Team Responsibilities & Deliverable Verification

| Member | Name | Designated Project Responsibilities | Verified Deliverables & Artifacts | Status |
| :---: | :--- | :--- | :--- | :---: |
| **Member 1** | **Rudraksh** | GMM baseline implementation, leak-free training protocol, anomaly scoring, Autoencoder core model implementation, PyTorch training loop, unit test suite integration. | `src/gmm_model.py`, `src/scoring.py`, `src/autoencoder_model.py`, `models/gmm_baseline.joblib`, `models/autoencoder_baseline.pt`, `tests/test_gmm_baseline.py`, `tests/test_autoencoder.py`. | **Verified & Completed** |
| **Member 2** | **Bhargav** | Dataset verification, exploratory data analysis, chronological split strategy (Strategy B), feature engineering (`log1p(Amount)`), leak-free `StandardScaler`. | `data/splits/split_manifest.csv`, `src/preprocessing.py`, `notebooks/01_dataset_verification_eda.ipynb`, `reports/data_cleaning_decisions.md`, `reports/split_strategy.md`. | **Verified & Completed** |
| **Member 3** | **Kaustubh** | GMM model selection: component counts ($K \in \{2..5\}$), covariance types (`full`, `diag`, `tied`), AIC/BIC analysis, numerical stability audit, candidate serialization. | `src/model_selection.py`, `scripts/run_model_selection.py`, `scripts/audit_stability.py`, `models/candidates/gmm_k4_full.joblib`, `reports/gmm_model_selection.md`. | **Verified & Completed** |
| **Member 4** | **Anurag** | Triage evaluation: Average Precision (PR-AUC), fixed-FPR recall, Top-N review capacity, tie-handling, cross-model comparison, final held-out test evaluation benchmark. | `src/triage_evaluation.py`, `scripts/run_triage_evaluation.py`, `scripts/run_final_test_evaluation.py`, `reports/triage_evaluation.md`, `reports/final_test_evaluation.md`, `results/tables/final_test_*.csv`. | **Verified & Completed** |
| **Member 5** | **Pratyush** | Autoencoder validation evaluation and comparison, project documentation consolidation, final academic report synthesis, and presentation slide deck. | `reports/autoencoder_baseline.md`, `reports/gmm_vs_autoencoder.md`, `README.md`, `Project_Report.md`. | **Verified & Completed** |

---

## 11. Conclusion & Submission Sign-Off
This project successfully designed, implemented, mathematically audited, and empirically validated a dual-paradigm machine learning framework for credit card fraud triage. By maintaining strict research integrity—fitting exclusively on legitimate transactions, aligning chronological timestamp boundaries, avoiding feature/target leakage, and freezing validation thresholds—the system demonstrated robust generalization on 56,962 unseen test transactions.

The frozen multi-component GMM ($K=4$ full covariance) provides an effective high-precision queue at low review capacities (achieving **$78.00\%$ precision at Top 50 alerts** on this test partition), while the frozen single-component GMM baseline provides robust high recall across broad operational thresholds (delivering **$84.00\%$ total fraud recall** at a conservative $4.01\%$ realized test FPR). The resulting tiered architecture bridges the gap between academic unsupervised anomaly detection and real-world banking operations.
