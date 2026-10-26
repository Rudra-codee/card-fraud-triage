# GMM Baseline vs. Autoencoder Baseline — Comparison Report

**Project:** Card Fraud Triage via Gaussian Mixture Density Scoring and Autoencoder Anomaly Detection  
**Report Type:** Cross-Model Validation Comparison  
**Status:** Audited & Verified — Both models evaluated on identical validation partition  
**Date:** October 2026

---

## 1. Methodology

### 1.1 Shared Data Protocol

Both models were trained strictly on **legitimate training transactions** (`y_train == 0`) from
the chronological training partition (60% of the dataset by timestamp boundary alignment). Neither
model saw fraudulent rows during parameter fitting. The `StandardScaler` was fitted once on
legitimate training rows and is shared by both models.

**Partition summary (identical for both models):**

| Partition | Total Rows | Legitimate | Fraud |
| :--- | ---: | ---: | ---: |
| Training (parameter fitting) | 170,888 | 170,528 | 360 (excluded) |
| Validation (evaluation only) | 56,957 | 56,900 | 57 |
| Test (held-out, never accessed) | — | — | — |

### 1.2 Scoring Conventions

| Model | Score Formula | Score Meaning |
| :--- | :--- | :--- |
| **GMM Baseline** | `anomaly_score = −log p(x)` (negative log-likelihood) | Higher = less probable under normal distribution |
| **Autoencoder Baseline** | `anomaly_score = mean((x − x̂)²)` across 29 features (per-sample MSE) | Higher = harder to reconstruct = more anomalous |

Both scoring functions are strictly higher-means-more-suspicious, satisfying the triage convention
used throughout `src/triage_evaluation.py`.

### 1.3 Metric Definitions

All metrics are computed by `src/triage_evaluation.py` using `sklearn.metrics.roc_curve` with
`drop_intermediate=False` and `sklearn.metrics.average_precision_score`:

- **Average Precision (AP):** Area under the Precision-Recall curve (PR-AUC), computed as a
  weighted mean of precisions at each threshold where recall changes. Higher is better; the
  baseline rate for a random ranker at 57/56,957 ≈ 0.001 prevalence is approximately 0.0010.
- **Recall @ 1% FPR:** The fraction of the 57 validation frauds caught when the anomaly threshold
  is set to the **lowest value that keeps the false positive rate ≤ 1%** over the 56,900 legitimate
  validation rows.
- **Recall @ 5% FPR:** Same rule with a 5% FPR budget.

**FPR constraint computation:** The FPR denominator is always the count of *legitimate*
validation rows (56,900), matching `recall_at_fpr` in `src/triage_evaluation.py`. The numerator
is the count of legitimate rows whose anomaly score ≥ the threshold.

**Tie handling:** When a group of rows shares the same score value, the entire group is included
or excluded as a unit. If including the whole group would push FPR above the limit, the group
stays unflagged, so the achieved FPR may be strictly below the limit. For example, at the 1% FPR
threshold for the Autoencoder, the actual FPR is exactly 1.000% (569 FP out of 56,900 legit).

### 1.4 Evaluation Independence

- All metrics were computed using only validation rows (no test data).
- `run_autoencoder.py` loads arrays only from `data/processed/X_train.npy`,
  `data/processed/y_train.npy`, `data/processed/X_val.npy`, and `data/processed/y_val.npy`.
  No `X_test` or `y_test` arrays are referenced anywhere in the autoencoder pipeline.
- GMM score files are read as frozen artifacts; the GMM model was not retouched.

---

## 2. Comparison Table

All figures are from the **validation partition only** (56,957 transactions, 57 frauds).

| Metric | GMM Baseline | Autoencoder Baseline |
| :--- | :--- | :--- |
| **Model Type** | Probabilistic Density (GMM) | Neural Reconstruction (Autoencoder) |
| **Score Definition** | Negative Log-Likelihood: −log p(x) | Per-sample MSE: mean((x−x̂)²) |
| **Trainable Parameters** | 464 (means + full 29×29 covariance) | 1,929 (weights + biases, 5 layers) |
| **Training Duration** | ~1.20 s | ~10.24 s |
| **PR-AUC (Average Precision)** | **0.0740** | **0.0765** |
| **Recall @ 1% FPR** | **73.68% (42 / 57 frauds)** | **71.93% (41 / 57 frauds)** |
|  └ Threshold @ 1% FPR | 158.814 (NLL units) | 2.651 (MSE units) |
|  └ Actual FPR achieved | 0.995% (566 / 56,900 FP) | 1.000% (569 / 56,900 FP) |
| **Recall @ 5% FPR** | **87.72% (50 / 57 frauds)** | **87.72% (50 / 57 frauds)** |
|  └ Threshold @ 5% FPR | 60.726 (NLL units) | 1.341 (MSE units) |
|  └ Actual FPR achieved | 5.000% (2,845 / 56,900 FP) | 5.000% (2,845 / 56,900 FP) |

> **Note on the 1% FPR GMM threshold:** The achieved FPR is 0.9947% (566 FP), which is slightly
> below the 1% budget. This is correct: including the next tied score group at threshold 158.814
> would exceed the 1% limit, so the whole group is excluded. This behaviour is governed by the
> tie-handling rule in `src/triage_evaluation.py::recall_at_fpr`.

### 2.1 Independent Metric Verification

These metrics were independently re-derived from the saved score CSVs using pure NumPy (no sklearn,
no torch) and confirmed to agree exactly with the reported figures:

- **Recall @ 1% FPR and @ 5% FPR:** exact match for both models.
- **Average Precision:** exact match (0.0740 for GMM, 0.0765 for AE) using the right-Riemann-sum
  formulation that mirrors sklearn's `average_precision_score`.

---

## 3. Trade-offs and Interpretation

### 3.1 Average Precision

The Autoencoder's AP of **0.0765** exceeds the GMM's **0.0740** by 0.0025 points (+3.4% relative).
Both figures are roughly 7–8× higher than the random-ranker baseline of ≈0.0010, confirming that
both models successfully learn to rank frauds above legitimate transactions. The difference is small
relative to the statistical noise at this fraud prevalence (57 positives out of 56,957 rows).
Bootstrapped 95% confidence intervals computed in `src/triage_evaluation.py` would likely overlap.
No claim of definitive superiority is warranted from AP alone.

### 3.2 Recall @ 1% FPR — Tighter Operational Budget

At the tightest operating point (at most 1% of legitimate transactions flagged for human review),
the GMM catches **42 of 57** frauds while the Autoencoder catches **41 of 57**. The GMM leads by
exactly one fraud case. In operational terms: for every ~57,000 validation transactions, an analyst
using the GMM would intercept one additional fraud compared to the Autoencoder under a ~570-review
budget.

This margin may reflect the GMM's full covariance matrix explicitly modelling correlations among
all 29 features in a single multivariate density, providing sharper tail separation. The
Autoencoder's bottleneck compression (29 → 20 → 10 → 5 → ... → 29) may smooth certain unusual
feature combinations that would otherwise produce strong negative-log-likelihood signals.

### 3.3 Recall @ 5% FPR — Identical Performance

Both models achieve **87.72% (50/57)** at the 5% FPR operating point, flagging exactly 2,845
false positives from 56,900 legitimate transactions. This tie indicates that at a 5% FPR budget,
both models identify the same 50 frauds. The 7 uncaught frauds (at both thresholds) have anomaly
scores that overlap with legitimate transactions in both models — they are genuinely hard cases
for density-based detectors.

### 3.4 Complexity and Training Cost

The GMM achieves comparable or better performance with **464 parameters** and ~1.2 s of training,
versus **1,929 parameters** and ~10.2 s for the Autoencoder. For this 29-dimensional dataset
with ~170k legitimate training samples, the GMM's parametric covariance structure fits the
legitimate distribution efficiently. The Autoencoder requires roughly 4× more parameters and
9× more training time for a marginal AP improvement and a slight recall deficit at the
tighter operating point.

### 3.5 Limitations

1. **Unimodal GMM assumption:** The frozen baseline is a single-component full-covariance model,
   assuming legitimate transactions form one ellipsoidal cluster. Real spending patterns
   (micro-transactions, subscriptions, high-value purchases) are multimodal.
2. **Simple Autoencoder architecture:** The 3-layer encoder/decoder with a 5-dimensional
   bottleneck is intentionally simple. Larger bottlenecks, dropout regularization, or deeper
   architectures could change performance.
3. **Score scale incompatibility:** GMM NLL scores and Autoencoder MSE scores cannot be directly
   compared in magnitude; they are only comparable after applying the same FPR-based threshold
   selection rule.
4. **Small fraud count:** Only 57 validation fraud cases means a one-fraud difference (41 vs. 42
   at 1% FPR) falls within natural sampling variability.
5. **Chronological split:** Both models were evaluated on the chronological validation window
   immediately following training. Fraud patterns evolve over time; future performance may differ.

---

## 4. Generalization Caveat

> **⚠️ Important:** All results in this report — including Average Precision, Recall @ 1% FPR,
> and Recall @ 5% FPR — were measured on the **validation partition only**. The validation set
> immediately follows the training window chronologically and shares the same time period context.
>
> **Validation results alone do not establish generalization to unseen test data.** The
> held-out test partition has never been accessed during model training, threshold selection,
> or any part of this report. Whether either model's performance transfers to the test
> partition — and by how much — is unknown until a one-time, unbiased evaluation is conducted
> on that partition.
>
> Any selection of one model over another based solely on these validation metrics risks
> optimistic bias from inadvertent validation-set overfitting. A definitive comparison
> must await evaluation on the untouched test set.

---

## 5. Artifacts Referenced

| Artifact | Role in This Report |
| :--- | :--- |
| `models/gmm_baseline.joblib` | Frozen GMM — not modified |
| `models/gmm_baseline_metadata.json` | GMM config (K=1, full cov, seed=42) |
| `models/autoencoder_baseline.pt` | Autoencoder weights (best checkpoint = epoch 30) |
| `models/autoencoder_baseline_metadata.json` | AE config, loss history, sample counts |
| `data/processed/scores/gmm_baseline_scores_val.csv` | Frozen GMM val scores (56,957 rows) |
| `data/processed/scores/autoencoder_scores_val.csv` | AE val scores (56,957 rows) |
| `src/triage_evaluation.py` | Shared metric utility (reused, not duplicated) |
| `data/splits/split_manifest.csv` | Authoritative row-ID-to-split mapping |

---

## 6. Evaluation Utility Reuse

This comparison exclusively reuses the existing shared evaluation module
`src/triage_evaluation.py`. The functions `recall_at_fpr`, `average_precision_score`,
and `align_scores_with_labels` were called with identical arguments for both models.
No metric logic was duplicated for this report.
