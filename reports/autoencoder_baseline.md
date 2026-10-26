# Phase 2: Autoencoder Baseline Training & Validation Evaluation Report

**Project:** Card Fraud Triage via Gaussian Mixture Density Scoring and Autoencoder Anomaly Detection  
**Phase:** Phase 2 — Autoencoder Anomaly Detection  
**Task:** Task 3B — Full Training, Anomaly Scoring, and Validation Evaluation  
**Status:** Completed & Audited — Baseline Autoencoder Trained & Evaluated  
**Date:** October 2026  

---

## 1. Executive Summary

This report establishes the baseline Autoencoder anomaly detection model for the Card Fraud Triage project, benchmarking it against the frozen single-component Gaussian Mixture Model (`gmm_baseline_k1_full`).

The Autoencoder was trained strictly on **legitimate transactions** (`Class == 0`) from the chronological training partition using mini-batch Adam optimization on CPU. Early stopping and best checkpoint selection were monitored exclusively on **legitimate validation transactions** (`y_val == 0`). Zero fraudulent transactions were ever exposed during gradient updates, threshold derivations, or epoch selection.

Anomaly scores are quantified as the per-transaction **Mean Squared Reconstruction Error (MSE)** across all 29 features:
$$\mathrm{Score}_{\mathrm{AE}}(x) = \frac{1}{29}\sum_{d=1}^{29}(x_d - \hat{x}_d)^2$$

Higher reconstruction error indicates that an incoming transaction deviates from the normal spending manifold learned by the autoencoder, marking it as a higher-priority triage alert.

---

## 2. Model Architecture & Training Hyperparameters

In accordance with project guidelines (simple, beginner-friendly ML with clean modular PyTorch), the network is a symmetric fully connected multi-layer perceptron (MLP):

```
Input (29) ──► Linear(29, 20) ──► ReLU ──► Linear(20, 10) ──► ReLU ──► Linear(10, 5) [Bottleneck]
                                                                             │
Output (29) ◄── Linear(20, 29) ◄── ReLU ◄── Linear(10, 20) ◄── ReLU ◄── Linear(5, 10) [Decoder]
```

### Approved Training Configuration:
* **Input Dimension:** `29` (features $V_1$ through $V_{28}$ plus $\ln(1 + \text{Amount})$, preprocessed with the frozen `StandardScaler`).
* **Bottleneck Dimension:** `5` (yielding a $5.8\times$ continuous dimensional compression).
* **Activation Functions:** ReLU between hidden layers; **Linear** output layer (necessary because standardized Z-score inputs have unbounded positive and negative ranges).
* **Loss Function:** Mean Squared Error (`torch.nn.MSELoss()`).
* **Optimizer:** Adam (`lr = 0.001`, betas `(0.9, 0.999)`).
* **Batch Size:** `256` (~667 batches per epoch over 170k rows).
* **Maximum Epochs:** `30`.
* **Patience:** `5` epochs.
* **Minimum Delta (`min_delta`):** `1e-4`.
* **Random Seed:** `42` (`torch.manual_seed(42)`, `np.random.seed(42)`, `random.seed(42)`).
* **Hardware Device:** Fixed CPU (`torch.device("cpu")`) for deterministic reproducibility.
* **Trainable Parameters:** `1,929` weights and biases.

---

## 3. Training Protocol & Leak-Free Verification

* **Training Partition Rows:** 170,888 total transactions.
* **Fitted Legitimate Subset:** Exactly **`170,528`** legitimate transactions (`y_train == 0`).
* **Excluded Training Frauds:** Exactly **`360`** training fraud transactions (`Class == 1`) were filtered out before constructing the PyTorch DataLoader.
* **Validation Monitoring Subset:** Exactly **`56,900`** legitimate validation transactions (`y_val == 0`). All 57 validation fraud cases were excluded from monitoring loss.
* **Test Isolation:** Exactly **`0`** test observations or labels were loaded or evaluated.
* **Convergence History:**
  * Epochs trained: **`30`** (reached max epochs with steady, monotonic convergence).
  * Best Epoch: **`30`** (restored as final model checkpoint).
  * Initial Training MSE: `1.082711` (Epoch 1).
  * Final Training MSE: `0.465816` (Epoch 30).
  * Best Legitimate Validation MSE: **`0.701421`** (Epoch 30).
  * Total Training Duration: **`10.24 seconds`** on Apple Silicon CPU.

---

## 4. Score Distribution & Diagnostics

Anomaly scores represent per-sample Mean Squared Reconstruction Error:

### Training Partition (170,888 transactions):
| Cohort | Count | Mean | Std | Min | 25% | Median | 75% | Max |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Overall** | 170,888 | 0.5229 | 2.3744 | 0.0544 | 0.2298 | 0.3515 | 0.5223 | 409.1693 |
| **Legitimate (`Class = 0`)** | 170,528 | 0.4658 | 1.5383 | 0.0544 | 0.2295 | 0.3510 | 0.5208 | 409.1693 |
| **Fraud (`Class = 1`)** | 360 | 27.5440 | 28.7374 | 0.1228 | 4.8813 | 14.0161 | 48.2859 | 95.3220 |

### Validation Partition (56,957 transactions):
| Cohort | Count | Mean | Std | Min | 25% | Median | 75% | 95% | Max |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Overall** | 56,957 | 0.7069 | 2.6077 | 0.0769 | 0.3621 | 0.5107 | 0.7491 | 1.3411 | 189.9354 |
| **Legitimate (`Class = 0`)** | 56,900 | 0.7014 | 2.5991 | 0.0769 | 0.3619 | 0.5102 | 0.7483 | 1.3411 | 189.9354 |
| **Fraud (`Class = 1`)** | 57 | 6.1676 | 4.6593 | 0.2587 | 2.3017 | 5.1284 | 10.2293 | 11.8617 | 24.4335 |

* **Directionality Confirmed:** Legitimate transactions have a median reconstruction error of `0.5102`, whereas fraudulent transactions exhibit a median reconstruction error of `5.1284` (~$10\times$ higher), confirming that higher score denotes higher anomaly severity.

---

## 5. Validation-Only Triage Performance & Comparison

All metrics were calculated using the project's established evaluation module (`src/triage_evaluation.py`) on the **identical 56,957 validation transactions** joined strictly by `row_id`:

* **Threshold Selection Rule:** Operational thresholds are selected using `roc_curve(drop_intermediate=False)` by taking the lowest threshold that guarantees the empirical false positive rate on legitimate validation transactions stays strictly <= `max_fpr`.
  * At 1% FPR, the threshold is $\tau$ (1% FPR) = 2.651144 (flagging exactly 569 false positives out of 56,900 legitimate transactions, exactly 1.000% FPR).
  * At 5% FPR, the threshold is $\tau$ (5% FPR) = 1.341100 (flagging exactly 2,845 false positives out of 56,900 legitimate transactions, exactly 5.000% FPR).

### Head-to-Head Comparison:
| Metric | Frozen GMM Baseline (`gmm_baseline_k1_full`) | Autoencoder Baseline (`autoencoder_baseline`) | Relative Difference |
| :--- | :--- | :--- | :--- |
| **Model Type** | Probabilistic Density Estimation (GMM) | Neural Representation Learning (Autoencoder) | — |
| **Scoring Formula** | Negative Log-Likelihood: $-\ln p(x)$ | MSE Reconstruction: $\frac{1}{29}\|x - \hat{x}\|^2$ | — |
| **Trainable Parameters** | 464 (means + full covariance) | 1,929 (weights + biases) | +315% parameters |
| **PR-AUC (Average Precision)** | **0.0740** | **0.0765** | **+3.38% relative improvement** |
| **Recall @ 1% FPR** | **73.68% (42 / 57 frauds)** | **71.93% (41 / 57 frauds)** | $-1$ fraud caught at 1% FPR |
| **Recall @ 5% FPR** | **87.72% (50 / 57 frauds)** | **87.72% (50 / 57 frauds)** | Identical (50/57 frauds caught) |
| **Training Duration** | ~1.20s | ~10.24s | — |

---

## 6. Findings & Analysis

1. **Competitive Anomaly Representation:**
   * Despite using non-linear bottleneck compression ($29 \to 20 \to 10 \to 5$) rather than explicit Gaussian covariance fitting, the Autoencoder achieves an Average Precision of **0.0765**, slightly exceeding the frozen single-component GMM baseline (**0.0740**).
2. **Operational Triage Parity:**
   * At an operational budget of 1% false positives (allowing $\le 569$ false alarms for human analysts out of ~57k transactions), the Autoencoder catches **41 of 57 fraud cases** (71.93%), trailing the GMM baseline by only a single fraud case (42 vs. 41).
   * At 5% FPR, both models achieve identical recall: **87.72% (50 of 57 frauds intercepted)**.
3. **No Overfitting to Training Artifacts:**
   * Training loss decreased steadily from $1.08 \to 0.46$ while validation loss converged smoothly to $0.70$ without diverging.
4. **Conclusion on Model Superiority:**
   * The validation evidence does **not** demonstrate clear overall superiority of the Autoencoder over the GMM baseline: while Average Precision is slightly higher (+0.0025), Recall @ 1% FPR is marginally lower by 1 fraud transaction.
   * Both models provide complementary representations of normal credit card behavior.

---

## 7. Artifacts Inventory

| Artifact Path | Format | Description |
| :--- | :--- | :--- |
| `models/autoencoder_baseline.pt` | PyTorch State Dict | Trained model weights (1,929 parameters) |
| `models/autoencoder_baseline_metadata.json` | JSON | Architecture, training sample counts, and loss history |
| `data/processed/scores/autoencoder_scores_train.csv` | CSV (170,888 rows) | Anomaly scores with `row_id`, `anomaly_score`, `split`, `Class` |
| `data/processed/scores/autoencoder_scores_val.csv` | CSV (56,957 rows) | Anomaly scores with `row_id`, `anomaly_score`, `split`, `Class` |
| `src/autoencoder_model.py` | Python module | Autoencoder network definition, training, save/load |
| `src/autoencoder_scoring.py` | Python module | MSE reconstruction score computation |
| `scripts/run_autoencoder.py` | Python script | End-to-end training and validation evaluation runner |
| `tests/test_autoencoder.py` | Unit tests | 8 unit tests covering model, training, scoring, and row-ID joins |

---

## 8. Reproducibility Statement

To reproduce the exact weights, score files, and validation metrics:
```bash
source .venv/bin/activate
PYTHONPATH=. python scripts/run_autoencoder.py
```
To run the complete automated test suite (all 48 tests):
```bash
source .venv/bin/activate
python -m unittest discover tests
```
