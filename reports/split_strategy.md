# Chronological Split Strategy & Manifest Documentation

**Project:** Card Fraud Triage via Gaussian Mixture Density Scoring and Autoencoder Anomaly Detection  
**Role:** Member 2 — Data Analysis, Data Quality, and Preprocessing  
**Task:** Task 3B — Implement Chronological Split Manifest  
**Status:** Approved & Verified (Strategy B: Timestamp Boundary Alignment)

---

## 1. Overview & Strategy Justification

In financial fraud detection, transactions arrive sequentially over time. Random train/test splits (such as standard `train_test_split`) are fundamentally flawed for this domain because:
1. **Time-Travel Leakage:** Training on future transactions to predict past transactions violates real-world causality.
2. **Evaluation Realism:** Anomaly detection models must be deployed prospectively to triage future unseen transactions.

To reflect production reality while preventing duplicate data leakage, we implemented **Strategy B: Chronological Split with Timestamp Boundary Alignment**.

---

## 2. Boundary Alignment Details

* **Total Dataset Size:** 284,807 rows (`Time` range: 0.0s to 172,792.0s ~ 48 hours).
* **Sorting Method:** `Time` ascending using a **stable sort** (`kind='stable'`), which guarantees deterministic tie-breaking by preserving original row order for identical timestamps.
* **Row Identifier:** Stable `row_id` (0 to 284,806) assigned based on the original raw CSV row order before sorting.

### The 60% Boundary Adjustment (Train / Validation)
* A strict 60% integer cutoff lands at index `170,884`.
* However, exactly 6 distinct transactions share `Time = 120,396.0s` (indices `170,882` to `170,887`).
* A strict cut would split this single second across Train (2 rows) and Validation (4 rows).
* **Approved Adjustment:** We shifted the boundary by 4 rows to index **`170,888`**, keeping all 6 transactions sharing `Time = 120,396.0s` together in the **training partition**.
* All 6 transactions are legitimate (`Class = 0`), so this shift does not alter fraud counts.

### The 80% Boundary (Validation / Test)
* Kept at index **`227,845`**.
* The transition between index `227,844` (`Time = 145,247.0s`) and `227,845` (`Time = 145,248.0s`) exhibits a clean 1-second step. No transactions share a timestamp across this boundary.

---

## 3. Actual Partition Statistics

| Partition | Integer Index Range (Sorted) | Row Count | Percentage | Time Range (s) | Duration (Hours) | Fraud Count | Fraud Rate |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | `0` to `170,887` | **170,888** | 60.001% | `0.0` – `120,396.0` | ~33.44 hrs | **360** | **0.2107%** |
| **Validation** | `170,888` to `227,844` | **56,957** | 19.998% | `120,397.0` – `145,247.0` | ~6.90 hrs | **57** | **0.1001%** |
| **Test** | `227,845` to `284,806` | **56,962** | 20.000% | `145,248.0` – `172,792.0` | ~7.65 hrs | **75** | **0.1317%** |
| **Total** | `0` to `284,806` | **284,807** | 100.000% | `0.0` – `172,792.0` | ~48.00 hrs | **492** | **0.1727%** |

---

## 4. Manifest File Specification

The split manifest is saved at:
[`data/splits/split_manifest.csv`](file:///Users/ShreePatil/Downloads/card-fraud-triage-ml/data/splits/split_manifest.csv)

* **Total Lines:** 284,808 (1 header line + 284,807 data rows)
* **Schema:**
  * `row_id` (integer: 0 to 284,806): The exact original 0-indexed row number from `data/raw/creditcard.csv`.
  * `split` (string: `'train'`, `'validation'`, or `'test'`).
* **Usage:** Any downstream script can reconstruct the exact partitions without re-sorting by simply joining on `row_id`.

---

## 5. Automated Validation Results

All 8 automated integrity checks passed with zero errors:

1. **Total Rows:** Exactly 284,807 rows present in manifest.
2. **Row ID Uniqueness:** Exactly 284,807 unique `row_id` values matching `0` through `284,806`.
3. **Completeness:** 0 missing or null split assignments.
4. **Disjoint Sets:** Train, Validation, and Test row ID sets are strictly pairwise disjoint ($A \cap B = \emptyset$).
5. **Fraud Preservation:** Exactly 492 fraud cases preserved ($360 + 57 + 75 = 492$).
6. **Chronological Monotonicity:**
   * $\max(\text{Time}_{\text{train}}) = 120,396.0\text{s} \le \min(\text{Time}_{\text{val}}) = 120,397.0\text{s}$
   * $\max(\text{Time}_{\text{val}}) = 145,247.0\text{s} \le \min(\text{Time}_{\text{test}}) = 145,248.0\text{s}$
7. **Timestamp Alignment:** All 6 transactions at `Time = 120,396.0s` are confirmed in the `train` split.
8. **Duplicate Group Isolation:** 0 exact duplicate groups cross any split boundary. All 773 duplicate groups fall 100% within a single partition.

---

## 6. Limitations and College Viva Talking Points

1. **Why does the fraud rate vary between splits (0.21% in Train vs. 0.10% in Val and 0.13% in Test)?**
   > *"In real-world payment networks, fraud occurs in non-stationary bursts rather than at a uniform constant rate. Because we use a realistic chronological split rather than an artificial randomized split, the natural temporal variation of fraud is preserved. This provides a rigorous, leakage-free test of how our GMM and Autoencoder models generalize across varying fraud intensities."*
2. **Why use a manifest file instead of saving separate CSV files for each split?**
   > *"Saving separate full-data CSV files would duplicate 150 MB of data three times (nearly 450 MB of disk space). A lightweight split manifest (`row_id`, `split`) is an industry best practice: it guarantees 100% reproducibility, requires only ~3.5 MB of storage, and leaves the raw dataset pristine and immutable."*
3. **How do we know there is zero duplicate leakage?**
   > *"Every exact duplicate group shares the exact same timestamp ($t_{\min} = t_{\max}$). Our boundary analysis verified that no duplicate group occurs at the 60% boundary (`120,396s`) or the 80% boundary (`145,248s`). Therefore, all 773 duplicate groups land strictly inside a single partition."*
