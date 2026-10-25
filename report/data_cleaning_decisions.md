# Data Cleaning Decisions & Preprocessing Policy

**Project:** Card Fraud Triage via Gaussian Mixture Density Scoring and Autoencoder Anomaly Detection  
**Role:** Member 2 — Data Analysis, Data Quality, and Preprocessing  
**Task:** Task 2 — Review findings and establish data-cleaning rules  
**Status:** Approved by User — Option 3 Selected (All Duplicate Rows Retained; Raw Data 100% Intact)

---

## 1. Dataset Overview

* **File Path:** `data/raw/creditcard.csv`
* **File Format:** CSV (Comma-Separated Values)
* **Total Rows:** 284,807 transactions
* **Total Columns:** 31 (30 input features + 1 target column)
* **Memory Usage:** ~67.36 MB

All data-quality checks below were directly verified against `data/raw/creditcard.csv` using standard Pandas and NumPy.

---

## 2. Data Cleaning Decision Table

The table below summarizes every data-quality check, the evidence measured, the decided action, the reason, and whether any change was implemented.

| Issue | Evidence / Measured Count | Decision | Reason | Implemented? |
| :--- | :--- | :--- | :--- | :--- |
| **Missing Values** | `0` null values across all 31 columns | **Take no action** | No missing data exists. Imputation or row deletion is unnecessary. | **Yes** (No action needed) |
| **Infinite Values** | `0` positive (`+inf`) and `0` negative (`-inf`) values | **Take no action** | All numeric values are finite. No capping or replacement is needed. | **Yes** (No action needed) |
| **Data Types** | 30 columns are `float64`, 1 column (`Class`) is `int64` | **Take no action** | All columns are already properly parsed as numeric. No type casting needed. | **Yes** (No action needed) |
| **Target Labels** | `Class` contains strictly values `0` and `1` (284,315 legit, 492 fraud) | **Take no action** | Target column is completely valid and clean. Fraud cases must never be dropped or relabeled. | **Yes** (No action needed) |
| **Negative Values** | `0` negative values in `Time` (min: `0.0s`) and `Amount` (min: `$0.00`) | **Take no action** | Both columns satisfy non-negative physical domain constraints. | **Yes** (No action needed) |
| **Zero-Dollar Transactions** | 1,825 transactions are `$0.00` (1,798 legitimate [0.63%], 27 fraud [5.49%]) | **Retain as valid transactions** | Zero amounts represent real authorization pings / account verification checks. `np.log1p(0) = 0`, so mathematical transformations remain safe. | **Yes** (No action needed) |
| **Statistical Outliers** | `Amount` reaches up to `$25,691.16`; PCA features have long tails (e.g., $V_1$ down to `-56.41`) | **Retain all outliers** | Extreme transactions are real financial events, not errors. In anomaly detection, legitimate variance must be modeled, not discarded. | **Yes** (No action needed) |
| **Duplicate Rows** | 1,081 redundant duplicate rows (1,854 rows in clusters). 32 fraud rows (19 duplicate copies across 13 unique attack profiles). | **Retain all duplicate rows for now (Option 3 Approved)** | Exact duplicate rows may represent repeated transactions or data-recording behavior; we cannot confirm the cause from this dataset alone. Keeping them preserves all 492 rare fraud cases. | **Yes** (All rows retained) |

---

## 3. Deep-Dive: The Duplicate Rows Analysis

### What Did We Measure?
When checking for exact row matches across all 31 columns (`Time`, `V1`–`V28`, `Amount`, `Class`):
* `df.duplicated().sum()` = **1,081 redundant duplicate copies**.
* `df.duplicated(keep=False).sum()` = **1,854 total rows involved in duplicate groups**.
* **Breakdown by Class:**
  * **Legitimate (`Class = 0`):** 1,822 rows (comprising 760 unique transaction profiles and 1,062 redundant repeat copies).
  * **Fraudulent (`Class = 1`):** 32 rows (comprising 13 unique transaction profiles and 19 redundant repeat copies).

### Example of Fraud Duplicates
For example, rows `102441` through `102446` are 6 exact identical copies occurring at the exact same second (`Time = 68207.0`), same amount (`$1.00`), and identical PCA components (`V1` to `V28`).

---

## 4. Final Approved Policy: Option 3 (Retain All Duplicate Rows For Now)

### Rationale and Root Cause Considerations
Exact duplicate rows in credit card transactions can arise from two distinct mechanisms:
1. **Real-world repeated transactions:** Automated carding bots repeatedly charging a card within fractions of a second, or a cardholder rapidly clicking "Submit" on a purchase.
2. **Data-recording / pipeline behavior:** Message broker retries, payment gateway timeouts, or logging pipeline replays recording identical payloads multiple times.

**Critical Conclusion:** We **cannot confirm the exact cause** from this dataset alone without backend merchant API and network telemetry logs. Therefore:
* Automatically dropping duplicates is unjustified and would destroy **19 rare fraud transactions** (3.86% of all positive fraud cases).
* The user has selected **Option 3: Keep all duplicate rows for now**.
* No duplicate rows are deleted from the dataset.

### Impact on Upcoming Chronological 60/20/20 Split
* Because we plan to use a **chronological 60/20/20 split** (ordered by transaction `Time`), we must ensure that any duplicate group spanning identical timestamps does not get sliced across split boundaries (e.g., between Train and Validation, or between Validation and Test).
* In the next task, we will investigate the exact timestamps of all duplicate groups near the 60% and 80% chronological cutoff points to prevent train-test data leakage.

---

## 5. Record of Data Modification Status

* **Raw Dataset Status:** Unchanged and intact (`data/raw/creditcard.csv`).
* **Cleaned File Status:** No cleaned CSV file has been created.
* **Row Count Before Review:** 284,807 rows
* **Row Count After Review:** 284,807 rows
* **Fraud Count Before Review:** 492 cases
* **Fraud Count After Review:** 492 cases
* **Duplicate Copies Count:** 1,081 copies (1,854 rows in clusters)
* **Conclusion:** The dataset is 100% identical to the raw source. Data fidelity is fully preserved.

---

## 6. Viva Talking Points for College Evaluation

If asked about data cleaning in your viva, here is how to explain our decisions in simple, confident English:

1. **"Why didn't you drop missing values or outliers?"**
   > *"Our audit verified that the dataset has zero missing values and zero infinite values. For outliers, transaction amounts reach up to $25,691, but these are legitimate high-value purchases, not data entry errors. In fraud triage and anomaly detection, our goal is to model real customer spending variability, so removing outliers would bias our models."*
2. **"Why did you keep duplicate rows instead of dropping them?"**
   > *"Exact duplicates can represent genuine rapid bot transactions or logging retries; without raw merchant server logs, we cannot know with certainty. More importantly, dropping duplicates would discard 19 of our 492 fraud cases (nearly 4% of our fraud data). We decided to keep all duplicates for now and handle cluster isolation during chronological splitting to prevent data leakage."*
3. **"Why are $0.00 transactions kept?"**
   > *"Zero-dollar transactions account for 5.49% of all fraud cases compared to only 0.63% of legitimate transactions. They represent card verification checks where an attacker tests if a stolen card is active before charging larger amounts. Deleting them would erase a strong fraud signal."*
