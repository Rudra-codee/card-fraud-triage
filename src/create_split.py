"""
create_split.py

Creates the chronological 60/20/20 train/validation/test split manifest 
for the Credit Card Fraud Triage project (Strategy B: Timestamp Boundary Alignment).
"""

import os
import pandas as pd
import numpy as np

# 1. Load the raw dataset without modifying it
RAW_DATA_PATH = "data/raw/creditcard.csv"
MANIFEST_PATH = "data/splits/split_manifest.csv"

print(f"Loading raw data from: {RAW_DATA_PATH}")
df = pd.read_csv(RAW_DATA_PATH)
n_rows = len(df)
print(f"Total rows loaded: {n_rows}")

# 2. Add stable row_id based on original raw CSV row order (0 to 284,806)
df['row_id'] = np.arange(n_rows, dtype=int)

# 3. Stable sort by Time ascending
# 'stable' keeps original tie-order when timestamps are identical
df_sorted = df.sort_values(by="Time", kind="stable").reset_index(drop=True)

# 4. Apply approved Strategy B boundaries
# Shifting 60% boundary to index 170888 keeps all 6 transactions with Time=120396.0 in train
IDX_TRAIN_END = 170888  # First 170,888 rows (indices 0 to 170,887)
IDX_VAL_END = 227845    # Next 56,957 rows (indices 170,888 to 227,844)

df_sorted['split'] = 'train'
df_sorted.loc[IDX_TRAIN_END:IDX_VAL_END - 1, 'split'] = 'validation'
df_sorted.loc[IDX_VAL_END:, 'split'] = 'test'

# 5. Build manifest matching original row_id order
# Manifest contains exactly two columns: row_id and split
manifest = df_sorted[['row_id', 'split']].sort_values(by='row_id').reset_index(drop=True)

# 6. Comprehensive Validation Assertions
print("\nRunning validation checks...")

# Check 1: Total manifest rows
assert len(manifest) == 284807, f"Expected 284,807 rows, got {len(manifest)}"

# Check 2: Row IDs unique and complete
assert manifest['row_id'].nunique() == 284807, "Duplicate row_ids found in manifest!"
assert set(manifest['row_id']) == set(range(284807)), "row_id set does not match 0..284806!"

# Check 3: No null split values
assert manifest['split'].isnull().sum() == 0, "Missing split assignments found!"
assert set(manifest['split'].unique()) == {'train', 'validation', 'test'}, "Unexpected split labels!"

# Check 4: Pairwise disjoint row sets
train_ids = set(manifest[manifest['split'] == 'train']['row_id'])
val_ids = set(manifest[manifest['split'] == 'validation']['row_id'])
test_ids = set(manifest[manifest['split'] == 'test']['row_id'])

assert len(train_ids.intersection(val_ids)) == 0, "Train and Validation overlap!"
assert len(train_ids.intersection(test_ids)) == 0, "Train and Test overlap!"
assert len(val_ids.intersection(test_ids)) == 0, "Validation and Test overlap!"

# Check 5: Preserve all 492 fraud cases
train_df = df.loc[list(train_ids)]
val_df = df.loc[list(val_ids)]
test_df = df.loc[list(test_ids)]

fraud_train = (train_df['Class'] == 1).sum()
fraud_val = (val_df['Class'] == 1).sum()
fraud_test = (test_df['Class'] == 1).sum()
total_fraud = fraud_train + fraud_val + fraud_test

assert total_fraud == 492, f"Expected 492 fraud cases, found {total_fraud}!"

# Check 6: Chronological order consistency
assert train_df['Time'].max() <= val_df['Time'].min(), (
    f"Chronological violation: Train max Time ({train_df['Time'].max()}) > Val min Time ({val_df['Time'].min()})"
)
assert val_df['Time'].max() <= test_df['Time'].min(), (
    f"Chronological violation: Val max Time ({val_df['Time'].max()}) > Test min Time ({test_df['Time'].min()})"
)

# Check 7: All 6 transactions at Time == 120396.0 are in train
time_120396_rows = df[df['Time'] == 120396.0]
assert len(time_120396_rows) == 6, f"Expected 6 rows at Time 120396.0, got {len(time_120396_rows)}"
time_120396_splits = manifest.loc[time_120396_rows.index, 'split'].unique()
assert list(time_120396_splits) == ['train'], (
    f"Expected all rows at Time 120396.0 to be in train, found: {time_120396_splits}"
)

# Check 8: No exact duplicate group is divided across partitions
feature_cols = [c for c in df.columns if c != 'row_id']
dup_mask = df.duplicated(subset=feature_cols, keep=False)
dup_df = df[dup_mask].copy()
dup_df['group_id'] = dup_df.groupby(feature_cols, sort=False).ngroup()
dup_df['split'] = manifest.loc[dup_df.index, 'split']

groups_crossing = (dup_df.groupby('group_id')['split'].nunique() > 1).sum()
assert groups_crossing == 0, f"Found {groups_crossing} exact duplicate groups crossing split boundaries!"

print("All validation checks passed successfully!")

# 7. Save the split manifest
os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)
manifest.to_csv(MANIFEST_PATH, index=False)
print(f"\nManifest successfully saved to: {MANIFEST_PATH}")

# 8. Report partition statistics
print("\n=== FINAL PARTITION SUMMARY ===")
partitions = [
    ("Train", train_df, fraud_train),
    ("Validation", val_df, fraud_val),
    ("Test", test_df, fraud_test)
]

for name, part, f_count in partitions:
    p_size = len(part)
    p_pct = (p_size / n_rows) * 100
    t_min = part['Time'].min()
    t_max = part['Time'].max()
    f_pct = (f_count / p_size) * 100
    print(f"\n{name} Partition:")
    print(f"  Rows:        {p_size:,} ({p_pct:.2f}%)")
    print(f"  Time range:  {t_min:.1f}s to {t_max:.1f}s (duration: {(t_max - t_min)/3600:.2f} hours)")
    print(f"  Fraud count: {f_count} ({f_pct:.4f}%)")

print("\nSplit successfully completed and verified.")
