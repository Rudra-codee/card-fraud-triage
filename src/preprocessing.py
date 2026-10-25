"""
preprocessing.py

Reusable, leak-free preprocessing pipeline for Credit Card Fraud Triage.
Transforms the Amount feature using log1p and fits a StandardScaler strictly 
on legitimate training data (Class == 0).
"""

import os
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def get_feature_names() -> List[str]:
    """
    Returns the exact list of 29 model feature names:
    V1 through V28 plus Amount.
    Excludes Time, Class, and row_id.
    """
    return [f"V{i}" for i in range(1, 29)] + ["Amount"]


def load_and_validate_data(
    raw_data_path: str = "data/raw/creditcard.csv",
    manifest_path: str = "data/splits/split_manifest.csv",
) -> pd.DataFrame:
    """
    Loads the raw credit card dataset and the split manifest.
    Validates manifest integrity and merges split labels based on original row_id.
    """
    if not os.path.exists(raw_data_path):
        raise FileNotFoundError(f"Raw dataset not found at: {raw_data_path}")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Split manifest not found at: {manifest_path}")

    # 1. Load raw data in original order and assign row_id
    df_raw = pd.read_csv(raw_data_path)
    total_rows = len(df_raw)
    df_raw["row_id"] = np.arange(total_rows, dtype=int)

    # 2. Load manifest
    manifest = pd.read_csv(manifest_path)

    # 3. Validate manifest schema and contents
    expected_cols = ["row_id", "split"]
    if list(manifest.columns) != expected_cols:
        raise ValueError(f"Manifest columns must be {expected_cols}, got {list(manifest.columns)}")

    if len(manifest) != total_rows:
        raise ValueError(f"Manifest row count ({len(manifest)}) does not match raw data ({total_rows})")

    if manifest["row_id"].nunique() != total_rows:
        raise ValueError("Manifest contains duplicate row_id values!")

    if set(manifest["row_id"]) != set(range(total_rows)):
        raise ValueError("Manifest row_id values do not exactly match 0 to len(df)-1!")

    if manifest["split"].isnull().any():
        raise ValueError("Manifest contains missing/null split assignments!")

    valid_splits = {"train", "validation", "test"}
    unique_splits = set(manifest["split"].unique())
    if not unique_splits.issubset(valid_splits):
        raise ValueError(f"Manifest contains invalid split labels: {unique_splits - valid_splits}")

    # 4. Map split label onto the raw data using row_id
    # We use map on row_id to avoid any reordering
    split_map = dict(zip(manifest["row_id"], manifest["split"]))
    df_raw["split"] = df_raw["row_id"].map(split_map)

    return df_raw


def reconstruct_partitions(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits the validated dataframe into train, validation, and test subsets.
    Verifies that partitions are non-empty and mutually disjoint.
    """
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "validation"].copy()
    test_df = df[df["split"] == "test"].copy()

    # Verify all rows are accounted for
    total_reconstructed = len(train_df) + len(val_df) + len(test_df)
    if total_reconstructed != len(df):
        raise ValueError(f"Partition rows ({total_reconstructed}) do not equal total rows ({len(df)})")

    # Verify pairwise disjoint row IDs
    train_ids = set(train_df["row_id"])
    val_ids = set(val_df["row_id"])
    test_ids = set(test_df["row_id"])

    if len(train_ids.intersection(val_ids)) > 0:
        raise ValueError("Train and Validation partitions share overlapping row IDs!")
    if len(train_ids.intersection(test_ids)) > 0:
        raise ValueError("Train and Test partitions share overlapping row IDs!")
    if len(val_ids.intersection(test_ids)) > 0:
        raise ValueError("Validation and Test partitions share overlapping row IDs!")

    return train_df, val_df, test_df


def transform_amount(df_subset: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """
    Extracts the 29 model features and applies np.log1p safely to Amount.
    Verifies that Amount is numeric, finite, and strictly greater than -1.
    """
    if "Amount" not in df_subset.columns:
        raise KeyError("Column 'Amount' not found in provided dataframe.")

    amount_series = df_subset["Amount"]

    # Safety checks before log1p
    if not pd.api.types.is_numeric_dtype(amount_series):
        raise TypeError("Column 'Amount' must be numeric.")

    if not np.isfinite(amount_series).all():
        raise ValueError("Column 'Amount' contains NaN or infinite values.")

    if (amount_series <= -1.0).any():
        raise ValueError("Cannot apply log1p: Amount contains values <= -1.0.")

    # Create feature dataframe
    X = df_subset[feature_cols].copy()
    X["Amount"] = np.log1p(X["Amount"].astype(float))

    return X


def fit_scaler_on_legitimate(X_train: pd.DataFrame, y_train: pd.Series) -> StandardScaler:
    """
    Fits a StandardScaler strictly on the legitimate rows (Class == 0) of the training set.
    Prevents data leakage from fraud patterns, validation data, and test data.
    """
    legit_mask = (y_train == 0)
    num_legit = legit_mask.sum()

    if num_legit == 0:
        raise ValueError("No legitimate rows (Class == 0) found in training set to fit scaler!")

    X_train_legit = X_train[legit_mask]

    scaler = StandardScaler()
    scaler.fit(X_train_legit)

    # Verify scaler properties
    expected_features = len(X_train.columns)
    if scaler.n_features_in_ != expected_features:
        raise ValueError(f"Scaler expected {expected_features} features, got {scaler.n_features_in_}")

    if scaler.n_samples_seen_ != num_legit:
        raise ValueError(f"Scaler expected {num_legit} samples, fitted on {scaler.n_samples_seen_}")

    return scaler


def transform_features(scaler: StandardScaler, X: pd.DataFrame) -> np.ndarray:
    """
    Transforms a feature DataFrame using an already fitted scaler.
    Returns a 2D numpy array of scaled features.
    """
    X_scaled = scaler.transform(X)

    if not np.isfinite(X_scaled).all():
        raise ValueError("Transformed features contain NaN or infinite values.")

    return X_scaled


def validate_preprocessing_outputs(data: Dict) -> None:
    """
    Runs automated validation assertions on the complete preprocessed pipeline output.
    """
    # 1. Feature count is exactly 29
    assert len(data["feature_names"]) == 29, f"Expected 29 features, got {len(data['feature_names'])}"
    assert "Time" not in data["feature_names"], "Time must not be in model features!"
    assert "Class" not in data["feature_names"], "Class must not be in model features!"
    assert "row_id" not in data["feature_names"], "row_id must not be in model features!"

    # 2. Scaler properties
    scaler = data["scaler"]
    assert scaler.n_features_in_ == 29, f"Scaler n_features_in_ is {scaler.n_features_in_}, expected 29"
    assert scaler.n_samples_seen_ == 170528, (
        f"Scaler n_samples_seen_ is {scaler.n_samples_seen_}, expected 170,528 legitimate training rows"
    )

    # 3. Partition shapes
    assert data["X_train"].shape == (170888, 29), f"X_train shape mismatch: {data['X_train'].shape}"
    assert data["X_val"].shape == (56957, 29), f"X_val shape mismatch: {data['X_val'].shape}"
    assert data["X_test"].shape == (56962, 29), f"X_test shape mismatch: {data['X_test'].shape}"

    # 4. Target alignment and fraud preservation
    assert len(data["y_train"]) == 170888, "y_train length mismatch"
    assert len(data["y_val"]) == 56957, "y_val length mismatch"
    assert len(data["y_test"]) == 56962, "y_test length mismatch"

    fraud_train = (data["y_train"] == 1).sum()
    fraud_val = (data["y_val"] == 1).sum()
    fraud_test = (data["y_test"] == 1).sum()

    assert fraud_train == 360, f"Expected 360 train fraud cases, got {fraud_train}"
    assert fraud_val == 57, f"Expected 57 val fraud cases, got {fraud_val}"
    assert fraud_test == 75, f"Expected 75 test fraud cases, got {fraud_test}"
    assert fraud_train + fraud_val + fraud_test == 492, "Total fraud count does not equal 492!"

    # 5. Row ID alignment and disjointness
    train_ids = set(data["row_ids_train"])
    val_ids = set(data["row_ids_val"])
    test_ids = set(data["row_ids_test"])

    assert len(train_ids) == 170888, "Duplicate row IDs in train"
    assert len(val_ids) == 56957, "Duplicate row IDs in val"
    assert len(test_ids) == 56962, "Duplicate row IDs in test"
    assert len(train_ids.intersection(val_ids)) == 0, "Train and Val row IDs overlap!"
    assert len(train_ids.intersection(test_ids)) == 0, "Train and Test row IDs overlap!"
    assert len(val_ids.intersection(test_ids)) == 0, "Val and Test row IDs overlap!"

    # 6. Finite values in transformed arrays
    for split_name in ["X_train", "X_val", "X_test"]:
        arr = data[split_name]
        assert np.isfinite(arr).all(), f"{split_name} contains non-finite values!"


def run_preprocessing_pipeline(
    raw_data_path: str = "data/raw/creditcard.csv",
    manifest_path: str = "data/splits/split_manifest.csv",
) -> Dict:
    """
    Executes the complete preprocessing pipeline:
    1. Loads and validates raw data and split manifest.
    2. Reconstructs partitions.
    3. Transforms Amount with log1p.
    4. Fits StandardScaler strictly on legitimate training rows (Class == 0).
    5. Transforms train, validation, and test features.
    6. Validates all outputs.
    7. Returns clean preprocessed dictionary.
    """
    feature_cols = get_feature_names()

    # Step 1: Load and validate
    df = load_and_validate_data(raw_data_path=raw_data_path, manifest_path=manifest_path)

    # Step 2: Reconstruct partitions
    train_df, val_df, test_df = reconstruct_partitions(df)

    # Separate targets and row IDs
    y_train = train_df["Class"].reset_index(drop=True)
    y_val = val_df["Class"].reset_index(drop=True)
    y_test = test_df["Class"].reset_index(drop=True)

    row_ids_train = train_df["row_id"].to_numpy()
    row_ids_val = val_df["row_id"].to_numpy()
    row_ids_test = test_df["row_id"].to_numpy()

    # Step 3: Transform Amount using log1p
    X_train_unscaled = transform_amount(train_df, feature_cols)
    X_val_unscaled = transform_amount(val_df, feature_cols)
    X_test_unscaled = transform_amount(test_df, feature_cols)

    # Step 4: Fit scaler on legitimate training rows ONLY
    scaler = fit_scaler_on_legitimate(X_train_unscaled, y_train)

    # Step 5: Transform all splits
    X_train_scaled = transform_features(scaler, X_train_unscaled)
    X_val_scaled = transform_features(scaler, X_val_unscaled)
    X_test_scaled = transform_features(scaler, X_test_unscaled)

    # Package output
    pipeline_data = {
        "X_train": X_train_scaled,
        "y_train": y_train,
        "row_ids_train": row_ids_train,
        "X_val": X_val_scaled,
        "y_val": y_val,
        "row_ids_val": row_ids_val,
        "X_test": X_test_scaled,
        "y_test": y_test,
        "row_ids_test": row_ids_test,
        "scaler": scaler,
        "feature_names": feature_cols,
    }

    # Step 6: Validate outputs
    validate_preprocessing_outputs(pipeline_data)

    return pipeline_data


if __name__ == "__main__":
    print("Executing preprocessing pipeline...")
    data = run_preprocessing_pipeline()

    print("\n=== PREPROCESSING COMPLETED SUCCESSFULLY ===")
    print(f"Features:                 {len(data['feature_names'])} {data['feature_names'][:3]} ... {data['feature_names'][-1]}")
    print(f"Scaler samples seen:      {int(data['scaler'].n_samples_seen_):,} legitimate training rows (Class == 0)")
    print(f"X_train shape:            {data['X_train'].shape} | Fraud: {(data['y_train'] == 1).sum():,}")
    print(f"X_val shape:              {data['X_val'].shape}  | Fraud: {(data['y_val'] == 1).sum():,}")
    print(f"X_test shape:             {data['X_test'].shape}  | Fraud: {(data['y_test'] == 1).sum():,}")
    print("All validation checks passed with zero errors.")
