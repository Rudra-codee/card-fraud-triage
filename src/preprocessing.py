"""
preprocessing.py

Reusable, leak-free preprocessing pipeline for Credit Card Fraud Triage.
Transforms the Amount feature using log1p and fits a StandardScaler strictly 
on legitimate training data (Class == 0).
Saves processed feature arrays, target labels, row IDs, scaler, and metadata 
under data/processed/ for team integration.
"""

import json
import os
from typing import Dict, List, Tuple
import joblib
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
    y_train = train_df["Class"].to_numpy(dtype=int)
    y_val = val_df["Class"].to_numpy(dtype=int)
    y_test = test_df["Class"].to_numpy(dtype=int)

    row_ids_train = train_df["row_id"].to_numpy(dtype=int)
    row_ids_val = val_df["row_id"].to_numpy(dtype=int)
    row_ids_test = test_df["row_id"].to_numpy(dtype=int)

    # Step 3: Transform Amount using log1p
    X_train_unscaled = transform_amount(train_df, feature_cols)
    X_val_unscaled = transform_amount(val_df, feature_cols)
    X_test_unscaled = transform_amount(test_df, feature_cols)

    # Step 4: Fit scaler on legitimate training rows ONLY
    scaler = fit_scaler_on_legitimate(X_train_unscaled, pd.Series(y_train))

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


def save_processed_data(data: Dict, output_dir: str = "data/processed") -> List[str]:
    """
    Saves preprocessed arrays, target labels, row IDs, scaler, and metadata
    under output_dir (default: data/processed) for team integration.
    Returns list of saved file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_files = []

    # 1. Save NumPy feature arrays
    np_arrays = {
        "X_train.npy": data["X_train"],
        "X_val.npy": data["X_val"],
        "X_test.npy": data["X_test"],
        "y_train.npy": data["y_train"],
        "y_val.npy": data["y_val"],
        "y_test.npy": data["y_test"],
        "row_ids_train.npy": data["row_ids_train"],
        "row_ids_val.npy": data["row_ids_val"],
        "row_ids_test.npy": data["row_ids_test"],
    }

    for filename, array in np_arrays.items():
        filepath = os.path.join(output_dir, filename)
        np.save(filepath, array)
        saved_files.append(filepath)

    # 2. Save feature names JSON
    feature_names_path = os.path.join(output_dir, "feature_names.json")
    with open(feature_names_path, "w", encoding="utf-8") as f:
        json.dump(data["feature_names"], f, indent=2)
    saved_files.append(feature_names_path)

    # 3. Save fitted scaler using joblib
    scaler_path = os.path.join(output_dir, "scaler.joblib")
    joblib.dump(data["scaler"], scaler_path)
    saved_files.append(scaler_path)

    # 4. Save preprocessing metadata JSON
    metadata = {
        "source_dataset_path": "data/raw/creditcard.csv",
        "split_manifest_path": "data/splits/split_manifest.csv",
        "feature_count": len(data["feature_names"]),
        "feature_names": data["feature_names"],
        "amount_transformation": "np.log1p(Amount)",
        "scaler_type": "StandardScaler",
        "scaler_fit_criteria": "Training partition rows where Class == 0 only (legitimate transactions)",
        "scaler_n_samples_seen": int(data["scaler"].n_samples_seen_),
        "partitions": {
            "train": {
                "total_rows": int(len(data["X_train"])),
                "fraud_count": int((data["y_train"] == 1).sum()),
                "legitimate_count": int((data["y_train"] == 0).sum()),
                "fraud_percentage": float((data["y_train"] == 1).mean() * 100),
            },
            "validation": {
                "total_rows": int(len(data["X_val"])),
                "fraud_count": int((data["y_val"] == 1).sum()),
                "legitimate_count": int((data["y_val"] == 0).sum()),
                "fraud_percentage": float((data["y_val"] == 1).mean() * 100),
            },
            "test": {
                "total_rows": int(len(data["X_test"])),
                "fraud_count": int((data["y_test"] == 1).sum()),
                "legitimate_count": int((data["y_test"] == 0).sum()),
                "fraud_percentage": float((data["y_test"] == 1).mean() * 100),
            },
        },
    }

    metadata_path = os.path.join(output_dir, "preprocessing_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    saved_files.append(metadata_path)

    return saved_files


def load_processed_data(processed_dir: str = "data/processed") -> Dict:
    """
    Convenience function for team members to load all preprocessed artifacts.
    """
    if not os.path.exists(processed_dir):
        raise FileNotFoundError(f"Processed directory not found: {processed_dir}")

    data = {
        "X_train": np.load(os.path.join(processed_dir, "X_train.npy")),
        "X_val": np.load(os.path.join(processed_dir, "X_val.npy")),
        "X_test": np.load(os.path.join(processed_dir, "X_test.npy")),
        "y_train": np.load(os.path.join(processed_dir, "y_train.npy")),
        "y_val": np.load(os.path.join(processed_dir, "y_val.npy")),
        "y_test": np.load(os.path.join(processed_dir, "y_test.npy")),
        "row_ids_train": np.load(os.path.join(processed_dir, "row_ids_train.npy")),
        "row_ids_val": np.load(os.path.join(processed_dir, "row_ids_val.npy")),
        "row_ids_test": np.load(os.path.join(processed_dir, "row_ids_test.npy")),
    }

    scaler_path = os.path.join(processed_dir, "scaler.joblib")
    if os.path.exists(scaler_path):
        data["scaler"] = joblib.load(scaler_path)

    feature_names_path = os.path.join(processed_dir, "feature_names.json")
    if os.path.exists(feature_names_path):
        with open(feature_names_path, "r", encoding="utf-8") as f:
            data["feature_names"] = json.load(f)

    metadata_path = os.path.join(processed_dir, "preprocessing_metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            data["metadata"] = json.load(f)

    return data


if __name__ == "__main__":
    print("Executing preprocessing pipeline...")
    pipeline_data = run_preprocessing_pipeline()

    print("\nSaving processed datasets to data/processed/ ...")
    saved_files_list = save_processed_data(pipeline_data)

    print("\n=== PREPROCESSING COMPLETED AND SAVED SUCCESSFULLY ===")
    print(f"Features:                 {len(pipeline_data['feature_names'])} {pipeline_data['feature_names'][:3]} ... {pipeline_data['feature_names'][-1]}")
    print(f"Scaler samples seen:      {int(pipeline_data['scaler'].n_samples_seen_):,} legitimate training rows (Class == 0)")
    print(f"X_train shape:            {pipeline_data['X_train'].shape} | Fraud: {(pipeline_data['y_train'] == 1).sum():,}")
    print(f"X_val shape:              {pipeline_data['X_val'].shape}  | Fraud: {(pipeline_data['y_val'] == 1).sum():,}")
    print(f"X_test shape:             {pipeline_data['X_test'].shape}  | Fraud: {(pipeline_data['y_test'] == 1).sum():,}")
    print(f"Total files saved:        {len(saved_files_list)}")
    for fpath in saved_files_list:
        print(f"  - {fpath}")
