"""
autoencoder_model.py

Beginner-friendly PyTorch Autoencoder implementation for Credit Card Fraud Triage.
Trains strictly on legitimate transactions (y_train == 0).
Uses early stopping on legitimate validation transactions (y_val == 0).
"""

import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


DEFAULT_AE_CONFIG: Dict[str, Any] = {
    "input_dim": 29,
    "bottleneck_dim": 5,
    "learning_rate": 1e-3,
    "batch_size": 256,
    "epochs": 30,
    "patience": 5,
    "min_delta": 1e-4,
    "random_state": 42,
}


def set_seed(seed: int = 42) -> None:
    """
    Sets random seeds across Python, NumPy, and PyTorch for deterministic reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Autoencoder(nn.Module):
    """
    Fully-connected symmetric autoencoder:
    Encoder: input_dim (29) -> 20 -> 10 -> bottleneck_dim (5)
    Decoder: bottleneck_dim (5) -> 10 -> 20 -> input_dim (29)
    Activations: ReLU on hidden layers, Linear on output.
    """

    def __init__(self, input_dim: int = 29, bottleneck_dim: int = 5):
        super().__init__()
        self.input_dim = input_dim
        self.bottleneck_dim = bottleneck_dim

        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, bottleneck_dim),
        )

        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 10),
            nn.ReLU(),
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encodes and reconstructs the input tensor.
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def create_autoencoder_model(config: Optional[Dict[str, Any]] = None) -> Autoencoder:
    """
    Instantiates an Autoencoder using the provided or default configuration.
    """
    cfg = DEFAULT_AE_CONFIG.copy()
    if config:
        cfg.update(config)

    return Autoencoder(
        input_dim=int(cfg["input_dim"]),
        bottleneck_dim=int(cfg["bottleneck_dim"]),
    )


def fit_autoencoder(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    config: Optional[Dict[str, Any]] = None,
    feature_names: Optional[List[str]] = None,
) -> Tuple[Autoencoder, Dict[str, Any]]:
    """
    Trains the Autoencoder strictly on legitimate training rows (y_train == 0).
    Uses legitimate validation rows (y_val == 0) for early stopping if provided.

    Validates:
    - X_train and y_train have matching row counts.
    - Only legitimate training rows (y_train == 0) are used for parameter updates.
    - No fraudulent transactions enter training or early-stopping loss calculation.
    - Restores the best model weights based on legitimate validation loss.
    """
    if len(X_train) != len(y_train):
        raise ValueError(
            f"X_train length ({len(X_train)}) does not match y_train length ({len(y_train)})"
        )

    unique_train_labels = set(np.unique(y_train))
    if not unique_train_labels.issubset({0, 1}):
        raise ValueError(f"y_train must contain only binary labels 0 and 1, got {unique_train_labels}")

    # Step 1: Merge configuration
    cfg = DEFAULT_AE_CONFIG.copy()
    if config:
        cfg.update(config)

    set_seed(int(cfg["random_state"]))
    device = torch.device("cpu")

    # Step 2: Filter training data strictly to legitimate transactions (y_train == 0)
    train_legit_mask = (y_train == 0)
    num_train_legit = int(train_legit_mask.sum())
    num_train_fraud = int((y_train == 1).sum())

    if num_train_legit == 0:
        raise ValueError("No legitimate rows (y_train == 0) found in training set!")

    X_train_legit = X_train[train_legit_mask]

    # Convert training data to PyTorch DataLoader
    train_tensor = torch.tensor(X_train_legit, dtype=torch.float32)
    train_dataset = TensorDataset(train_tensor)
    batch_size = int(cfg["batch_size"])
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    # Step 3: Filter validation data strictly to legitimate transactions (y_val == 0) if provided
    has_val = X_val is not None and y_val is not None
    val_tensor = None
    num_val_legit = 0
    num_val_fraud = 0

    if has_val:
        if len(X_val) != len(y_val):
            raise ValueError(
                f"X_val length ({len(X_val)}) does not match y_val length ({len(y_val)})"
            )
        val_legit_mask = (y_val == 0)
        num_val_legit = int(val_legit_mask.sum())
        num_val_fraud = int((y_val == 1).sum())
        if num_val_legit > 0:
            val_tensor = torch.tensor(X_val[val_legit_mask], dtype=torch.float32).to(device)

    # Step 4: Model, loss function, and optimizer
    model = create_autoencoder_model(cfg).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["learning_rate"]))

    # Step 5: Training loop with early stopping
    max_epochs = int(cfg["epochs"])
    patience = int(cfg["patience"])
    min_delta = float(cfg["min_delta"])

    best_loss = float("inf")
    best_weights = None
    best_epoch = 0
    patience_counter = 0

    train_losses = []
    val_losses = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_train_loss = 0.0
        total_batches = 0

        for batch in train_loader:
            x_batch = batch[0].to(device)

            optimizer.zero_grad()
            reconstructed = model(x_batch)
            loss = criterion(reconstructed, x_batch)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
            total_batches += 1

        avg_train_loss = epoch_train_loss / total_batches if total_batches > 0 else 0.0
        train_losses.append(avg_train_loss)

        # Validation step on legitimate transactions only
        if val_tensor is not None:
            model.eval()
            with torch.no_grad():
                val_reconstructed = model(val_tensor)
                val_loss = criterion(val_reconstructed, val_tensor).item()
            val_losses.append(val_loss)
            monitoring_loss = val_loss
        else:
            monitoring_loss = avg_train_loss

        # Early stopping logic based on monitoring loss
        if monitoring_loss < (best_loss - min_delta):
            best_loss = monitoring_loss
            best_epoch = epoch
            patience_counter = 0
            # Clone state_dict
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    # Restore best checkpoint
    if best_weights is not None:
        model.load_state_dict(best_weights)

    # Package metadata
    metadata: Dict[str, Any] = {
        "model_name": "autoencoder_baseline",
        "model_type": "Autoencoder",
        "config": cfg,
        "n_training_samples": num_train_legit,
        "total_train_partition_rows": len(X_train),
        "train_fraud_rows_excluded": num_train_fraud,
        "n_val_legit_samples": num_val_legit,
        "val_fraud_rows_excluded": num_val_fraud,
        "input_dim": int(model.input_dim),
        "bottleneck_dim": int(model.bottleneck_dim),
        "feature_names": feature_names if feature_names else [f"feature_{i}" for i in range(X_train.shape[1])],
        "epochs_trained": len(train_losses),
        "best_epoch": int(best_epoch),
        "best_monitoring_loss": float(best_loss),
        "final_train_loss": float(train_losses[-1]) if train_losses else 0.0,
        "final_val_loss": float(val_losses[-1]) if val_losses else None,
        "train_loss_history": train_losses,
        "val_loss_history": val_losses,
        "torch_version": torch.__version__,
    }

    return model, metadata


def save_autoencoder_model(
    model: Autoencoder,
    metadata: Dict[str, Any],
    output_dir: str = "models",
    model_basename: str = "autoencoder_baseline",
) -> Tuple[str, str]:
    """
    Saves the Autoencoder model weights and metadata JSON.

    Returns:
    - Tuple of (model_file_path, metadata_file_path)
    """
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(output_dir, f"{model_basename}.pt")
    torch.save(model.state_dict(), model_path)

    metadata_path = os.path.join(output_dir, f"{model_basename}_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return model_path, metadata_path


def load_autoencoder_model(
    model_path: str = "models/autoencoder_baseline.pt",
    metadata_path: Optional[str] = None,
) -> Tuple[Autoencoder, Optional[Dict[str, Any]]]:
    """
    Loads saved Autoencoder weights and optional metadata.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    metadata = None
    if metadata_path is None:
        default_meta = model_path.replace(".pt", "_metadata.json")
        if os.path.exists(default_meta):
            metadata_path = default_meta

    config = None
    if metadata_path and os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
            config = metadata.get("config", {})

    input_dim = 29
    bottleneck_dim = 5
    if config:
        input_dim = int(config.get("input_dim", 29))
        bottleneck_dim = int(config.get("bottleneck_dim", 5))

    model = Autoencoder(input_dim=input_dim, bottleneck_dim=bottleneck_dim)
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    return model, metadata
