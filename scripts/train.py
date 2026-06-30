#!/usr/bin/env python
"""Stage 3: train EmbeddingMLPRecommender and save best model weights."""

from __future__ import annotations

import json
import logging
import os
import pathlib
import random

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_params() -> dict:
    """Load train section from params.yaml.

    Returns:
        Dict with training hyperparameters.
    """
    with open("params.yaml") as fh:
        return yaml.safe_load(fh)["train"]  # type: ignore[return-value]


def apply_env_overrides(params: dict) -> None:
    """Set environment variables so Pydantic Settings picks up params.yaml values.

    Args:
        params: Training hyperparameters from params.yaml.
    """
    os.environ.update(
        {
            "EMBEDDING_DIM": str(params["embedding_dim"]),
            "HIDDEN_DIMS": ",".join(str(d) for d in params["hidden_dims"]),
            "DROPOUT": str(params["dropout"]),
            "LEARNING_RATE": str(params["learning_rate"]),
            "BATCH_SIZE": str(params["batch_size"]),
            "MAX_EPOCHS": str(params["max_epochs"]),
            "EARLY_STOPPING_PATIENCE": str(params["early_stopping_patience"]),
            "SEED": str(params["seed"]),
        }
    )


def set_seeds(seed: int) -> None:
    """Fix random seeds for reproducibility.

    Args:
        seed: Seed value applied to Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_data() -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Load metadata and train/val splits produced by the feature_eng stage.

    Returns:
        Tuple of (metadata dict, train DataFrame, val DataFrame).
    """
    with open("data/processed/metadata.json") as fh:
        meta = json.load(fh)
    train_df = pd.read_parquet("data/processed/train.parquet")
    val_df = pd.read_parquet("data/processed/val.parquet")
    return meta, train_df, val_df


def build_loaders(
    train_df: pd.DataFrame, val_df: pd.DataFrame, batch_size: int
) -> tuple[DataLoader, DataLoader]:
    """Wrap train and validation DataFrames in PyTorch DataLoaders.

    Args:
        train_df: Training interactions.
        val_df: Validation interactions.
        batch_size: Batch size for both loaders.

    Returns:
        Tuple of (train_loader, val_loader).
    """
    from src.models.dataset import InteractionDataset

    train_loader = DataLoader(
        InteractionDataset(train_df), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(InteractionDataset(val_df), batch_size=batch_size)
    return train_loader, val_loader


def build_model(meta: dict, params: dict) -> torch.nn.Module:
    """Instantiate the EmbeddingMLPRecommender via ModelFactory.

    Args:
        meta: Metadata dict with num_users and num_items.
        params: Training hyperparameters from params.yaml.

    Returns:
        Initialized model ready for training.
    """
    from src.models.factory import ModelFactory, ModelType

    return ModelFactory.create(
        ModelType.EMBEDDING_MLP,
        num_users=meta["num_users"],
        num_items=meta["num_items"],
        embedding_dim=params["embedding_dim"],
        hidden_dims=params["hidden_dims"],
        dropout=params["dropout"],
    )


def main() -> None:
    """Entry point for DVC stage 3."""
    params = load_params()
    apply_env_overrides(params)
    set_seeds(params["seed"])

    meta, train_df, val_df = load_data()
    train_loader, val_loader = build_loaders(train_df, val_df, params["batch_size"])
    model = build_model(meta, params)

    logger.info(
        "Training EmbeddingMLP: %d users × %d items | emb_dim=%d",
        meta["num_users"],
        meta["num_items"],
        params["embedding_dim"],
    )

    from src.training.trainer import run_training

    model = run_training(model, train_loader, val_loader, run_name="embedding_mlp")

    pathlib.Path("models").mkdir(exist_ok=True)
    torch.save(model.state_dict(), "models/best_model.pt")
    logger.info("Best model saved to models/best_model.pt")


if __name__ == "__main__":
    main()
