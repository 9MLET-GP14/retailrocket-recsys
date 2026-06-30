#!/usr/bin/env python
"""Stage 2: encode user/item IDs and split interactions into train / val / test."""

from __future__ import annotations

import json
import logging
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.features.engineering import encode_ids, split_interactions

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_params() -> dict[str, float | int]:
    """Load feature_eng section from params.yaml.

    Returns:
        Dict with val_size, test_size, and seed keys.
    """
    with open("params.yaml") as fh:
        return yaml.safe_load(fh)["feature_eng"]  # type: ignore[return-value]


def save_splits(splits: object, out: Path) -> None:
    """Persist train / val / test DataFrames as parquet files.

    Args:
        splits: DataSplits dataclass.
        out: Output directory path.
    """
    splits.train.to_parquet(out / "train.parquet", index=False)  # type: ignore[attr-defined]
    splits.val.to_parquet(out / "val.parquet", index=False)  # type: ignore[attr-defined]
    splits.test.to_parquet(out / "test.parquet", index=False)  # type: ignore[attr-defined]


def run(val_size: float, test_size: float, seed: int) -> None:
    """Execute feature engineering pipeline.

    Args:
        val_size: Fraction of data for validation.
        test_size: Fraction of data for test.
        seed: Random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)

    interactions = pd.read_parquet("data/processed/interactions.parquet")
    encoded = encode_ids(interactions)
    splits = split_interactions(encoded.df, val_size=val_size, test_size=test_size)

    out = Path("data/processed")
    out.mkdir(parents=True, exist_ok=True)

    save_splits(splits, out)

    with open(out / "encoders.pkl", "wb") as fh:
        pickle.dump(
            {
                "user_encoder": encoded.user_encoder,
                "item_encoder": encoded.item_encoder,
            },
            fh,
        )

    metadata = {"num_users": encoded.num_users, "num_items": encoded.num_items}
    with open(out / "metadata.json", "w") as fh:
        json.dump(metadata, fh)

    logger.info(
        "Saved splits — train: %d | val: %d | test: %d | users: %d | items: %d",
        len(splits.train),
        len(splits.val),
        len(splits.test),
        encoded.num_users,
        encoded.num_items,
    )


def main() -> None:
    """Entry point for DVC stage 2."""
    params = load_params()
    run(
        val_size=float(params["val_size"]),
        test_size=float(params["test_size"]),
        seed=int(params["seed"]),
    )


if __name__ == "__main__":
    main()
