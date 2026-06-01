#!/usr/bin/env python
"""End-to-end training pipeline: load → preprocess → train → evaluate → compare."""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import TYPE_CHECKING

import mlflow
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.config import settings
from src.data.loader import RetailRocketLoader
from src.data.preprocessor import EventWeightPreprocessor, MinInteractionsFilter
from src.evaluation.metrics import compute_all_metrics
from src.features.engineering import DataSplits, encode_ids, split_interactions
from src.models.baseline import PopularityRecommender, SVDRecommender
from src.models.dataset import InteractionDataset
from src.models.factory import EmbeddingMLPRecommender, ModelFactory, ModelType
from src.training.trainer import run_training

if TYPE_CHECKING:
    pass

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

_EVAL_USERS = 500  # cap on test users for ranking evaluation (speed)


def set_seeds(seed: int) -> None:
    """Fix all random seeds for reproducibility.

    Args:
        seed: Seed value applied to Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_loaders(splits: DataSplits) -> tuple[DataLoader, DataLoader]:
    """Wrap train and validation DataFrames in PyTorch DataLoaders.

    Args:
        splits: DataSplits produced by ``split_interactions``.

    Returns:
        Tuple of (train_loader, val_loader).
    """
    train_ds = InteractionDataset(splits.train)
    val_ds = InteractionDataset(splits.val)
    return (
        DataLoader(train_ds, batch_size=settings.batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=settings.batch_size),
    )


def evaluate_neural(
    model: EmbeddingMLPRecommender,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    num_items: int,
    k: int,
    device: torch.device,
) -> pd.DataFrame:
    """Score all items per test user and compute ranking metrics.

    For each user, all items are scored and training-seen items are masked
    before selecting the top-K recommendations.

    Args:
        model: Trained ``EmbeddingMLPRecommender``.
        train_df: Training interactions used to exclude seen items.
        test_df: Test interactions used as ground truth.
        num_items: Total item count (embedding table size).
        k: Recommendation cutoff.
        device: Torch device for inference.

    Returns:
        DataFrame with mean Precision@K, Recall@K, NDCG@K, HitRate@K.
    """
    model.eval()
    seen: dict[int, set[int]] = (
        train_df.groupby("user_idx")["item_idx"].apply(set).to_dict()
    )
    ground_truth: dict[int, set[int]] = (
        test_df.groupby("user_idx")["item_idx"].apply(set).to_dict()
    )
    sample_users = list(ground_truth.keys())[:_EVAL_USERS]
    all_items = torch.arange(num_items, dtype=torch.long, device=device)
    recommendations: dict[int, list[int]] = {}

    with torch.no_grad():
        for uid in sample_users:
            user_tensor = torch.full(
                (num_items,), uid, dtype=torch.long, device=device
            )
            scores = model(user_tensor, all_items).squeeze().cpu().numpy()
            for item in seen.get(uid, set()):
                if item < num_items:
                    scores[item] = -np.inf
            recommendations[uid] = np.argsort(scores)[::-1][:k].tolist()

    return compute_all_metrics(recommendations, ground_truth, k=k)


def evaluate_baseline(
    recommender: PopularityRecommender | SVDRecommender,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    k: int,
) -> pd.DataFrame:
    """Generate and evaluate recommendations from a baseline model.

    Args:
        recommender: Fitted baseline (Popularity or SVD).
        train_df: Training interactions (passed to SVD for seen-item masking).
        test_df: Test interactions used as ground truth.
        k: Recommendation cutoff.

    Returns:
        DataFrame with mean Precision@K, Recall@K, NDCG@K, HitRate@K.
    """
    ground_truth: dict[int, set[int]] = (
        test_df.groupby("user_idx")["item_idx"].apply(set).to_dict()
    )
    user_ids = list(ground_truth.keys())[:_EVAL_USERS]

    if isinstance(recommender, SVDRecommender):
        recs = recommender.recommend(user_ids, train_df=train_df, k=k)
    else:
        recs = recommender.recommend(user_ids, k=k)

    return compute_all_metrics(recs, ground_truth, k=k)


def log_comparison(comparison: pd.DataFrame, k: int) -> None:
    """Log per-model metric summary to the active MLflow run.

    Args:
        comparison: DataFrame with model names as columns and metric names as rows.
        k: Cutoff used for metric naming.
    """
    for model_name in comparison.columns:
        for metric, value in comparison[model_name].items():
            mlflow.log_metric(f"{model_name}_{metric}", float(value))


def main() -> None:
    """Run full training, evaluation, and comparison pipeline."""
    set_seeds(settings.seed)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    logger.info("Loading RetailRocket events from: %s", settings.data_raw_path)
    loader = RetailRocketLoader()
    events = loader.load_events()

    # ── 2. Preprocess ─────────────────────────────────────────────────────────
    logger.info("Applying event weighting and interaction filtering...")
    events = EventWeightPreprocessor().transform(events)
    events = MinInteractionsFilter(min_user=5, min_item=5).transform(events)

    # ── 3. Feature engineering ────────────────────────────────────────────────
    logger.info("Encoding IDs and splitting into train / val / test...")
    encoded = encode_ids(events)
    splits = split_interactions(encoded.df, val_size=0.1, test_size=0.1)
    logger.info(
        "Split sizes — train: %d | val: %d | test: %d",
        len(splits.train),
        len(splits.val),
        len(splits.test),
    )

    # ── 4. Neural model training ──────────────────────────────────────────────
    train_loader, val_loader = build_loaders(splits)
    model: EmbeddingMLPRecommender = ModelFactory.create(  # type: ignore[assignment]
        ModelType.EMBEDDING_MLP,
        num_users=encoded.num_users,
        num_items=encoded.num_items,
        embedding_dim=settings.embedding_dim,
        hidden_dims=settings.hidden_dims_list,
        dropout=settings.dropout,
    )
    logger.info(
        "Training EmbeddingMLP: %d users × %d items | emb_dim=%d",
        encoded.num_users,
        encoded.num_items,
        settings.embedding_dim,
    )
    model = run_training(model, train_loader, val_loader, run_name="embedding_mlp")

    # ── 5. Ranking evaluation ─────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    k = settings.top_k

    logger.info("Evaluating EmbeddingMLP on %d test users (k=%d)...", _EVAL_USERS, k)
    neural_metrics = evaluate_neural(
        model, splits.train, splits.test, encoded.num_items, k, device
    )

    # ── 6. Scikit-Learn baselines ─────────────────────────────────────────────
    logger.info("Fitting PopularityRecommender baseline...")
    popularity = PopularityRecommender().fit(splits.train)
    popularity_metrics = evaluate_baseline(popularity, splits.train, splits.test, k)

    logger.info("Fitting SVDRecommender baseline...")
    n_components = min(50, encoded.num_items - 1)
    svd = SVDRecommender(n_components=n_components, seed=settings.seed)
    svd.fit(splits.train, encoded.num_users, encoded.num_items)
    svd_metrics = evaluate_baseline(svd, splits.train, splits.test, k)

    # ── 7. Comparison summary ─────────────────────────────────────────────────
    comparison = pd.concat(
        [neural_metrics, popularity_metrics, svd_metrics],
        axis=1,
        keys=["EmbeddingMLP", "Popularity", "SVD"],
    )
    comparison.columns = [f"{m}_{c}" for m, c in comparison.columns]
    logger.info("Metrics comparison (mean @%d):\n%s", k, comparison.to_string())

    out_dir = Path(settings.data_processed_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "metrics_comparison.csv"
    comparison.to_csv(csv_path)
    logger.info("Comparison saved to %s", csv_path)


if __name__ == "__main__":
    main()
