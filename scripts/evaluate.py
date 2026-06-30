#!/usr/bin/env python
"""Stage 4: evaluate neural model and baselines, log metrics, save comparison."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
import yaml

from src.evaluation.metrics import compute_all_metrics
from src.models.baseline import PopularityRecommender, SVDRecommender
from src.models.factory import EmbeddingMLPRecommender, ModelFactory, ModelType

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_params() -> dict:
    """Load full params.yaml.

    Returns:
        Dict with all stage sections.
    """
    with open("params.yaml") as fh:
        return yaml.safe_load(fh)  # type: ignore[return-value]


def load_test_data() -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Load metadata, train, and test splits for evaluation.

    Returns:
        Tuple of (metadata dict, train DataFrame, test DataFrame).
    """
    with open("data/processed/metadata.json") as fh:
        meta = json.load(fh)
    train_df = pd.read_parquet("data/processed/train.parquet")
    test_df = pd.read_parquet("data/processed/test.parquet")
    return meta, train_df, test_df


def load_model(
    meta: dict, train_params: dict, device: torch.device
) -> EmbeddingMLPRecommender:
    """Reconstruct and load the saved EmbeddingMLPRecommender.

    Args:
        meta: Metadata dict with num_users and num_items.
        train_params: Training params for model architecture.
        device: Torch device for inference.

    Returns:
        Loaded model in eval mode.
    """
    model: EmbeddingMLPRecommender = ModelFactory.create(  # type: ignore[assignment]
        ModelType.EMBEDDING_MLP,
        num_users=meta["num_users"],
        num_items=meta["num_items"],
        embedding_dim=train_params["embedding_dim"],
        hidden_dims=train_params["hidden_dims"],
        dropout=train_params["dropout"],
    )
    state = torch.load("models/best_model.pt", map_location=device)
    model.load_state_dict(state)
    return model.to(device).eval()


def score_neural(
    model: EmbeddingMLPRecommender,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    num_items: int,
    k: int,
    device: torch.device,
    max_users: int,
) -> pd.DataFrame:
    """Generate top-K recommendations from the neural model and compute metrics.

    Args:
        model: Trained EmbeddingMLPRecommender in eval mode.
        train_df: Training interactions (for masking seen items).
        test_df: Test interactions as ground truth.
        num_items: Total item count.
        k: Recommendation cutoff.
        device: Torch device.
        max_users: Maximum number of test users to evaluate.

    Returns:
        DataFrame with mean ranking metrics.
    """
    seen: dict[int, set[int]] = (
        train_df.groupby("user_idx")["item_idx"].apply(set).to_dict()
    )
    ground_truth: dict[int, set[int]] = (
        test_df.groupby("user_idx")["item_idx"].apply(set).to_dict()
    )
    users = list(ground_truth.keys())[:max_users]
    all_items = torch.arange(num_items, dtype=torch.long, device=device)
    recs: dict[int, list[int]] = {}

    with torch.no_grad():
        for uid in users:
            uid_tensor = torch.full((num_items,), uid, dtype=torch.long, device=device)
            scores = model(uid_tensor, all_items).squeeze().cpu().numpy()
            for item in seen.get(uid, set()):
                if item < num_items:
                    scores[item] = -np.inf
            recs[uid] = np.argsort(scores)[::-1][:k].tolist()

    return compute_all_metrics(recs, ground_truth, k=k)


def score_baseline(
    recommender: PopularityRecommender | SVDRecommender,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    k: int,
    max_users: int,
) -> pd.DataFrame:
    """Generate recommendations from a baseline and compute metrics.

    Args:
        recommender: Fitted baseline recommender.
        train_df: Training interactions (for SVD masking).
        test_df: Test ground truth.
        k: Recommendation cutoff.
        max_users: Maximum number of test users to evaluate.

    Returns:
        DataFrame with mean ranking metrics.
    """
    ground_truth: dict[int, set[int]] = (
        test_df.groupby("user_idx")["item_idx"].apply(set).to_dict()
    )
    users = list(ground_truth.keys())[:max_users]
    if isinstance(recommender, SVDRecommender):
        recs = recommender.recommend(users, train_df=train_df, k=k)
    else:
        recs = recommender.recommend(users, k=k)
    return compute_all_metrics(recs, ground_truth, k=k)


def evaluate_all(
    model: EmbeddingMLPRecommender,
    meta: dict,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    eval_params: dict,
    seed: int,
    device: torch.device,
) -> pd.DataFrame:
    """Score the neural model and both baselines, returning a combined comparison.

    Args:
        model: Loaded neural model in eval mode.
        meta: Metadata dict with num_users and num_items.
        train_df: Training interactions.
        test_df: Test interactions as ground truth.
        eval_params: Evaluation section of params.yaml.
        seed: Random seed for baseline reproducibility.
        device: Torch device for neural inference.

    Returns:
        DataFrame with one row per metric and one column per model.
    """
    k = eval_params["top_k"]
    max_users = eval_params["eval_users"]

    logger.info("Evaluating EmbeddingMLP on up to %d users (k=%d)...", max_users, k)
    neural_m = score_neural(
        model, train_df, test_df, meta["num_items"], k, device, max_users
    )

    logger.info("Fitting and evaluating PopularityRecommender...")
    pop = PopularityRecommender().fit(train_df)
    pop_m = score_baseline(pop, train_df, test_df, k, max_users)

    logger.info("Fitting and evaluating SVDRecommender...")
    n_comp = min(eval_params["n_svd_components"], meta["num_items"] - 1)
    svd = SVDRecommender(n_components=n_comp, seed=seed).fit(
        train_df, meta["num_users"], meta["num_items"]
    )
    svd_m = score_baseline(svd, train_df, test_df, k, max_users)

    comparison = pd.concat(
        [neural_m, pop_m, svd_m], axis=1, keys=["EmbeddingMLP", "Popularity", "SVD"]
    )
    comparison.columns = [f"{m}_{c}" for m, c in comparison.columns]
    return comparison


def log_to_mlflow(comparison: pd.DataFrame) -> None:
    """Log every comparison metric to a new MLflow evaluation run.

    Args:
        comparison: DataFrame with model × metric scores.
    """
    from src.config import settings

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)
    with mlflow.start_run(run_name="evaluation"):
        for col in comparison.columns:
            mlflow.log_metric(col, float(comparison[col].iloc[0]))


def save_metrics(comparison: pd.DataFrame) -> None:
    """Persist comparison as CSV and DVC-compatible JSON.

    Args:
        comparison: DataFrame with model × metric scores.
    """
    out = Path("data/processed")
    out.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(out / "metrics_comparison.csv")

    metrics_flat = {col: float(comparison[col].iloc[0]) for col in comparison.columns}
    with open(out / "metrics.json", "w") as fh:
        json.dump(metrics_flat, fh, indent=2)
    logger.info("Metrics saved to %s", out)


def main() -> None:
    """Entry point for DVC stage 4."""
    all_params = load_params()
    eval_params = all_params["evaluate"]
    seed = eval_params["seed"]
    random.seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    meta, train_df, test_df = load_test_data()
    model = load_model(meta, all_params["train"], device)

    comparison = evaluate_all(model, meta, train_df, test_df, eval_params, seed, device)
    logger.info("Results:\n%s", comparison.to_string())

    log_to_mlflow(comparison)
    save_metrics(comparison)


if __name__ == "__main__":
    main()
