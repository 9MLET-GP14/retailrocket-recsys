"""Ranking evaluation metrics: Precision@K, Recall@K, NDCG@K, Hit Rate."""

import numpy as np
import pandas as pd


def precision_at_k(
    recommended: list[int], relevant: set[int], k: int
) -> float:
    """Fraction of top-K recommendations that are relevant.

    Args:
        recommended: Ordered list of recommended item IDs.
        relevant: Set of ground-truth relevant item IDs.
        k: Cutoff position.

    Returns:
        Precision@K score in [0, 1].
    """
    top_k = recommended[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / k


def recall_at_k(
    recommended: list[int], relevant: set[int], k: int
) -> float:
    """Fraction of relevant items found in top-K recommendations.

    Args:
        recommended: Ordered list of recommended item IDs.
        relevant: Set of ground-truth relevant item IDs.
        k: Cutoff position.

    Returns:
        Recall@K score in [0, 1].
    """
    if not relevant:
        return 0.0
    top_k = recommended[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / len(relevant)


def ndcg_at_k(
    recommended: list[int], relevant: set[int], k: int
) -> float:
    """Normalized Discounted Cumulative Gain at K.

    Args:
        recommended: Ordered list of recommended item IDs.
        relevant: Set of ground-truth relevant item IDs.
        k: Cutoff position.

    Returns:
        NDCG@K score in [0, 1].
    """
    top_k = recommended[:k]
    dcg = sum(
        1.0 / np.log2(i + 2)
        for i, item in enumerate(top_k)
        if item in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def hit_rate_at_k(
    recommended: list[int], relevant: set[int], k: int
) -> float:
    """Whether at least one relevant item appears in top-K.

    Args:
        recommended: Ordered list of recommended item IDs.
        relevant: Set of ground-truth relevant item IDs.
        k: Cutoff position.

    Returns:
        1.0 if any relevant item is in top-K, else 0.0.
    """
    top_k = set(recommended[:k])
    return float(bool(top_k & relevant))


def compute_all_metrics(
    user_recommendations: dict[int, list[int]],
    user_ground_truth: dict[int, set[int]],
    k: int = 10,
) -> pd.DataFrame:
    """Compute all ranking metrics across all users.

    Args:
        user_recommendations: Mapping of user_id → ordered recommended item list.
        user_ground_truth: Mapping of user_id → set of relevant item IDs.
        k: Cutoff position.

    Returns:
        DataFrame with mean Precision@K, Recall@K, NDCG@K, HitRate@K.
    """
    results: list[dict[str, float]] = []
    for user_id, recommended in user_recommendations.items():
        relevant = user_ground_truth.get(user_id, set())
        results.append({
            f"precision@{k}": precision_at_k(recommended, relevant, k),
            f"recall@{k}": recall_at_k(recommended, relevant, k),
            f"ndcg@{k}": ndcg_at_k(recommended, relevant, k),
            f"hit_rate@{k}": hit_rate_at_k(recommended, relevant, k),
        })
    return pd.DataFrame(results).mean().to_frame(name="mean_score")
