"""Scikit-Learn-based baseline recommenders for comparison with the neural model."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

logger = logging.getLogger(__name__)


class PopularityRecommender:
    """Non-personalised baseline: recommends the globally most popular items.

    Popularity is measured by the sum of interaction scores in the training set.
    All users receive the same ranked list, making this a strong non-personalised
    lower bound for evaluation.
    """

    def __init__(self) -> None:
        """Initialise PopularityRecommender."""
        self._popular_items: list[int] = []

    def fit(self, df: pd.DataFrame) -> PopularityRecommender:
        """Compute item popularity from training interactions.

        Args:
            df: Training DataFrame with columns ``item_idx`` and ``score``.

        Returns:
            Self (fitted recommender).
        """
        self._popular_items = (
            df.groupby("item_idx")["score"]
            .sum()
            .sort_values(ascending=False)
            .index.tolist()
        )
        logger.info(
            "PopularityRecommender fitted with %d items", len(self._popular_items)
        )
        return self

    def recommend(self, user_ids: list[int], k: int = 10) -> dict[int, list[int]]:
        """Return top-K popular items for each requested user.

        Args:
            user_ids: Users to generate recommendations for.
            k: Number of items to recommend.

        Returns:
            Mapping of user_id → list of recommended item indices.
        """
        top_k = self._popular_items[:k]
        return {uid: top_k for uid in user_ids}


class SVDRecommender:
    """Collaborative filtering via Truncated SVD on the user-item matrix.

    Uses ``sklearn.decomposition.TruncatedSVD`` to factorise implicit feedback,
    then predicts scores as the dot product of user and item latent factors.
    Seen items are masked to ``-inf`` before ranking.

    Args:
        n_components: Number of latent SVD dimensions.
        seed: Random seed for reproducibility.
    """

    def __init__(self, n_components: int = 50, seed: int = 42) -> None:
        """Initialise SVDRecommender."""
        self._svd = TruncatedSVD(n_components=n_components, random_state=seed)
        self._user_factors: np.ndarray | None = None
        self._item_factors: np.ndarray | None = None

    def fit(self, df: pd.DataFrame, num_users: int, num_items: int) -> SVDRecommender:
        """Fit SVD on the sparse user-item interaction matrix.

        Args:
            df: Training DataFrame with ``user_idx``, ``item_idx``, ``score``.
            num_users: Total number of users (defines matrix row count).
            num_items: Total number of items (defines matrix column count).

        Returns:
            Self (fitted recommender).
        """
        matrix = csr_matrix(
            (df["score"].values, (df["user_idx"].values, df["item_idx"].values)),
            shape=(num_users, num_items),
            dtype=np.float32,
        )
        self._user_factors = self._svd.fit_transform(matrix)
        self._item_factors = self._svd.components_.T
        logger.info(
            "SVDRecommender fitted: %d users × %d items → %d components",
            num_users,
            num_items,
            self._svd.n_components,
        )
        return self

    def recommend(
        self,
        user_ids: list[int],
        train_df: pd.DataFrame,
        k: int = 10,
    ) -> dict[int, list[int]]:
        """Generate top-K recommendations, excluding already-seen items.

        Args:
            user_ids: Users to recommend for.
            train_df: Training interactions used to mask seen items.
            k: Number of items to recommend.

        Returns:
            Mapping of user_id → ordered list of item indices.

        Raises:
            AssertionError: If ``fit()`` has not been called.
        """
        assert self._user_factors is not None, "Call fit() before recommend()"
        seen: dict[int, set[int]] = (
            train_df.groupby("user_idx")["item_idx"].apply(set).to_dict()
        )
        result: dict[int, list[int]] = {}
        for uid in user_ids:
            scores: np.ndarray = self._user_factors[uid] @ self._item_factors.T  # type: ignore[operator]
            for item in seen.get(uid, set()):
                scores[item] = -np.inf
            result[uid] = np.argsort(scores)[::-1][:k].tolist()
        return result
