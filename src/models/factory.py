"""Model factory and neural architectures for recommendation."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

import torch
import torch.nn as nn


class ModelType(StrEnum):
    """Supported recommendation model architecture types."""

    MLP = "mlp"
    EMBEDDING_MLP = "embedding_mlp"


class MLPRecommender(nn.Module):
    """Feedforward MLP for pre-computed feature vectors.

    Args:
        input_dim: Dimension of the concatenated user+item feature vector.
        hidden_dims: Sizes of each hidden layer.
        dropout: Dropout probability applied after each hidden activation.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        dropout: float = 0.3,
    ) -> None:
        """Initialize MLPRecommender."""
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_dim
        for out_dim in hidden_dims:
            layers += [nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict interaction scores from pre-computed feature vectors.

        Args:
            x: Float tensor of shape (batch_size, input_dim).

        Returns:
            Score tensor of shape (batch_size, 1).
        """
        return self.net(x)


class EmbeddingMLPRecommender(nn.Module):
    """Embedding-based MLP for collaborative filtering.

    Learns separate embedding tables for users and items.
    Concatenates embeddings and feeds them through a MLP to predict scores.

    Args:
        num_users: Number of distinct users.
        num_items: Number of distinct items.
        embedding_dim: Dimension of each user and item embedding.
        hidden_dims: Sizes of MLP hidden layers after embedding concatenation.
        dropout: Dropout probability between hidden layers.
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.3,
    ) -> None:
        """Initialize EmbeddingMLPRecommender."""
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        layers: list[nn.Module] = []
        in_dim = embedding_dim * 2
        for out_dim in hidden_dims:
            layers += [nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """Predict interaction scores from user and item integer indices.

        Args:
            user_ids: Long tensor of user indices, shape (batch_size,).
            item_ids: Long tensor of item indices, shape (batch_size,).

        Returns:
            Score tensor of shape (batch_size, 1).
        """
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        return self.mlp(torch.cat([user_emb, item_emb], dim=-1))


class ModelFactory:
    """Factory for creating recommendation model instances by type."""

    @staticmethod
    def create(model_type: ModelType, **kwargs: Any) -> nn.Module:
        """Instantiate a model by architecture type.

        Args:
            model_type: Which architecture to build.
            **kwargs: Constructor keyword arguments for the chosen model.

        Returns:
            Initialised nn.Module ready for training.

        Raises:
            ValueError: If model_type is not a recognised ModelType.
        """
        if model_type == ModelType.MLP:
            return MLPRecommender(**kwargs)
        if model_type == ModelType.EMBEDDING_MLP:
            return EmbeddingMLPRecommender(**kwargs)
        raise ValueError(f"Unknown model type: {model_type!r}")
