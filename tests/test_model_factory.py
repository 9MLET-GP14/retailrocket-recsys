"""Unit tests for the ModelFactory."""

import pytest
import torch

from src.models.factory import EmbeddingMLPRecommender, MLPRecommender, ModelFactory, ModelType


def test_factory_creates_mlp() -> None:
    """ModelFactory should return an MLPRecommender for ModelType.MLP."""
    model = ModelFactory.create(ModelType.MLP, input_dim=128, hidden_dims=[64, 32])
    assert isinstance(model, MLPRecommender)


def test_factory_creates_embedding_mlp() -> None:
    """ModelFactory should return an EmbeddingMLPRecommender for ModelType.EMBEDDING_MLP."""
    model = ModelFactory.create(
        ModelType.EMBEDDING_MLP,
        num_users=1000,
        num_items=5000,
        embedding_dim=32,
        hidden_dims=[64, 32],
    )
    assert isinstance(model, EmbeddingMLPRecommender)


def test_factory_raises_on_unknown_type() -> None:
    """ModelFactory should raise ValueError for unsupported model types."""
    with pytest.raises(ValueError, match="Unknown model type"):
        ModelFactory.create("unknown_model")  # type: ignore[arg-type]


def test_embedding_mlp_forward_shape() -> None:
    """EmbeddingMLPRecommender forward pass should return (batch, 1) tensor."""
    model = ModelFactory.create(
        ModelType.EMBEDDING_MLP,
        num_users=100,
        num_items=500,
        embedding_dim=16,
        hidden_dims=[32, 16],
    )
    user_ids = torch.randint(0, 100, (8,))
    item_ids = torch.randint(0, 500, (8,))
    output = model(user_ids, item_ids)
    assert output.shape == (8, 1)
