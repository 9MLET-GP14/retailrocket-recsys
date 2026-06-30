"""Models module: factory, neural architectures, baselines, and dataset."""

from src.models.baseline import PopularityRecommender, SVDRecommender
from src.models.dataset import InteractionDataset
from src.models.factory import (
    EmbeddingMLPRecommender,
    MLPRecommender,
    ModelFactory,
    ModelType,
)

__all__ = [
    "ModelFactory",
    "ModelType",
    "MLPRecommender",
    "EmbeddingMLPRecommender",
    "InteractionDataset",
    "PopularityRecommender",
    "SVDRecommender",
]
