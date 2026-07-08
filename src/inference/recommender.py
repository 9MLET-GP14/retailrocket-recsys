"""Load trained artifacts and serve top-K recommendations."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder

from src.models.factory import EmbeddingMLPRecommender, ModelFactory, ModelType


class ArtifactsMissingError(FileNotFoundError):
    """Raised when a required model artifact is not found on disk."""


class UnknownUserError(KeyError):
    """Raised when a user id was never seen during training (cold start)."""


@dataclass
class Recommendation:
    """Single recommended item with its predicted score.

    Attributes:
        item_id: Original RetailRocket item id.
        score: Predicted interaction score (higher is better).
    """

    item_id: int
    score: float


def _load_model(model_path: Path) -> EmbeddingMLPRecommender:
    """Rebuild the EmbeddingMLP from its self-describing checkpoint.

    The checkpoint carries the architecture (dimensions, hidden layers,
    dropout) alongside the trained weights, so serving never depends on
    the current params.yaml matching the artifact.

    Args:
        model_path: Path to the saved checkpoint.

    Returns:
        Loaded model in eval mode on CPU.
    """
    ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
    model: EmbeddingMLPRecommender = ModelFactory.create(  # type: ignore[assignment]
        ModelType.EMBEDDING_MLP,
        num_users=ckpt["num_users"],
        num_items=ckpt["num_items"],
        embedding_dim=ckpt["embedding_dim"],
        hidden_dims=ckpt["hidden_dims"],
        dropout=ckpt["dropout"],
    )
    model.load_state_dict(ckpt["state_dict"])
    return model.eval()


def _build_seen_items(
    interactions: pd.DataFrame,
    user_encoder: LabelEncoder,
    item_encoder: LabelEncoder,
) -> dict[int, set[int]]:
    """Map each encoded user to the set of items they already interacted with.

    Args:
        interactions: DataFrame with visitorid and itemid columns.
        user_encoder: Fitted LabelEncoder for users.
        item_encoder: Fitted LabelEncoder for items.

    Returns:
        Dict of user_idx to set of seen item_idx.
    """
    df = interactions[["visitorid", "itemid"]].copy()
    df["user_idx"] = user_encoder.transform(df["visitorid"])
    df["item_idx"] = item_encoder.transform(df["itemid"])
    return df.groupby("user_idx")["item_idx"].apply(set).to_dict()


class RecommenderService:
    """Serves top-K item recommendations from trained artifacts.

    Args:
        model: Trained EmbeddingMLPRecommender in eval mode.
        user_encoder: Fitted LabelEncoder mapping visitorid to user_idx.
        item_encoder: Fitted LabelEncoder mapping itemid to item_idx.
        seen_items: Dict of user_idx to item_idx already interacted with.
    """

    def __init__(
        self,
        model: EmbeddingMLPRecommender,
        user_encoder: LabelEncoder,
        item_encoder: LabelEncoder,
        seen_items: dict[int, set[int]],
    ) -> None:
        """Initialize the service with loaded artifacts."""
        self._model = model
        self._user_encoder = user_encoder
        self._item_encoder = item_encoder
        self._seen_items = seen_items
        self.num_users = len(user_encoder.classes_)
        self.num_items = len(item_encoder.classes_)

    @classmethod
    def from_artifacts(
        cls,
        models_dir: Path,
        processed_dir: Path,
    ) -> RecommenderService:
        """Build the service from the artifacts produced by the DVC pipeline.

        Args:
            models_dir: Directory containing best_model.pt.
            processed_dir: Directory with encoders.pkl and
                interactions.parquet.

        Returns:
            Ready-to-serve RecommenderService.

        Raises:
            ArtifactsMissingError: If any required artifact is absent.
        """
        paths = {
            "model": models_dir / "best_model.pt",
            "encoders": processed_dir / "encoders.pkl",
            "interactions": processed_dir / "interactions.parquet",
        }
        missing = [str(p) for p in paths.values() if not p.exists()]
        if missing:
            raise ArtifactsMissingError(
                f"Missing artifacts: {missing}. Run the pipeline (make repro) first."
            )
        with open(paths["encoders"], "rb") as fh:
            encoders = pickle.load(fh)  # noqa: S301
        model = _load_model(paths["model"])
        seen = _build_seen_items(
            pd.read_parquet(paths["interactions"]),
            encoders["user_encoder"],
            encoders["item_encoder"],
        )
        return cls(model, encoders["user_encoder"], encoders["item_encoder"], seen)

    def recommend(self, user_id: int, k: int) -> list[Recommendation]:
        """Return the top-K unseen items for a known user.

        Args:
            user_id: Original RetailRocket visitorid.
            k: Number of recommendations to return.

        Returns:
            Recommendations sorted by descending score.

        Raises:
            UnknownUserError: If the user was not seen during training.
        """
        try:
            user_idx = int(self._user_encoder.transform([user_id])[0])
        except ValueError as exc:
            raise UnknownUserError(f"Unknown user_id: {user_id}") from exc

        users = torch.full((self.num_items,), user_idx, dtype=torch.long)
        items = torch.arange(self.num_items, dtype=torch.long)
        with torch.no_grad():
            scores = self._model(users, items).squeeze().numpy()
        for item_idx in self._seen_items.get(user_idx, set()):
            scores[item_idx] = -np.inf

        ranked = np.argsort(scores)[::-1]
        top_idx = [idx for idx in ranked if np.isfinite(scores[idx])][:k]
        item_ids = self._item_encoder.inverse_transform(top_idx)
        return [
            Recommendation(item_id=int(item_id), score=float(scores[idx]))
            for idx, item_id in zip(top_idx, item_ids, strict=True)
        ]
