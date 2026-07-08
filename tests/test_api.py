"""Tests for the inference service and the recommendation API."""

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
import pytest
import torch
from fastapi.testclient import TestClient
from sklearn.preprocessing import LabelEncoder

from src.api.app import app
from src.inference.recommender import (
    ArtifactsMissingError,
    RecommenderService,
    UnknownUserError,
)
from src.models.factory import ModelFactory, ModelType

USER_IDS = [11, 22, 33]
ITEM_IDS = [101, 202, 303]
TRAIN_PARAMS = {"embedding_dim": 4, "hidden_dims": [8], "dropout": 0.0}


@pytest.fixture()
def artifacts_dir(tmp_path: Path) -> Path:
    """Write a minimal set of valid artifacts to a temp directory.

    Args:
        tmp_path: pytest-provided temp directory.

    Returns:
        Root path containing models/ and processed/ subdirectories.
    """
    models_dir = tmp_path / "models"
    processed_dir = tmp_path / "processed"
    models_dir.mkdir()
    processed_dir.mkdir()

    model = ModelFactory.create(
        ModelType.EMBEDDING_MLP,
        num_users=len(USER_IDS),
        num_items=len(ITEM_IDS),
        **TRAIN_PARAMS,
    )
    checkpoint = {
        "state_dict": model.state_dict(),
        "num_users": len(USER_IDS),
        "num_items": len(ITEM_IDS),
        **TRAIN_PARAMS,
    }
    torch.save(checkpoint, models_dir / "best_model.pt")

    with open(processed_dir / "encoders.pkl", "wb") as fh:
        pickle.dump(
            {
                "user_encoder": LabelEncoder().fit(USER_IDS),
                "item_encoder": LabelEncoder().fit(ITEM_IDS),
            },
            fh,
        )
    pd.DataFrame(
        {"visitorid": [11, 11, 22], "itemid": [101, 202, 101], "score": [1.0] * 3}
    ).to_parquet(processed_dir / "interactions.parquet", index=False)
    return tmp_path


@pytest.fixture()
def service(artifacts_dir: Path) -> RecommenderService:
    """Build a RecommenderService from the temp artifacts.

    Args:
        artifacts_dir: Root path with models/ and processed/ subdirectories.

    Returns:
        Loaded RecommenderService.
    """
    return RecommenderService.from_artifacts(
        models_dir=artifacts_dir / "models",
        processed_dir=artifacts_dir / "processed",
    )


@pytest.fixture()
def client(service: RecommenderService) -> TestClient:
    """TestClient with the tiny service injected into app state.

    Args:
        service: Loaded RecommenderService.

    Returns:
        TestClient ready for requests.
    """
    app.state.recommender = service
    return TestClient(app)


def test_missing_artifacts_raise(tmp_path: Path) -> None:
    """Building the service fails clearly when artifacts are absent."""
    with pytest.raises(ArtifactsMissingError, match="make repro"):
        RecommenderService.from_artifacts(tmp_path, tmp_path)


def test_recommend_masks_seen_items(service: RecommenderService) -> None:
    """User 11 interacted with items 101 and 202, so only 303 remains."""
    recs = service.recommend(user_id=11, k=3)
    assert [r.item_id for r in recs] == [303]


def test_recommend_unknown_user_raises(service: RecommenderService) -> None:
    """Users absent from training raise UnknownUserError."""
    with pytest.raises(UnknownUserError):
        service.recommend(user_id=999, k=3)


def test_recommend_respects_k(service: RecommenderService) -> None:
    """User 33 has no seen items, so all items are eligible."""
    recs = service.recommend(user_id=33, k=2)
    assert len(recs) == 2
    assert recs[0].score >= recs[1].score


def test_root_endpoint(client: TestClient) -> None:
    """Root describes the service instead of returning 404."""
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["service"] == "RetailRocket RecSys API"
    assert body["docs"] == "/docs"


def test_health_endpoint(client: TestClient) -> None:
    """Health reports status and model dimensions."""
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["num_users"] == len(USER_IDS)
    assert body["num_items"] == len(ITEM_IDS)


def test_recommendations_endpoint(client: TestClient) -> None:
    """Known user gets a sorted top-K list."""
    response = client.get("/recommendations/33", params={"k": 2})
    assert response.status_code == 200
    body = response.json()
    assert body["user_id"] == 33
    assert len(body["items"]) == 2


def test_recommendations_unknown_user_returns_404(client: TestClient) -> None:
    """Cold-start users get a 404 with an explanatory message."""
    response = client.get("/recommendations/999")
    assert response.status_code == 404
    assert "cold start" in response.json()["detail"]


def test_recommendations_invalid_k_returns_422(client: TestClient) -> None:
    """Values of k outside 1-100 are rejected by validation."""
    assert client.get("/recommendations/33", params={"k": 0}).status_code == 422
    assert client.get("/recommendations/33", params={"k": 101}).status_code == 422
