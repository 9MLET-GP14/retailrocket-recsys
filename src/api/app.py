"""FastAPI application serving top-K product recommendations."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request

from src.api.schemas import (
    HealthResponse,
    RecommendationItem,
    RecommendationResponse,
)
from src.config import settings
from src.inference.recommender import RecommenderService, UnknownUserError


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load model artifacts once at startup.

    Args:
        app: FastAPI application receiving the loaded service.

    Yields:
        None after the recommender is ready.
    """
    app.state.recommender = RecommenderService.from_artifacts(
        models_dir=Path(settings.models_path),
        processed_dir=Path(settings.data_processed_path),
    )
    yield


app = FastAPI(
    title="RetailRocket RecSys API",
    description="Product recommendations from user navigation behavior.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/")
def root() -> dict[str, str]:
    """Describe the service and point to the useful routes.

    Returns:
        Service name, version and paths for docs and health.
    """
    return {
        "service": app.title,
        "version": app.version,
        "docs": "/docs",
        "health": "/health",
        "recommendations": "/recommendations/{user_id}?k=10",
    }


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    """Report service status and loaded-model dimensions.

    Args:
        request: Incoming request carrying the app state.

    Returns:
        HealthResponse with model summary.
    """
    recommender: RecommenderService = request.app.state.recommender
    return HealthResponse(
        status="ok",
        model_loaded=True,
        num_users=recommender.num_users,
        num_items=recommender.num_items,
    )


@app.get("/recommendations/{user_id}", response_model=RecommendationResponse)
def recommendations(
    request: Request,
    user_id: int,
    k: int = Query(default=settings.top_k, ge=1, le=100),
) -> RecommendationResponse:
    """Return the top-K recommended items for a known user.

    Args:
        request: Incoming request carrying the app state.
        user_id: Original RetailRocket visitorid.
        k: Number of recommendations (1-100).

    Returns:
        RecommendationResponse sorted by descending score.

    Raises:
        HTTPException: 404 if the user was not seen during training.
    """
    recommender: RecommenderService = request.app.state.recommender
    try:
        recs = recommender.recommend(user_id=user_id, k=k)
    except UnknownUserError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"user_id {user_id} not found in training data (cold start)",
        ) from exc
    return RecommendationResponse(
        user_id=user_id,
        k=k,
        items=[RecommendationItem(item_id=r.item_id, score=r.score) for r in recs],
    )
