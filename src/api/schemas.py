"""Pydantic schemas for the recommendation API responses."""

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Service health and loaded-model summary."""

    status: str
    model_loaded: bool
    num_users: int
    num_items: int


class RecommendationItem(BaseModel):
    """Single recommended item."""

    item_id: int
    score: float


class RecommendationResponse(BaseModel):
    """Top-K recommendations for a user."""

    user_id: int
    k: int
    items: list[RecommendationItem]
