"""Unit tests for preprocessors and evaluation metrics."""

import pandas as pd
import pytest

from src.data.preprocessor import EventWeightPreprocessor, MinInteractionsFilter
from src.evaluation.metrics import (
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)

# ─── Preprocessor Tests ───────────────────────────────────────────────────────


@pytest.fixture()
def sample_events() -> pd.DataFrame:
    """Sample events DataFrame for testing."""
    return pd.DataFrame(
        {
            "visitorid": [1, 1, 1, 2, 2, 3],
            "itemid": [10, 10, 20, 10, 30, 40],
            "event": ["view", "addtocart", "view", "transaction", "view", "view"],
        }
    )


def test_event_weight_preprocessor_aggregates(sample_events: pd.DataFrame) -> None:
    """EventWeightPreprocessor should sum weights per (user, item) pair."""
    preprocessor = EventWeightPreprocessor()
    result = preprocessor.transform(sample_events)

    # user=1, item=10 should have score 1 (view) + 3 (addtocart) = 4
    row = result[(result["visitorid"] == 1) & (result["itemid"] == 10)]
    assert row["score"].iloc[0] == 4


def test_min_interactions_filter_removes_sparse(sample_events: pd.DataFrame) -> None:
    """MinInteractionsFilter should remove low-activity users/items."""
    weighted = EventWeightPreprocessor().transform(sample_events)
    filtered = MinInteractionsFilter(min_user=2, min_item=2).transform(weighted)

    # user=3 has only 1 interaction — should be removed
    assert 3 not in filtered["visitorid"].values


# ─── Metrics Tests ────────────────────────────────────────────────────────────


def test_precision_at_k_perfect() -> None:
    """Precision@K should be 1.0 when all recommendations are relevant."""
    assert precision_at_k([1, 2, 3], {1, 2, 3}, k=3) == 1.0


def test_recall_at_k_partial() -> None:
    """Recall@K should be 0.5 when half of relevant items are found."""
    assert recall_at_k([1, 2], {1, 2, 3, 4}, k=2) == 0.5


def test_hit_rate_at_k_hit() -> None:
    """HitRate@K should be 1.0 when at least one relevant item is in top-K."""
    assert hit_rate_at_k([5, 1], {1}, k=2) == 1.0


def test_hit_rate_at_k_miss() -> None:
    """HitRate@K should be 0.0 when no relevant item is in top-K."""
    assert hit_rate_at_k([5, 6], {1}, k=2) == 0.0


def test_ndcg_at_k_perfect() -> None:
    """NDCG@K should be 1.0 for a perfect ranking."""
    assert ndcg_at_k([1, 2, 3], {1, 2, 3}, k=3) == pytest.approx(1.0)
