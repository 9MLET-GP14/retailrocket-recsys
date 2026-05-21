"""Preprocessors for RetailRocket events data — Strategy pattern."""

import logging
from abc import ABC, abstractmethod

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class PreprocessorStrategy(ABC):
    """Abstract base for preprocessing strategies."""

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply preprocessing transformation.

        Args:
            df: Input DataFrame.

        Returns:
            Transformed DataFrame.
        """


class EventWeightPreprocessor(PreprocessorStrategy):
    """Assigns numeric weights to event types and aggregates by user-item.

    Weights: view=1, addtocart=3, transaction=5.
    This creates an implicit feedback score per (user, item) pair.
    """

    EVENT_WEIGHTS: dict[str, int] = {
        "view": 1,
        "addtocart": 3,
        "transaction": 5,
    }

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map events to weights and sum per (visitorid, itemid).

        Args:
            df: Events DataFrame with columns: visitorid, itemid, event.

        Returns:
            Aggregated DataFrame with columns: visitorid, itemid, score.
        """
        logger.info(
            "EventWeightPreprocessor: applying weights %s to %d events",
            self.EVENT_WEIGHTS,
            len(df),
        )
        df = df.copy()
        df["score"] = df["event"].map(self.EVENT_WEIGHTS).fillna(0)
        result = df.groupby(["visitorid", "itemid"], as_index=False)["score"].sum()
        logger.info(
            "EventWeightPreprocessor: aggregated to %d (user, item) pairs | "
            "score range: [%.1f, %.1f]",
            len(result),
            result["score"].min(),
            result["score"].max(),
        )
        return result


class MinInteractionsFilter(PreprocessorStrategy):
    """Removes users and items with fewer than min_interactions interactions.

    Helps reduce sparsity and cold-start noise.
    """

    def __init__(self, min_user: int = 5, min_item: int = 5) -> None:
        """Initialize filter thresholds.

        Args:
            min_user: Minimum interactions per user.
            min_item: Minimum interactions per item.
        """
        self._min_user = min_user
        self._min_item = min_item
        logger.info(
            "MinInteractionsFilter initialized: min_user=%d, min_item=%d",
            self._min_user,
            self._min_item,
        )

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter out low-activity users and items.

        Args:
            df: Aggregated DataFrame with columns: visitorid, itemid, score.

        Returns:
            Filtered DataFrame.
        """
        n_before = len(df)
        users_before = df["visitorid"].nunique()
        items_before = df["itemid"].nunique()

        df = self._filter_by_count(df, "visitorid", self._min_user)
        df = self._filter_by_count(df, "itemid", self._min_item)
        df = df.reset_index(drop=True)

        logger.info(
            "MinInteractionsFilter: %d → %d rows | " "users: %d → %d | items: %d → %d",
            n_before,
            len(df),
            users_before,
            df["visitorid"].nunique(),
            items_before,
            df["itemid"].nunique(),
        )
        return df

    @staticmethod
    def _filter_by_count(df: pd.DataFrame, col: str, min_count: int) -> pd.DataFrame:
        """Keep only rows where col has at least min_count occurrences.

        Args:
            df: Input DataFrame.
            col: Column to count.
            min_count: Minimum count threshold.

        Returns:
            Filtered DataFrame.
        """
        counts = df[col].value_counts()
        valid = counts[counts >= min_count].index
        return df[df[col].isin(valid)]
