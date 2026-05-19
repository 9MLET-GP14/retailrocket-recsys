"""RetailRocket dataset loader."""

from pathlib import Path

import pandas as pd

from src.config import settings


class RetailRocketLoader:
    """Loads and validates RetailRocket raw CSV files.

    RetailRocket has 3 files:
    - events.csv: user interaction events (view, addtocart, transaction)
    - item_properties_part1.csv: item metadata part 1
    - item_properties_part2.csv: item metadata part 2
    """

    EVENT_TYPES: list[str] = ["view", "addtocart", "transaction"]

    def __init__(self, raw_path: str | None = None) -> None:
        """Initialize loader with raw data path.

        Args:
            raw_path: Path to raw data directory. Defaults to settings value.
        """
        self._raw_path = Path(raw_path or settings.data_raw_path)

    def load_events(self) -> pd.DataFrame:
        """Load and validate events.csv.

        Returns:
            DataFrame with columns: timestamp, visitorid, event, itemid, transactionid.

        Raises:
            FileNotFoundError: If events.csv is not found.
        """
        path = self._raw_path / "events.csv"
        self._assert_exists(path)
        df = pd.read_csv(path)
        df = self._validate_events(df)
        return df

    def load_item_properties(self) -> pd.DataFrame:
        """Load and concatenate both item properties files.

        Returns:
            Combined DataFrame with item metadata.
        """
        parts = [
            self._raw_path / "item_properties_part1.csv",
            self._raw_path / "item_properties_part2.csv",
        ]
        for p in parts:
            self._assert_exists(p)
        return pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)

    def _validate_events(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter to known event types and parse timestamp.

        Args:
            df: Raw events DataFrame.

        Returns:
            Cleaned events DataFrame.
        """
        df = df[df["event"].isin(self.EVENT_TYPES)].copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df.reset_index(drop=True)

    @staticmethod
    def _assert_exists(path: Path) -> None:
        """Raise FileNotFoundError if path does not exist.

        Args:
            path: Path to check.

        Raises:
            FileNotFoundError: If path does not exist.
        """
        if not path.exists():
            raise FileNotFoundError(f"Expected file not found: {path}")
