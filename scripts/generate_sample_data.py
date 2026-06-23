#!/usr/bin/env python
"""Generate synthetic RetailRocket-format data for local pipeline testing.

Produces data/raw/events.csv with realistic distributions (power-law item
popularity, three event types) so that dvc repro can run without downloading
the real dataset from Kaggle.

Usage:
    python scripts/generate_sample_data.py
    python scripts/generate_sample_data.py --users 2000 --items 800 --events 60000
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SEED = 42
EVENT_TYPES = ["view", "addtocart", "transaction"]
# Realistic proportions: ~88 % views, ~10 % addtocarts, ~2 % transactions
EVENT_PROBS = [0.88, 0.10, 0.02]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed namespace with users, items, and events counts.
    """
    parser = argparse.ArgumentParser(description="Generate synthetic RetailRocket data")
    parser.add_argument("--users", type=int, default=3_000)
    parser.add_argument("--items", type=int, default=1_000)
    parser.add_argument("--events", type=int, default=80_000)
    return parser.parse_args()


def power_law_weights(n: int, exponent: float = 1.5) -> np.ndarray:
    """Return normalised power-law weights for n items.

    Args:
        n: Number of elements.
        exponent: Steepness of the distribution (higher = more skewed).

    Returns:
        Float array of probabilities summing to 1.
    """
    ranks = np.arange(1, n + 1, dtype=float)
    weights = 1.0 / (ranks**exponent)
    return weights / weights.sum()


def generate_events(
    num_users: int, num_items: int, num_events: int, rng: np.random.Generator
) -> pd.DataFrame:
    """Create a synthetic events DataFrame matching the RetailRocket schema.

    Args:
        num_users: Total distinct users.
        num_items: Total distinct items.
        num_events: Total number of interaction rows to generate.
        rng: NumPy random generator for reproducibility.

    Returns:
        DataFrame with columns: timestamp, visitorid, event, itemid, transactionid.
    """
    item_weights = power_law_weights(num_items)
    user_weights = power_law_weights(num_users, exponent=1.2)

    visitor_ids = rng.choice(num_users, size=num_events, p=user_weights) + 1
    item_ids = rng.choice(num_items, size=num_events, p=item_weights) + 1
    events = rng.choice(EVENT_TYPES, size=num_events, p=EVENT_PROBS)

    # Timestamps spanning ~6 months in milliseconds
    base_ts = 1_430_000_000_000
    timestamps = base_ts + rng.integers(0, 180 * 24 * 3600 * 1000, size=num_events)

    tx_mask = events == "transaction"
    raw_ids = rng.integers(1_000_000, 9_999_999, size=num_events)
    tx_ids = np.where(tx_mask, raw_ids, np.nan)

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "visitorid": visitor_ids.astype(int),
            "event": events,
            "itemid": item_ids.astype(int),
            "transactionid": tx_ids,
        }
    )


def main() -> None:
    """Generate and save synthetic RetailRocket data."""
    args = parse_args()
    rng = np.random.default_rng(SEED)

    out_dir = Path("data/raw")
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Generating %d events for %d users × %d items...",
        args.events,
        args.users,
        args.items,
    )
    df = generate_events(args.users, args.items, args.events, rng)
    df = df.sort_values("timestamp").reset_index(drop=True)

    out_path = out_dir / "events.csv"
    df.to_csv(out_path, index=False)
    logger.info(
        "Saved %d events to %s  |  users: %d  |  items: %d",
        len(df),
        out_path,
        df["visitorid"].nunique(),
        df["itemid"].nunique(),
    )


if __name__ == "__main__":
    main()
