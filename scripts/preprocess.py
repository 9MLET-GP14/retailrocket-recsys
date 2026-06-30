#!/usr/bin/env python
"""Stage 1: load raw events, apply event weighting, filter sparse interactions."""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

from src.data.loader import RetailRocketLoader
from src.data.preprocessor import EventWeightPreprocessor, MinInteractionsFilter

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_params() -> dict[str, int]:
    """Load preprocess section from params.yaml.

    Returns:
        Dict with min_user and min_item keys.
    """
    with open("params.yaml") as fh:
        return yaml.safe_load(fh)["preprocess"]  # type: ignore[return-value]


def run(min_user: int, min_item: int) -> None:
    """Execute the preprocessing pipeline.

    Args:
        min_user: Minimum interactions required per user.
        min_item: Minimum interactions required per item.
    """
    events = RetailRocketLoader().load_events()
    events = EventWeightPreprocessor().transform(events)
    filt = MinInteractionsFilter(min_user=min_user, min_item=min_item)
    events = filt.transform(events)

    out = Path("data/processed/interactions.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    events.to_parquet(out, index=False)
    logger.info("Saved %d interactions to %s", len(events), out)


def main() -> None:
    """Entry point for DVC stage 1."""
    params = load_params()
    run(min_user=params["min_user"], min_item=params["min_item"])


if __name__ == "__main__":
    main()
