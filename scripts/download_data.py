#!/usr/bin/env python
"""Download the RetailRocket dataset from Kaggle into data/raw/."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import kagglehub
from dotenv import load_dotenv

from src.config.settings import settings

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

EXPECTED_FILES = (
    "category_tree.csv",
    "events.csv",
    "item_properties_part1.csv",
    "item_properties_part2.csv",
)


def missing_files(raw_dir: Path) -> list[str]:
    """List expected dataset files not yet present in raw_dir.

    Args:
        raw_dir: Directory where the raw CSVs should live.

    Returns:
        Names of the expected files that are missing.
    """
    return [name for name in EXPECTED_FILES if not (raw_dir / name).exists()]


def download(raw_dir: Path, files: list[str]) -> None:
    """Fetch the dataset via kagglehub and copy the CSVs into raw_dir.

    Kaggle credentials (KAGGLE_USERNAME/KAGGLE_KEY) are optional for this
    public dataset; kagglehub falls back to anonymous download.

    Args:
        raw_dir: Destination directory for the raw CSVs.
        files: File names to copy from the kagglehub cache.

    Raises:
        FileNotFoundError: If an expected file is absent from the download.
    """
    cache_dir = Path(kagglehub.dataset_download(settings.kaggle_dataset_slug))
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name in files:
        source = next(cache_dir.rglob(name), None)
        if source is None:
            raise FileNotFoundError(f"{name} not found in download: {cache_dir}")
        shutil.copy2(source, raw_dir / name)
        logger.info("Copied %s to %s", name, raw_dir)


def main() -> None:
    """Entry point: download only the files that are missing."""
    load_dotenv()
    raw_dir = Path(settings.data_raw_path)
    files = missing_files(raw_dir)
    if not files:
        logger.info("All dataset files already present in %s", raw_dir)
        return
    download(raw_dir, files)


if __name__ == "__main__":
    main()
