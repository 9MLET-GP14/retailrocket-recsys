"""Pytest configuration and shared fixtures."""

import random

import numpy as np
import pytest
import torch

from src.config import settings


@pytest.fixture(autouse=True)
def fix_seeds() -> None:
    """Fix all random seeds for reproducibility in every test."""
    seed = settings.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
