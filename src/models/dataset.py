"""PyTorch Dataset for user-item interaction DataFrames."""

from __future__ import annotations

import pandas as pd
import torch
from torch.utils.data import Dataset


class InteractionDataset(Dataset):
    """Wraps a user-item interaction DataFrame as a PyTorch Dataset.

    Args:
        df: DataFrame with integer columns ``user_idx``, ``item_idx``
            and numeric column ``score``.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        """Initialise by pre-loading tensors from the DataFrame."""
        self._user_ids = torch.tensor(df["user_idx"].values, dtype=torch.long)
        self._item_ids = torch.tensor(df["item_idx"].values, dtype=torch.long)
        self._scores = torch.tensor(df["score"].values, dtype=torch.float32)

    def __len__(self) -> int:
        """Return number of interaction rows."""
        return len(self._user_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the interaction tensors at position ``idx``.

        Args:
            idx: Row index.

        Returns:
            Tuple of (user_id, item_id, score) scalar tensors.
        """
        return self._user_ids[idx], self._item_ids[idx], self._scores[idx]
