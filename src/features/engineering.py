"""Feature engineering: encode user/item IDs and build train/val/test splits."""

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.config import settings


@dataclass
class EncodedDataset:
    """Holds encoded interaction data and label encoders.

    Attributes:
        df: DataFrame with encoded user_idx, item_idx, score columns.
        user_encoder: Fitted LabelEncoder for users.
        item_encoder: Fitted LabelEncoder for items.
        num_users: Number of unique users.
        num_items: Number of unique items.
    """

    df: pd.DataFrame
    user_encoder: LabelEncoder
    item_encoder: LabelEncoder
    num_users: int
    num_items: int


@dataclass
class DataSplits:
    """Train / validation / test DataFrames.

    Attributes:
        train: Training interactions.
        val: Validation interactions.
        test: Test interactions.
    """

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def encode_ids(df: pd.DataFrame) -> EncodedDataset:
    """Encode visitorid and itemid to contiguous integer indices.

    Args:
        df: Aggregated DataFrame with columns: visitorid, itemid, score.

    Returns:
        EncodedDataset with zero-indexed integer IDs.
    """
    user_enc = LabelEncoder()
    item_enc = LabelEncoder()
    df = df.copy()
    df["user_idx"] = user_enc.fit_transform(df["visitorid"])
    df["item_idx"] = item_enc.fit_transform(df["itemid"])
    return EncodedDataset(
        df=df,
        user_encoder=user_enc,
        item_encoder=item_enc,
        num_users=len(user_enc.classes_),
        num_items=len(item_enc.classes_),
    )


def split_interactions(
    df: pd.DataFrame,
    val_size: float = 0.1,
    test_size: float = 0.1,
) -> DataSplits:
    """Split interactions into train / val / test sets.

    Args:
        df: Encoded interactions DataFrame.
        val_size: Fraction for validation set.
        test_size: Fraction for test set.

    Returns:
        DataSplits with train, val, and test DataFrames.
    """
    seed = settings.seed
    train_val, test = train_test_split(df, test_size=test_size, random_state=seed)
    relative_val = val_size / (1 - test_size)
    train, val = train_test_split(train_val, test_size=relative_val, random_state=seed)
    return DataSplits(
        train=train.reset_index(drop=True),
        val=val.reset_index(drop=True),
        test=test.reset_index(drop=True),
    )
