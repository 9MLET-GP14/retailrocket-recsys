"""Training loop with early stopping and MLflow tracking."""

import logging

import mlflow
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import settings

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class EarlyStopping:
    """Stop training when validation loss stops improving.

    Args:
        patience: Epochs to wait before stopping.
        min_delta: Minimum improvement to count as progress.
    """

    def __init__(self, patience: int = 5, min_delta: float = 1e-4) -> None:
        """Initialize EarlyStopping.

        Args:
            patience: Epochs to wait before stopping.
            min_delta: Minimum improvement to count as progress.
        """
        self._patience = patience
        self._min_delta = min_delta
        self._counter = 0
        self._best_loss: float = float("inf")

    @property
    def should_stop(self) -> bool:
        """Return True if training should stop."""
        return self._counter >= self._patience

    def step(self, val_loss: float) -> None:
        """Update state with new validation loss.

        Args:
            val_loss: Current epoch validation loss.
        """
        if val_loss < self._best_loss - self._min_delta:
            self._best_loss = val_loss
            self._counter = 0
        else:
            self._counter += 1


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Run one training epoch.

    Args:
        model: PyTorch model to train.
        loader: Training DataLoader.
        optimizer: Optimizer instance.
        criterion: Loss function.
        device: CPU or CUDA device.

    Returns:
        Mean training loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    for user_ids, item_ids, scores in loader:
        user_ids = user_ids.to(device)
        item_ids = item_ids.to(device)
        scores = scores.float().to(device)
        optimizer.zero_grad()
        preds = model(user_ids, item_ids).squeeze()
        loss = criterion(preds, scores)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Run one evaluation epoch.

    Args:
        model: PyTorch model to evaluate.
        loader: Validation DataLoader.
        criterion: Loss function.
        device: CPU or CUDA device.

    Returns:
        Mean validation loss for the epoch.
    """
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for user_ids, item_ids, scores in loader:
            user_ids = user_ids.to(device)
            item_ids = item_ids.to(device)
            scores = scores.float().to(device)
            preds = model(user_ids, item_ids).squeeze()
            total_loss += criterion(preds, scores).item()
    return total_loss / len(loader)


def run_training(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    run_name: str = "embedding_mlp",
) -> nn.Module:
    """Full training loop with MLflow tracking and early stopping.

    Args:
        model: Initialized PyTorch model.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        run_name: MLflow run name.

    Returns:
        Trained model with best validation loss weights.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on device: %s", device)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.learning_rate)
    criterion = nn.MSELoss()
    early_stopping = EarlyStopping(patience=settings.early_stopping_patience)

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    with mlflow.start_run(run_name=run_name):
        _log_hyperparams()
        best_weights = model.state_dict()

        for epoch in range(1, settings.max_epochs + 1):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
            val_loss = evaluate_epoch(model, val_loader, criterion, device)
            mlflow.log_metrics(
                {"train_loss": train_loss, "val_loss": val_loss}, step=epoch
            )
            logger.info(
                "Epoch %03d | train_loss: %.4f | val_loss: %.4f",
                epoch,
                train_loss,
                val_loss,
            )
            early_stopping.step(val_loss)
            if early_stopping._counter == 0:
                best_weights = model.state_dict().copy()
            if early_stopping.should_stop:
                logger.info("Early stopping triggered at epoch %d", epoch)
                break

        model.load_state_dict(best_weights)
        mlflow.pytorch.log_model(model, artifact_path="model")
        logger.info("Training complete. Model logged to MLflow.")

    return model


def _log_hyperparams() -> None:
    """Log all relevant hyperparameters to the active MLflow run."""
    mlflow.log_params(
        {
            "embedding_dim": settings.embedding_dim,
            "hidden_dims": settings.hidden_dims,
            "dropout": settings.dropout,
            "learning_rate": settings.learning_rate,
            "batch_size": settings.batch_size,
            "max_epochs": settings.max_epochs,
            "early_stopping_patience": settings.early_stopping_patience,
            "seed": settings.seed,
        }
    )
