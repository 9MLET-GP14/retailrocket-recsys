#!/usr/bin/env python
"""Promote the best MLflow run's model through Staging to Production."""

from __future__ import annotations

import logging

import mlflow
from mlflow import MlflowClient

from src.config import settings

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

MODEL_NAME = "retailrocket-embedding-mlp"


def get_best_run_id(experiment_name: str, metric: str = "val_loss") -> str:
    """Find the MLflow run with the lowest validation loss.

    Args:
        experiment_name: Name of the MLflow experiment to search.
        metric: Metric name to optimise (ascending order).

    Returns:
        Run ID string of the best run.

    Raises:
        ValueError: If the experiment or any runs are not found.
    """
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found.")
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} ASC"],
        max_results=1,
    )
    if not runs:
        raise ValueError(
            f"No runs found in experiment '{experiment_name}'. "
            "Run scripts/train.py first."
        )
    best = runs[0]
    run_id = best.info.run_id
    loss = best.data.metrics.get(metric, float("nan"))
    logger.info("Best run: %s  (%s=%.4f)", run_id, metric, loss)
    return run_id


def register_and_promote(run_id: str) -> None:
    """Register the model artifact and transition it to Production.

    The model is registered under ``MODEL_NAME``, then transitioned
    Staging → Production in two steps so the audit trail is preserved.

    Args:
        run_id: MLflow run ID that contains the ``model`` artifact.
    """
    client = MlflowClient()
    logger.info("Registering model from run %s ...", run_id)
    model_version = mlflow.register_model(
        model_uri=f"runs:/{run_id}/model",
        name=MODEL_NAME,
    )
    version = model_version.version
    logger.info("Registered as '%s' version %s", MODEL_NAME, version)

    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=version,
        stage="Staging",
    )
    logger.info("Transitioned version %s to Staging", version)

    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=version,
        stage="Production",
    )
    logger.info("Transitioned version %s to Production", version)


def main() -> None:
    """Entry point: find best run and register its model to Production."""
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    run_id = get_best_run_id(settings.mlflow_experiment_name)
    register_and_promote(run_id)
    logger.info(
        "Model '%s' is now live in Production. Run `mlflow ui` to inspect.",
        MODEL_NAME,
    )


if __name__ == "__main__":
    main()
