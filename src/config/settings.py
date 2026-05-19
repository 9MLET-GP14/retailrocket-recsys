"""Application settings loaded from environment variables."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration via Pydantic Settings.

    All values are loaded from .env or environment variables.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Paths
    data_raw_path: str = Field(default="data/raw")
    data_processed_path: str = Field(default="data/processed")
    models_path: str = Field(default="models")

    # MLflow
    mlflow_tracking_uri: str = Field(default="http://localhost:5000")
    mlflow_experiment_name: str = Field(default="retailrocket-recsys")

    # DVC Remote
    dvc_remote_url: str | None = Field(default=None)

    # Model hyperparameters
    embedding_dim: int = Field(default=64)
    hidden_dims: str = Field(default="256,128,64")
    dropout: float = Field(default=0.3)
    learning_rate: float = Field(default=0.001)
    batch_size: int = Field(default=512)
    max_epochs: int = Field(default=50)
    early_stopping_patience: int = Field(default=5)
    seed: int = Field(default=42)

    # Evaluation
    top_k: int = Field(default=10)

    @property
    def hidden_dims_list(self) -> list[int]:
        """Parse hidden_dims string into a list of integers."""
        return [int(d) for d in self.hidden_dims.split(",")]


settings = Settings()
