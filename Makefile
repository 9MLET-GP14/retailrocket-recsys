.PHONY: help install lock lint format test validate data repro train evaluate register lab mlflow-up mlflow-down docker docker-up docker-down clean

UV  ?= uv
RUN := $(UV) run

help:
	@echo "make install     - sync .venv from pyproject.toml + uv.lock (with dev extras)"
	@echo "make lock        - regenerate uv.lock from pyproject.toml"
	@echo "make lint        - ruff check (no fixes)"
	@echo "make format      - ruff format + ruff --fix"
	@echo "make test        - pytest with coverage"
	@echo "make validate    - check environment (packages + env vars)"
	@echo "make data        - download the RetailRocket dataset into data/raw/"
	@echo "make repro       - run the full DVC pipeline (preprocess -> evaluate)"
	@echo "make train       - run only the training stage"
	@echo "make evaluate    - run only the evaluation stage"
	@echo "make register    - promote best run to Production in MLflow Registry"
	@echo "make lab         - launch JupyterLab (notebooks/)"
	@echo "make mlflow-up   - start the MLflow server container (http://localhost:6060)"
	@echo "make mlflow-down - stop the MLflow server container"
	@echo "make docker      - build the Docker images"
	@echo "make docker-up   - rebuild and start the containers"
	@echo "make docker-down - stop and remove the containers"
	@echo "make clean       - remove caches and build artifacts"

install:
	$(UV) sync --extra dev

lock:
	$(UV) lock

lint:
	$(RUN) ruff check src tests scripts

format:
	$(RUN) ruff format src tests scripts
	$(RUN) ruff check --fix src tests scripts

test:
	$(RUN) pytest

validate:
	$(RUN) python scripts/validate_env.py

data:
	$(RUN) python scripts/download_data.py

repro:
	$(RUN) dvc repro

train:
	$(RUN) python scripts/train.py

evaluate:
	$(RUN) python scripts/evaluate.py

register:
	$(RUN) python scripts/register_model.py

lab:
	$(RUN) --extra jupyter jupyter lab

mlflow-up:
	docker compose -f build/docker-compose.yml up -d mlflow

mlflow-down:
	docker compose -f build/docker-compose.yml stop mlflow

docker:
	docker compose -f build/docker-compose.yml build

docker-up:
	docker compose -f build/docker-compose.yml up -d --build

docker-down:
	docker compose -f build/docker-compose.yml down

clean:
	rm -rf dist .pytest_cache .ruff_cache .coverage htmlcov
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name "*.egg-info" -prune -not -path "./.venv/*" -exec rm -rf {} +
