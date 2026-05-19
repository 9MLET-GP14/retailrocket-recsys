# RetailRocket RecSys 🛒

Sistema de recomendação de produtos baseado no comportamento de navegação de usuários do dataset RetailRocket.

## Stack

| Camada | Tecnologia |
|--------|-----------|
| Modelo | PyTorch (MLP + Embeddings) |
| Baseline | Scikit-Learn |
| Tracking | MLflow |
| Versionamento | DVC |
| Container | Docker multi-stage |
| Deps | Poetry |

## Estrutura

```
retailrocket-recsys/
├── src/
│   ├── config/         # Pydantic Settings
│   ├── data/           # Loaders e preprocessors
│   ├── features/       # Feature engineering
│   ├── models/         # Factory + implementações
│   ├── training/       # Train loop, early stopping
│   └── evaluation/     # Métricas e comparação
├── tests/
├── data/
│   ├── raw/
│   └── processed/
├── models/             # Artefatos salvos
├── configs/            # YAML de experimentos
├── scripts/            # validate_env.py, etc.
├── notebooks/          # EDA
├── .dvc/
├── dvc.yaml
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
└── .env.example
```

## Quickstart

```bash
# 1. Instalar dependências
poetry install

# 2. Validar ambiente
python scripts/validate_env.py

# 3. Configurar variáveis
cp .env.example .env

# 4. Reproduzir pipeline
dvc repro

# 5. Ver experimentos
mlflow ui
```

## Pipeline DVC

```
preprocess → feature_eng → train → evaluate
```

## Métricas principais

- Precision@K
- Recall@K
- NDCG@K
- Hit Rate
