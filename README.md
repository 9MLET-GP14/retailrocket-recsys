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
| Deps | uv |

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

### 1. Instalar dependências (via uv)
Certifique-se de ter o `uv` instalado. Se não tiver, instale com `curl -LsSf https://astral.sh/uv/install.sh | sh`.

```bash
# Criar ambiente virtual e instalar dependências
uv venv
source venv/bin/activate
uv pip install -e .

2. Validar ambiente
Bash
python scripts/validate_env.py

3. Configurar variáveis
Bash
cp .env.example .env

4. Executar Testes e Qualidade
Bash
# Rodar testes com cobertura
pytest -v --cov=src --cov-report=term-missing

5. Reproduzir pipeline (DVC )
Bash
dvc repro

6. Ver experimentos
Bash
mlflow ui
Pipeline DVC
O pipeline segue o fluxo: preprocess → feature_eng → train → evaluate.
Métricas principais
Precision@K
Recall@K
NDCG@K
Hit Rate
