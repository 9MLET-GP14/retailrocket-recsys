# RetailRocket RecSys

Sistema de recomendação de produtos baseado no comportamento de navegação de usuários do dataset RetailRocket.

## Stack

| Camada        | Tecnologia                          |
|---------------|-------------------------------------|
| Modelo        | PyTorch (EmbeddingMLP)              |
| Baselines     | Scikit-Learn (Popularity, SVD)      |
| Tracking      | MLflow + Model Registry             |
| Versionamento | DVC                                 |
| Container     | Docker multi-stage                  |
| Dependências  | uv + uv.lock                        |

## Estrutura

```
retailrocket-recsys/
├── src/
│   ├── config/         # Pydantic Settings (carregadas do .env)
│   ├── data/           # RetailRocketLoader + Strategy preprocessors
│   ├── features/       # Encode IDs + train/val/test split
│   ├── models/         # ModelFactory, EmbeddingMLP, baselines, Dataset
│   ├── training/       # Train loop, EarlyStopping, MLflow tracking
│   └── evaluation/     # Precision/Recall/NDCG/HitRate @K
├── tests/              # pytest unit tests (≥ 80 % coverage)
├── scripts/
│   ├── train.py        # Pipeline completo: load → treinar → avaliar → comparar
│   ├── register_model.py  # Promove melhor run para Production no MLflow Registry
│   └── validate_env.py    # Verifica pacotes e variáveis de ambiente
├── configs/
│   └── default.yaml    # Hiperparâmetros padrão do experimento
├── data/
│   ├── raw/            # CSVs originais do RetailRocket (gerenciados pelo DVC)
│   └── processed/      # Artefatos intermediários e métricas
├── models/             # Checkpoints locais
├── MODEL_CARD.md       # Documentação do modelo
├── dvc.yaml            # Pipeline DVC: preprocess → feature_eng → train → evaluate
├── docker-compose.yml  # Serviços: training + MLflow server
├── Dockerfile          # Multi-stage: builder + runtime
├── pyproject.toml      # Dependências prod/dev
├── uv.lock             # Lock file gerado por `uv lock`
└── .env.example        # Template de variáveis de ambiente
```

## Quickstart

### 1. Clonar e instalar dependências

```bash
# Instalar uv (caso ainda não tenha)
pip install uv

# Criar ambiente virtual e instalar pacotes a partir do lock file
uv venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

uv sync
```

### 2. Configurar variáveis de ambiente

```bash
cp .env.example .env
# Edite .env conforme necessário (caminhos, MLflow URI, hiperparâmetros)
```

### 3. Validar ambiente

```bash
python scripts/validate_env.py
```

### 4. Baixar o dataset RetailRocket

Faça o download manual em <https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset>
e coloque os três arquivos CSV em `data/raw/`:

```
data/raw/
├── events.csv
├── item_properties_part1.csv
└── item_properties_part2.csv
```

### 5. Executar testes

```bash
pytest -v --cov=src --cov-report=term-missing
```

### 6. Treinar e avaliar

```bash
# Garanta que o MLflow Tracking Server esteja rodando:
mlflow server --host 0.0.0.0 --port 5000 &

python scripts/train.py
```

O script executa o pipeline completo:

1. Carrega eventos do RetailRocket
2. Aplica pesos por tipo de evento e filtra interações esparsas
3. Codifica IDs e divide em train / val / test
4. Treina o `EmbeddingMLPRecommender` (PyTorch) com early stopping
5. Treina baselines Scikit-Learn (Popularity, SVD)
6. Avalia todos com Precision@10, Recall@10, NDCG@10, HitRate@10
7. Salva a comparação em `data/processed/metrics_comparison.csv`

### 7. Registrar modelo no MLflow Registry

```bash
python scripts/register_model.py
```

Promove automaticamente o melhor run para `Staging → Production`.

### 8. Visualizar experimentos

```bash
mlflow ui  # Abre em http://localhost:5000
```

## Métricas Avaliadas

| Métrica       | Descrição                                               |
|---------------|---------------------------------------------------------|
| Precision@K   | Fração das K recomendações que são relevantes           |
| Recall@K      | Fração dos itens relevantes encontrados no top-K        |
| NDCG@K        | Ganho cumulativo descontado normalizado até posição K   |
| HitRate@K     | Fração dos usuários com ao menos 1 acerto no top-K      |

## Design Patterns

| Padrão   | Onde                                             |
|----------|--------------------------------------------------|
| Factory  | `ModelFactory.create(ModelType, **kwargs)`       |
| Strategy | `PreprocessorStrategy` → `EventWeightPreprocessor`, `MinInteractionsFilter` |

## Pipeline DVC

```
preprocess → feature_eng → train → evaluate
```

```bash
dvc repro   # Reproduz o pipeline completo
dvc dag     # Visualiza o DAG de dependências
```

## Reprodutibilidade

- Seed fixado em todas as bibliotecas: Python `random`, NumPy, PyTorch.
- Lock file `uv.lock` (196 pacotes resolvidos) garante instalação determinística.
- Configurações externalizadas via `.env` + Pydantic `Settings`.

## Desenvolvimento

```bash
# Linting e formatação
ruff check src tests scripts
ruff format src tests scripts

# Pre-commit hooks (ruff + trailing-whitespace + end-of-file-fixer)
pre-commit run --all-files
```

## Model Card

Consulte [MODEL_CARD.md](MODEL_CARD.md) para arquitetura detalhada, limitações,
vieses e instruções de uso do modelo.
