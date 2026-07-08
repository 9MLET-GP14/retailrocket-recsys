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
├── tests/              # pytest unit tests
├── scripts/
│   ├── train.py        # Pipeline completo: load → treinar → avaliar → comparar
│   ├── register_model.py  # Promove melhor run para Production no MLflow Registry
│   └── validate_env.py    # Verifica pacotes e variáveis de ambiente
├── notebooks/          # EDA e experimentos exploratórios (fora do pipeline)
├── data/
│   ├── raw/            # CSVs originais do RetailRocket (gerenciados pelo DVC)
│   └── processed/      # Artefatos intermediários e métricas
├── models/             # Checkpoints locais
├── build/
│   ├── Dockerfile          # Multi-stage: builder + runtime
│   └── docker-compose.yml  # Serviços: training + MLflow server
├── docs/
│   └── model_card.md   # Documentação do modelo
├── dvc.yaml            # Pipeline DVC: preprocess → feature_eng → train → evaluate
├── params.yaml         # Hiperparâmetros do pipeline (fonte única, lida pelo DVC)
├── Makefile            # Interface padrão de comandos (make help)
├── pyproject.toml      # Dependências prod/dev
├── uv.lock             # Lock file gerado por `uv lock`
├── .python-version     # Python 3.11 (mesma versão da imagem Docker)
└── .env.example        # Template de variáveis de ambiente
```

## Quickstart

Sequência completa, do clone ao modelo servido (requer `uv` e Docker;
`make help` lista todos os comandos):

```bash
# 1. Setup
make install                # cria .venv (Python 3.11) a partir do uv.lock
cp .env.example .env
make validate               # opcional: confere pacotes e variáveis de ambiente

# 2. Dados e pipeline
make data                   # baixa o dataset RetailRocket (~940 MB) para data/raw/
make repro                  # pipeline DVC: preprocess → feature_eng → train → evaluate
make register               # promove o melhor run a Production no MLflow Registry

# 3. Serving
make docker-up              # builda e sobe API + MLflow server via Docker Compose
```

Ao final:

- **API**: <http://localhost:6061> (Swagger em `/docs`)
- **MLflow UI**: <http://localhost:6060> (runs, métricas e Model Registry)

Notas:

- Os comandos que usam MLflow (`repro`, `train`, `evaluate`, `register`)
  sobem o container do MLflow automaticamente e aguardam o healthcheck.
- A imagem da API embute os artefatos de inferência gerados pelo `make repro`,
  portanto o pipeline precisa rodar antes do `make docker-up`.
- `make data` usa o `kagglehub` (dataset público, sem credenciais) e é
  idempotente. Download manual:
  <https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset> →
  CSVs em `data/raw/` (`category_tree.csv` é usado apenas nos notebooks de EDA).
- `make test` roda a suíte com coverage; `make clean` desfaz tudo
  (containers, volumes, imagens e artefatos do pipeline — preserva `data/raw/`).

### O que o pipeline executa

1. Carrega eventos do RetailRocket
2. Aplica pesos por tipo de evento e filtra interações esparsas
3. Codifica IDs e divide em train / val / test
4. Treina o `EmbeddingMLPRecommender` (PyTorch) com early stopping
5. Treina baselines Scikit-Learn (Popularity, SVD)
6. Avalia todos com Precision@10, Recall@10, NDCG@10, HitRate@10
7. Salva a comparação em `data/processed/metrics_comparison.csv`

## API de Recomendação

```bash
make serve       # desenvolvimento local (hot reload)
make docker-up   # produção: API + MLflow server via Docker Compose
```

| Endpoint | Descrição |
|----------|-----------|
| `GET /` | Informações do serviço e rotas disponíveis |
| `GET /health` | Status e dimensões do modelo carregado |
| `GET /recommendations/{user_id}?k=N` | Top-K itens não vistos para o usuário (1 ≤ k ≤ 100) |
| `GET /docs` | Documentação interativa (Swagger UI) |

A API roda na porta `6061` e o MLflow na `6060`. Usuários fora do treino
(cold start) recebem `404`. Os artefatos de inferência são embutidos na
imagem Docker no build — a API não depende do MLflow em runtime.

### Exemplo de uso

```bash
curl "http://localhost:6061/recommendations/51?k=3"
```

```json
{
  "user_id": 51,
  "k": 3,
  "items": [
    { "item_id": 119736, "score": 5.658553123474121 },
    { "item_id": 9877, "score": 4.803702354431152 },
    { "item_id": 461686, "score": 4.394866466522217 }
  ]
}
```

Usuário desconhecido:

```bash
curl "http://localhost:6061/recommendations/1"
```

```json
{ "detail": "user_id 1 not found in training data (cold start)" }
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

Consulte [docs/model_card.md](docs/model_card.md) para arquitetura detalhada, limitações,
vieses e instruções de uso do modelo.
