# Deploy — API de Recomendação

## Decisão: API real-time vs. batch

| Critério | API real-time (escolhida) | Batch (pré-computado) |
|---|---|---|
| Latência de resposta | ~dezenas de ms por request | Instantânea (lookup) |
| Frescor das recomendações | Score calculado na hora | Defasado até o próximo job |
| Custo de infraestrutura | 1 container leve, CPU | Storage + job scheduler |
| Complexidade | Baixa (stateless) | Média (pipeline extra) |

A inferência do `EmbeddingMLP` para um usuário (score de todos os itens + top-K)
é barata o suficiente em CPU para servir online, e a API stateless simplifica
o deploy e o scale-out horizontal.

## Topologia

```
Cliente ──> Reverse proxy (TLS, rate limit)
              ├── /          ──> API FastAPI/uvicorn  (127.0.0.1:6061)
              └── (subdomínio) ─> MLflow UI            (127.0.0.1:6060)
```

Ambos os serviços rodam via Docker Compose (`build/docker-compose.yml`) no
mesmo host, expostos apenas em `localhost`; o reverse proxy faz a terminação
TLS e é o único ponto público.

## Artefatos de inferência

A imagem da API (`target: api` no Dockerfile) embute os artefatos produzidos
pelo pipeline DVC — não há dependência de rede no startup:

| Artefato | Papel |
|---|---|
| `models/best_model.pt` | Checkpoint self-describing: pesos + arquitetura (dims, hidden layers, dropout) |
| `data/processed/encoders.pkl` | LabelEncoders (ids originais ↔ índices) |
| `data/processed/interactions.parquet` | Itens já vistos por usuário (masking) |

Para atualizar o modelo em produção: rodar o pipeline (`make repro`),
reconstruir a imagem (`make docker-up`).

## Endpoints

| Endpoint | Descrição |
|---|---|
| `GET /health` | Status + dimensões do modelo carregado (usado pelo healthcheck) |
| `GET /recommendations/{user_id}?k=N` | Top-K itens não vistos para o usuário (k entre 1 e 100) |
| `GET /docs` | OpenAPI/Swagger gerado pelo FastAPI |

Usuários fora do treino (cold start) recebem `404` — estratégia de fallback
está fora do escopo desta fase (ver limitações no model card).

## Passo a passo do deploy

1. Local: `make serve` (uvicorn com reload) e testes com `make test`.
2. Container local: `make docker-up` e smoke test em `localhost:6061/health`.
3. Servidor: clonar o repositório, disponibilizar os artefatos de inferência
   (pipeline ou DVC pull), `docker compose -f build/docker-compose.yml up -d --build`.
4. Configurar o reverse proxy com TLS apontando para as portas locais
   (API em `6061`, MLflow em `6060` com autenticação básica).

## Modos de falha

| Falha | Comportamento | Mitigação |
|---|---|---|
| Artefatos ausentes na imagem | Startup falha com `ArtifactsMissingError` | Healthcheck impede tráfego; rebuild com artefatos |
| Container cai | — | `restart: unless-stopped` no compose |
| Usuário desconhecido | `404` controlado | Documentado; fallback futuro |
| MLflow indisponível | Sem efeito na API | API não depende do MLflow em runtime |

## Segurança

- Containers expostos somente em `127.0.0.1`; TLS e rate limit no proxy.
- MLflow UI sem autenticação própria → proteger com auth básica no proxy.
- API roda com usuário não-root (`apiuser`) na imagem.
- Autenticação de API (API key/JWT) fora do escopo desta fase; o design
  stateless permite adicioná-la via dependency do FastAPI sem refatoração.
