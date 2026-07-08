# Monitoramento

## O que monitorar

| Sinal | Fonte | Alerta quando |
|---|---|---|
| Disponibilidade | `GET /health` (healthcheck do Docker + proxy) | Falhas consecutivas |
| Latência p95 | Logs de acesso do uvicorn/proxy | Acima de ~100 ms sustentado |
| Taxa de erros 5xx | Logs de acesso | > 1% das requests |
| Taxa de 404 (cold start) | Logs de acesso | Crescimento indica base de usuários defasada |
| Staleness do modelo | Data do run em Production no MLflow Registry | Modelo sem retreino após novos dados |

## Como monitorar (fase atual)

- **Healthcheck**: o container da API expõe `/health`; o Docker marca o
  container como `unhealthy` e o proxy pode drenar tráfego.
- **Logs**: uvicorn loga cada request (método, rota, status). Em produção,
  os logs dos containers são acessíveis via `docker compose logs`.
- **Experimentos e registry**: MLflow UI mostra runs, métricas de treino e o
  modelo em Production — é a fonte de verdade sobre qual modelo está servindo.

## Degradação do modelo

O sinal de negócio (usuários interagindo com itens recomendados) não é
capturado nesta fase. Proxies observáveis:

- Aumento da taxa de 404: novos usuários que o modelo não conhece —
  indica necessidade de retreino com dados recentes.
- Métricas offline (`data/processed/metrics_comparison.csv`) recalculadas a
  cada `make repro`: comparar o modelo neural com os baselines antes de
  promover um novo run.

## Evolução futura

- Métricas estruturadas (Prometheus + Grafana) no lugar de logs.
- Log de payloads de recomendação para métricas online (CTR).
- Retreino agendado com gatilho por drift de dados.
