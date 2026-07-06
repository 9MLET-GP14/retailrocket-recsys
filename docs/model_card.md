# Model Card: RetailRocket EmbeddingMLP Recommender

## Descrição do Modelo

**Nome:** `retailrocket-embedding-mlp`
**Arquitetura:** MLP baseado em Embeddings (Filtragem Colaborativa Neural)
**Framework:** PyTorch
**Registro:** MLflow Model Registry

Modelo de filtragem colaborativa neural que aprende representações latentes densas
(embeddings) para usuários e itens, prevendo scores de interação por meio de um
perceptron multicamada. Os embeddings de usuário e item são concatenados e passados
por camadas totalmente conectadas com ativações ReLU e regularização por dropout.

### Diagrama da Arquitetura

```
user_id ─► Embedding(num_users, emb_dim) ─┐
                                           ├─► Concat ─► MLP ─► score
item_id ─► Embedding(num_items, emb_dim) ─┘
```

**Hiperparâmetros padrão**

| Parâmetro               | Padrão        |
|-------------------------|---------------|
| `embedding_dim`         | 64            |
| `hidden_dims`           | 256, 128, 64  |
| `dropout`               | 0.3           |
| `learning_rate`         | 0.001         |
| `batch_size`            | 512           |
| `max_epochs`            | 50            |
| `early_stopping_patience` | 5           |
| `seed`                  | 42            |

---

## Uso Previsto

- **Caso de uso principal:** Recomendação de produtos em plataforma de e-commerce,
  utilizando feedback implícito de sessões de navegação (visualizações, adições ao
  carrinho, transações).
- **Público-alvo:** Cientistas de dados e engenheiros de ML que implantam ou
  avaliam sistemas de recomendação personalizados.
- **Fora do escopo:** Usuários ou itens novos não vistos durante o treinamento
  (cold start); serviço em tempo real com requisitos de baixa latência sem
  embeddings pré-computados.

---

## Dados de Treinamento

| Propriedade      | Detalhe                                                      |
|------------------|--------------------------------------------------------------|
| Dataset          | Eventos de e-commerce RetailRocket (~2,75 M eventos brutos)  |
| Tipos de evento  | `view` (peso 1), `addtocart` (peso 3), `transaction` (peso 5) |
| Agregação        | Soma dos eventos ponderados por par (usuário, item)          |
| Filtragem        | Usuários e itens com menos de 5 interações removidos         |
| Divisão          | 80 % treino / 10 % validação / 10 % teste (estratificação aleatória) |
| Seed             | 42                                                           |

---

## Avaliação

As métricas são calculadas no conjunto de teste usando avaliação por ranking top-K.
Itens vistos no treinamento são excluídos da lista de candidatos antes do ranking.

| Métrica        | EmbeddingMLP | Popularity | SVD    |
|----------------|:------------:|:----------:|:------:|
| Precision@10   | 0.088        | 0.089      | 0.010  |
| Recall@10      | 0.479        | 0.535      | 0.040  |
| NDCG@10        | **0.412**    | 0.324      | 0.023  |
| HitRate@10     | 0.649        | 0.699      | 0.079  |

*Avaliação em dados sintéticos (3.000 usuários × 1.000 itens, 80.000 eventos).*
*O EmbeddingMLP supera a Popularidade em NDCG@10 (+27%), indicando melhor ordenação dos itens relevantes.*
*Execute `dvc repro` para re-gerar com seus dados e atualizar esta tabela.*

---

## Baselines Comparados

| Baseline              | Método                                                              |
|-----------------------|---------------------------------------------------------------------|
| `PopularityRecommender` | Ordena todos os itens pela soma global dos scores de interação; sem personalização |
| `SVDRecommender`      | SVD Truncado (50 componentes) na matriz esparsa usuário-item via sklearn |

Todos os baselines são avaliados com as mesmas quatro métricas e o mesmo corte K.

---

## Limitações e Vieses

- **Viés de popularidade:** Itens frequentemente interagidos dominam o sinal de
  treinamento, podendo sub-recomendar itens de cauda longa do catálogo.
- **Cold start:** Usuários e itens ausentes do treinamento não podem receber ou
  gerar recomendações significativas.
- **Feedback implícito:** A ausência de interação não sinaliza desinteresse —
  apenas exemplos positivos são observados.
- **Deriva temporal:** O modelo é treinado em um snapshot estático e não se adapta
  a tendências sazonais ou mudanças nas preferências dos usuários ao longo do tempo.
- **Contexto de sessão:** Os embeddings de usuário agregam todo o histórico de
  interações; o modelo não captura mudanças de intenção dentro de uma sessão.

---

## Considerações Éticas

- Nenhuma informação demográfica ou de identificação pessoal é utilizada — o modelo
  opera exclusivamente sobre logs de eventos de interação anonimizados.
- Loops de recomendação podem reforçar itens já populares, reduzindo a descoberta
  de produtos de nicho e potencialmente prejudicando vendedores menores.
- Retreinamento periódico e monitoramento de deriva de viés são recomendados antes
  de qualquer implantação em produção.

---

## Como Usar

```python
import torch
from src.models.factory import ModelFactory, ModelType

# Instanciar
model = ModelFactory.create(
    ModelType.EMBEDDING_MLP,
    num_users=...,
    num_items=...,
    embedding_dim=64,
    hidden_dims=[256, 128, 64],
)

# Carregar pesos de um checkpoint local
model.load_state_dict(torch.load("models/best_model.pt"))
model.eval()

# Pontuar um lote de pares (usuário, item)
user_ids = torch.tensor([0, 1, 2])
item_ids = torch.tensor([10, 20, 30])
scores = model(user_ids, item_ids)  # shape: (3, 1)
```

Ou carregar diretamente do MLflow:

```python
import mlflow.pytorch

model = mlflow.pytorch.load_model("models:/retailrocket-embedding-mlp/Production")
```

---

## MLflow Registry

O modelo é registrado com o nome **`retailrocket-embedding-mlp`**.

```bash
# Após o treinamento, promova o melhor run para Production:
python scripts/register_model.py   # ou: make register

# Abra a interface do MLflow para inspecionar runs e o registro:
make mlflow-up   # http://localhost:6060
```

Ciclo de vida: `None → Staging → Production`
