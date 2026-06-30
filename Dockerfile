# ── Stage 1: builder — instala dependências de produção ──────────────────────
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# Instala uv para resolução de dependências a partir do lock file
RUN pip install uv==0.11.18

COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/

# Cria venv isolado e instala apenas dependências de produção
# Usa índice do PyTorch para versão CPU (imagem menor)
RUN python -m uv venv /opt/venv --python python3.11
ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

RUN uv pip install \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        .


# ── Stage 2: runtime — imagem mínima sem ferramentas de build ────────────────
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Copia apenas o venv construído no estágio anterior
COPY --from=builder /opt/venv /opt/venv

# Copia código-fonte, scripts e configurações
COPY src/     ./src/
COPY scripts/ ./scripts/
COPY configs/ ./configs/
COPY params.yaml ./

# Diretórios de dados e modelos são montados via volume em runtime
RUN mkdir -p data/raw data/processed models

CMD ["python", "scripts/train.py"]
