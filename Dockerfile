# Pin to digest for reproducibility and supply-chain safety
FROM python:3.11-slim@sha256:9358444059ed78e2975ada2c189f1c1a3144a5dab6f35bff8c981afb38946634

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml .
COPY requirements.lock .
COPY src/ src/

# Install pinned deps, then the project (non-editable)
RUN pip install --no-cache-dir -r requirements.lock && \
    pip install --no-cache-dir --no-deps .

COPY configs/ configs/
COPY scripts/ scripts/
COPY data/ data/
COPY notebooks/ notebooks/

# Run as non-root
RUN useradd --create-home appuser
USER appuser

ENTRYPOINT ["python", "-m"]
CMD ["safety_finetune.train", "--config", "configs/smoke.yaml"]
