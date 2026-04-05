FROM python:3.11-slim

# Works on both ARM (Mac) and x86 (GCP)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml .
COPY src/ src/

# Install base deps. bitsandbytes needs special handling on ARM vs x86.
RUN pip install --no-cache-dir -e ".[dev]"

COPY configs/ configs/
COPY scripts/ scripts/
COPY data/ data/
COPY notebooks/ notebooks/

ENTRYPOINT ["python", "-m"]
CMD ["safety_finetune.train", "--config", "configs/smoke.yaml"]
