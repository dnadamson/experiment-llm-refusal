# CUDA base for GPU support (bitsandbytes needs CUDA runtime)
# Falls back gracefully to CPU when no GPU is available
FROM nvidia/cuda:12.6.3-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip curl && \
    ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml .
COPY requirements.lock .
COPY src/ src/

# Install pinned deps, then the project (non-editable)
RUN pip install --no-cache-dir -r requirements.lock && \
    pip install --no-cache-dir --no-deps ".[dev]"

COPY configs/ configs/
COPY scripts/ scripts/
COPY data/ data/
COPY notebooks/ notebooks/
COPY tests/ tests/

# Run as non-root
RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

ENTRYPOINT ["python", "-m"]
CMD ["safety_finetune.train", "--config", "configs/smoke.yaml"]
