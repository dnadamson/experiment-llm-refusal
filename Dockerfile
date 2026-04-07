# CUDA base for GPU support (bitsandbytes needs CUDA runtime)
# Falls back gracefully to CPU when no GPU is available
FROM nvidia/cuda:12.6.3-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip curl && \
    ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PYTHONPATH=/app/src
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

COPY pyproject.toml .
COPY requirements.lock .
COPY src/ src/

# Install pinned deps with Python 3.11 explicitly
RUN python3.11 -m pip install --no-cache-dir -r requirements.lock && \
    python3.11 -m pip install --no-cache-dir --no-deps ".[dev]"

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
