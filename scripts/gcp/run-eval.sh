#!/bin/bash
# Run evaluation on the GCP VM.
# Usage: SSH into the VM, then: bash scripts/gcp/run-eval.sh [config] [extra args]

set -euo pipefail

CONFIG="${1:-configs/full.yaml}"
shift 2>/dev/null || true

cd ~/experiment-llm-refusal

echo "=== Running eval with $CONFIG ==="
docker run --gpus all \
    -v "$(pwd)/outputs:/app/outputs" \
    -v "$(pwd)/hf-cache:/hf-cache" \
    -e HF_HOME=/hf-cache \
    safety-finetune \
    safety_finetune.eval --config "$CONFIG" "$@"
