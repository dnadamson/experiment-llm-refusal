#!/bin/bash
# Run training on the GCP VM.
# Usage: SSH into the VM, then: bash scripts/gcp/run-training.sh [config]

set -euo pipefail

CONFIG="${1:-configs/full.yaml}"

cd ~/experiment-llm-refusal

echo "=== Starting training with $CONFIG ==="
docker run --gpus all \
    -v "$(pwd)/outputs:/app/outputs" \
    -v "$(pwd)/hf-cache:/hf-cache" \
    -e HF_HOME=/hf-cache \
    safety-finetune \
    safety_finetune.train --config "$CONFIG"

echo ""
echo "=== Training complete ==="
echo "Run eval:"
echo "  bash scripts/gcp/run-eval.sh"
