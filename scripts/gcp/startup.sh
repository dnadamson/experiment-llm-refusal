#!/bin/bash
# GCP VM startup script for training
# Usage: gcloud compute instances create ... --metadata-from-file startup-script=scripts/gcp/startup.sh

set -euo pipefail

# Install NVIDIA drivers + Docker
apt-get update
apt-get install -y nvidia-driver-535 docker.io nvidia-container-toolkit

systemctl restart docker

# Clone repo and run training
cd /opt
git clone https://github.com/dnadamson/experiment-llm-refusal.git
cd experiment-llm-refusal

docker build -t safety-finetune .
docker run --gpus all \
    -v /opt/experiment-llm-refusal/outputs:/app/outputs \
    safety-finetune \
    safety_finetune.train --config configs/full.yaml
