#!/bin/bash
# Run this on a fresh GCP Ubuntu VM to install all dependencies.
# Usage: gcloud compute ssh safety-finetune --zone=us-west1-b --command="bash -s" < scripts/gcp/setup-vm.sh
#    or: SSH in, then: bash setup-vm.sh

set -euo pipefail

echo "=== Installing NVIDIA drivers ==="
sudo apt-get update
sudo apt-get install -y nvidia-driver-535

echo "=== Installing Docker ==="
sudo apt-get install -y docker.io

echo "=== Adding NVIDIA container toolkit repo ==="
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update

echo "=== Installing NVIDIA container toolkit ==="
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

echo "=== Adding user to docker group ==="
sudo usermod -aG docker "$USER"

echo "=== Cloning repo ==="
cd ~
if [ ! -d experiment-llm-refusal ]; then
  git clone https://github.com/dnadamson/experiment-llm-refusal.git
fi
cd experiment-llm-refusal

echo "=== Building container ==="
# newgrp docker runs the build in a subshell with the docker group active
newgrp docker <<'INNERSCRIPT'
cd ~/experiment-llm-refusal
docker build -t safety-finetune .
INNERSCRIPT

echo ""
echo "=== Setup complete ==="
echo "Log out and back in (for docker group), then run:"
echo ""
echo "  cd ~/experiment-llm-refusal"
echo "  docker run --gpus all \\"
echo "    -v \$(pwd)/outputs:/app/outputs \\"
echo "    -v \$(pwd)/hf-cache:/hf-cache \\"
echo "    -e HF_HOME=/hf-cache \\"
echo "    safety-finetune \\"
echo "    safety_finetune.train --config configs/full.yaml"
