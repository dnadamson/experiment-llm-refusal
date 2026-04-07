#!/bin/bash
# Create a GCP spot VM with L4 GPU for training.
# Usage: GCP_PROJECT=my-project bash scripts/gcp/create-vm.sh

set -euo pipefail

PROJECT="${GCP_PROJECT:?Set GCP_PROJECT env var}"
ZONE="${GCP_ZONE:-us-west1-b}"
INSTANCE_NAME="safety-finetune-${USER}"

gcloud compute instances create "$INSTANCE_NAME" \
    --project="$PROJECT" \
    --zone="$ZONE" \
    --machine-type="g2-standard-8" \
    --accelerator="type=nvidia-l4,count=1" \
    --provisioning-model=SPOT \
    --instance-termination-action=STOP \
    --boot-disk-size=200GB \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --scopes=cloud-platform

echo ""
echo "VM created: $INSTANCE_NAME in $ZONE"
echo ""
echo "Next steps:"
echo "  1. Set up the VM:"
echo "     gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command='bash -s' < scripts/gcp/setup-vm.sh"
echo ""
echo "  2. SSH in and run training:"
echo "     gcloud compute ssh $INSTANCE_NAME --zone=$ZONE"
echo ""
echo "  3. Monitor (from your Mac, separate terminal):"
echo "     gcloud compute ssh $INSTANCE_NAME --zone=$ZONE -- -L 6006:localhost:6006"
echo ""
echo "  4. When done, stop billing:"
echo "     gcloud compute instances stop $INSTANCE_NAME --zone=$ZONE"
