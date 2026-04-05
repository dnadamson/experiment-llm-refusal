#!/bin/bash
# Create a GCP spot VM with L4 GPU for training
# Adjust project, zone, and machine type as needed.

set -euo pipefail

PROJECT="${GCP_PROJECT:?Set GCP_PROJECT env var}"
ZONE="us-central1-a"
INSTANCE_NAME="safety-finetune-${USER}"

gcloud compute instances create "$INSTANCE_NAME" \
    --project="$PROJECT" \
    --zone="$ZONE" \
    --machine-type="g2-standard-8" \
    --accelerator="type=nvidia-l4,count=1" \
    --provisioning-model=SPOT \
    --instance-termination-action=STOP \
    --boot-disk-size=100GB \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --metadata-from-file startup-script=scripts/gcp/startup.sh \
    --scopes=cloud-platform

echo "VM created: $INSTANCE_NAME in $ZONE"
echo "SSH: gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --project=$PROJECT"
