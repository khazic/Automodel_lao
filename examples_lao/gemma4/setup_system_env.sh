#!/bin/bash
# Install required Python packages into the system Python environment.
# Run once on a fresh server before launching any training job.
#
# Usage:
#   bash examples_lao/gemma4/setup_system_env.sh

set -euo pipefail

PIP="pip install --break-system-packages"

echo "==> Fixing blinker (distutils-installed, must be force-replaced)"
$PIP --ignore-installed blinker

echo "==> Installing training dependencies"
$PIP \
    megatron-fsdp \
    torchao \
    torchdata \
    "transformers==5.5.0" \
    datasets \
    wandb \
    mlflow \
    flashoptim \
    pyyaml \
    tiktoken \
    "mistral-common[image,audio,hf-hub,sentencepiece]" \
    "opencv-python-headless==4.10.0.84"

echo "==> Done. All packages installed."
