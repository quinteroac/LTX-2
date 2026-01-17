#!/bin/bash

# Script to launch LTX-2 Local Training with Podman

# Default values
IMAGE_NAME="ltx2-local-trainer"
MODELS_HOST_DIR="./models"       # Path to models on host
INPUT_HOST_DIR="./training_data" # Path to training data (must contain 'videos' folder)
OUTPUT_HOST_DIR="./outputs"      # Path for outputs
CONFIG_HOST_FILE="./train_config.yaml" # Path to your local config file

# Args defaults
BUCKETS="960x544x49"
GPUS="all" # For nvidia-container-toolkit

echo "Buildin image (if needed)..."
podman build -t $IMAGE_NAME -f local/Containerfile .

echo "Ensuring output directory exists..."
mkdir -p $OUTPUT_HOST_DIR

echo "Starting training container..."
# Note: --device nvidia.com/gpu=all or --gpus all depends on setup. Assuming standard nvidia support.
# We mount the config file to /app/config.yaml inside container for simplicity

podman run --rm -it \
    --device nvidia.com/gpu=all \
    --security-opt=label=disable \
    -v "$MODELS_HOST_DIR":/models \
    -v "$INPUT_HOST_DIR":/data/input \
    -v "$OUTPUT_HOST_DIR":/data/output \
    -v "$CONFIG_HOST_FILE":/app/config.yaml \
    $IMAGE_NAME \
    --resolution-buckets "$BUCKETS" \
    --config-file "/app/config.yaml" \
    "$@"
