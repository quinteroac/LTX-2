# LTX-2 Local Training Guide (Podman)

This guide explains how to run LTX-2 training locally using Podman, with `uv` for dependency management.

## Prerequisites

1.  **Podman** installed with NVIDIA Container Toolkit support (CDI).
2.  **Model Checkpoints**: You must have the LTX-2 models downloaded locally.
3.  **Training Data**: A folder containing your input videos.

## Directory Structure

Ensure your file structure looks similar to this:

```
/path/to/workspace/
├── models/
│   ├── ltx-2-19b-dev.safetensors
│   └── gemma3/ (directory with Gemma model files)
├── training_data/
│   └── videos/ (your .mp4 files)
├── outputs/ (will be created automatically)
└── my_config.yaml (your modified config file)
```

## Configuration File

You need a training configuration file (`.yaml`). You can use the templates in `packages/ltx-trainer/configs/` as a base.

**IMPORTANT**: Adjust paths in your config file to match the container's mount points:

*   `model_path`: `/models/ltx-2-19b-dev.safetensors`
*   `text_encoder_path`: `/models/gemma3`
*   `preprocessed_data_root`: `/app/LTX-2/packages/ltx-trainer/preprocessed_data` (Default output of pipeline)
*   `output_dir`: `/data/output/my_experiment_name`

## Quick Start with `run_podman.sh`

We provided a helper script `local/run_podman.sh`.

```bash
# 1. Edit the script to point to your local directories or call it with proper environment variables.
# Example usage:

MODELS_HOST_DIR="/path/to/my/models" \
INPUT_HOST_DIR="/path/to/my/data" \
CONFIG_HOST_FILE="./packages/ltx-trainer/configs/ltx2_av_lora_modal.yaml" \
./local/run_podman.sh \
    --caption-low-mem \
    --no-audio
```

## Manual Execution

1.  **Build the Image**:
    ```bash
    podman build -t ltx2-local-trainer -f local/Containerfile .
    ```

2.  **Run the Container**:
    ```bash
    podman run --rm -it \
        --device nvidia.com/gpu=all \
        --security-opt=label=disable \
        -v /path/to/local/models:/models \
        -v /path/to/local/data:/data/input \
        -v /path/to/local/outputs:/data/output \
        -v /path/to/local/config.yaml:/app/config.yaml \
        ltx2-local-trainer \
        --resolution-buckets "960x544x49" \
        --config-file "/app/config.yaml" \
        --caption-low-mem
    ```

## Parameters

The entrypoint script accepts similar parameters to the Modal version:

*   `--resolution-buckets`: (Required) e.g., "960x544x49".
*   `--config-file`: (Required) Path to config (inside container).
*   `--caption-low-mem`: Enable 8-bit captioning.
*   `--caption-gemini <KEY>`: Use Gemini API.
*   `--no-audio`: Disable audio processing.
