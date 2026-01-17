import argparse
import os
import subprocess
import shutil
import sys

# Constants matching standard mount points in Containerfile
MODELS_DIR = "/models"
INPUT_DATA_DIR = "/data/input"
OUTPUT_DIR = "/data/output"
REPO_PATH = "/app/LTX-2"

def main():
    parser = argparse.ArgumentParser(description="LTX-2 Local Training Pipeline")
    
    parser.add_argument("--resolution-buckets", type=str, required=True, help="Resolution buckets (e.g., '960x544x49')")
    parser.add_argument("--config-file", type=str, required=True, help="Path to config file (relative to container or absolute)")
    parser.add_argument("--caption-low-mem", action="store_true", help="Use 8-bit mode for captioning")
    parser.add_argument("--caption-gemini", type=str, help="Google API Key for Gemini Flash captioning")
    parser.add_argument("--no-audio", action="store_true", help="Disable audio processing")
    
    args = parser.parse_args()

    # --- 0. Setup Paths --
    ltx_trainer_path = os.path.join(REPO_PATH, "packages", "ltx-trainer")
    
    # --- 1. Copy Training Data ---
    print("Preparing training data...")
    source_videos = os.path.join(INPUT_DATA_DIR, "videos")
    dest_videos_dir = os.path.join(ltx_trainer_path, "videos")
    
    if os.path.exists(source_videos):
        if os.path.exists(dest_videos_dir):
            shutil.rmtree(dest_videos_dir)
        shutil.copytree(source_videos, dest_videos_dir)
        print(f"'videos' folder copied successfully to {dest_videos_dir}")
    else:
        print(f"CRITICAL ERROR: 'videos' folder not found at {source_videos}.")
        print("Please ensure you have mounted your training data volume to /data/input.")
        sys.exit(1)

    # --- 2. Captioning ---
    print("Starting video captioning...")
    
    # We run from ltx_trainer_path so videos/ is relative to CWD
    caption_cmd = ["uv", "run", "python", "scripts/caption_videos.py", "videos/", "--output", "dataset.json"]
    
    if args.caption_low_mem:
        print("caption-low-mem enabled: Using --use-8bit")
        caption_cmd.append("--use-8bit")

    if args.caption_gemini:
        print("caption-gemini enabled: Using Gemini Flash")
        caption_cmd.extend(["--captioner-type", "gemini_flash", "--api-key", args.caption_gemini])

    if args.no_audio:
        print("no-audio enabled: Ignoring audio in videos")
        caption_cmd.append("--no-audio")

    try:
        subprocess.run(caption_cmd, cwd=ltx_trainer_path, check=True)
        print("Captioning completed. dataset.json generated.")
    except subprocess.CalledProcessError as e:
        print(f"Error during captioning: {e}")
        sys.exit(1)

    # --- 3. Dataset Processing ---
    print(f"Starting dataset processing with buckets: {args.resolution_buckets}...")
    
    # Model paths in the mounted /models volume
    # Assuming standard filenames from the original script
    # LTX_MODEL -> ltx-2-19b-dev.safetensors
    # GEMMA3_MODEL -> gemma3 (directory)
    
    ltx_model_path = os.path.join(MODELS_DIR, "ltx-2-19b-dev.safetensors")
    gemma_path = os.path.join(MODELS_DIR, "gemma3")
    
    if not os.path.exists(ltx_model_path) or not os.path.exists(gemma_path):
        print(f"Error: Base models not found in {MODELS_DIR}.")
        print("Ensure 'ltx-2-19b-dev.safetensors' and 'gemma3' directory exist in the mounted models volume.")
        sys.exit(1)

    process_cmd = [
        "uv", "run", "python", "scripts/process_dataset.py", "dataset.json",
        "--resolution-buckets", args.resolution_buckets,
        "--model-path", ltx_model_path,
        "--text-encoder-path", gemma_path
    ]

    if not args.no_audio:
        process_cmd.append("--with-audio")
    
    try:
        subprocess.run(process_cmd, cwd=ltx_trainer_path, check=True)
        print("Dataset processing completed.")
    except subprocess.CalledProcessError as e:
        print(f"Error during dataset processing: {e}")
        sys.exit(1)

    # --- 4. Training ---
    print("Starting training...")
    
    # We assume usage of a config file passed by the user. 
    # If it's a path inside the container, we use it directly.
    # If the user mounted their config at runtime, passed path applies.
    # NOTE: The config file MUST have correct paths for the CONTAINER environment.
    # i.e., model_path should be /models/..., output_dir /data/output/...
    
    # To facilitate this, we could attempt to patch the config file or assume user provides a correct one.
    # Given the prompt asked for "similar script", let's assume we use the passed config directly.
    
    train_cmd = ["uv", "run", "python", "scripts/train.py", args.config_file]
    
    try:
        subprocess.run(train_cmd, cwd=ltx_trainer_path, check=True)
        print("Training finished successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error during training: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
