import modal
import os
import subprocess
import shutil

LTX_REPO = "https://github.com/Lightricks/LTX-2.git"
LTX_MODEL = "https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-19b-dev.safetensors"
LTX_UPSCALE_MODEL = "https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-spatial-upscaler-x2-1.0.safetensors"
LTX_DISTILLED_LORA = "https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-19b-distilled-lora-384.safetensors"
GEMMA3_MODEL = "google/gemma-3-12b-it-qat-q4_0-unquantized"


# Definition of the Image and Modal App
import os
import subprocess
import shutil

# ... (Previous constants LTX_REPO, etc.)

# Define constant paths for the image
REPO_PATH = "/root/LTX-2"
MODELS_DIR = f"{REPO_PATH}/models"

def build_image_env():
    import os
    import subprocess
    from huggingface_hub import snapshot_download

def build_image_env():
    import os
    import subprocess
    from huggingface_hub import snapshot_download

    print("Starting image build...")

    # 1. Clone LTX_REPO
    if not os.path.exists(REPO_PATH):
        print(f"Cloning repository: {LTX_REPO}")
        subprocess.run(["git", "clone", LTX_REPO, REPO_PATH], check=True)
    
    # 2. Download Models (Individual files)
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    model_urls = [LTX_MODEL, LTX_UPSCALE_MODEL, LTX_DISTILLED_LORA]
    for url in model_urls:
        filename = url.split("/")[-1]
        dest_path = os.path.join(MODELS_DIR, filename)
        if not os.path.exists(dest_path):
            print(f"Downloading {filename}...")
            subprocess.run(["wget", "-O", dest_path, url], check=True)

    # 3. Download Gemma Model
    print(f"Downloading snapshot of {GEMMA3_MODEL}...")
    try:
        snapshot_download(
            repo_id=GEMMA3_MODEL,
            local_dir=os.path.join(MODELS_DIR, "gemma3"),
            token=os.environ.get("HF_TOKEN")
        )
    except Exception as e:
        print(f"Error downloading Gemma: {e}")

    # 4. Configure environment with uv
    print("Syncing environment with uv...")
    subprocess.run(["uv", "sync"], cwd=REPO_PATH, check=True)

# Define an image with dependencies and run the build function
image = (
    modal.Image.debian_slim()
    .apt_install("git", "wget", "ffmpeg", "libavcodec-dev", "libavformat-dev", "libswscale-dev", "libavutil-dev")
    .pip_install("huggingface_hub", "uv")
    .env({"HF_TOKEN": os.environ.get("HF_TOKEN", "")}) # Pass token optionally if exists locally for build, or use secrets
    .run_function(
        build_image_env, 
        secrets=[modal.Secret.from_name("huggingface-secret")]
    )
)

app = modal.App("ltx-2-training", image=image)

# Create a persistent volume for training data and outputs
vol = modal.Volume.from_name("training-data", create_if_missing=False)
VOLUME_PATH = "/data"

def _train_logic(
    resolution_buckets: str,
    config_content: str,
    caption_low_mem: bool = False, 
    caption_gemini: str = None,
    no_audio: bool = False
):


    import shutil
    
    # 5. Copy training data ('videos' folder) from volume
    print("Preparing training data...")
    
    source_videos = os.path.join(VOLUME_PATH, "videos")
    dest_videos_dir = os.path.join(REPO_PATH, "packages", "ltx-trainer", "videos")
    
    if os.path.exists(source_videos):
        if os.path.exists(dest_videos_dir):
            shutil.rmtree(dest_videos_dir)
        
        shutil.copytree(source_videos, dest_videos_dir)
        print(f"'videos' folder copied successfully to {dest_videos_dir}")
    else:
        raise FileNotFoundError(f"CRITICAL ERROR: 'videos' folder not found in {VOLUME_PATH}. Videos are required for training.")

    # 6. Run video captioning
    print("Starting video captioning...")
    ltx_trainer_path = os.path.join(REPO_PATH, "packages", "ltx-trainer")
    
    # script assumes 'videos/' is in CWD and outputs 'dataset.json' there
    caption_cmd = ["uv", "run", "python", "scripts/caption_videos.py", "videos/", "--output", "dataset.json"]
    
    if caption_low_mem:
        print("caption-low-mem enabled: Using --use-8bit")
        caption_cmd.append("--use-8bit")

    if caption_gemini:
        print("caption-gemini enabled: Using Gemini Flash")
        caption_cmd.extend(["--captioner-type", "gemini_flash", "--api-key", caption_gemini])

    if no_audio:
        print("no-audio enabled: Ignoring audio in videos")
        caption_cmd.append("--no-audio")

    subprocess.run(
        caption_cmd,
        cwd=ltx_trainer_path,
        check=True
    )
    subprocess.run(
        caption_cmd,
        cwd=ltx_trainer_path,
        check=True
    )
    print("Captioning completed. dataset.json generated.")

    # 7. Run dataset processing
    if not resolution_buckets:
        raise ValueError("'resolution_buckets' parameter is required (e.g., '960x544x49').")

    print(f"Starting dataset processing with buckets: {resolution_buckets}...")
    
    # Build absolute paths to models for the script
    ltx_model_path = os.path.join(MODELS_DIR, LTX_MODEL.split("/")[-1])
    gemma_path = os.path.join(MODELS_DIR, "gemma3")
    
    process_cmd = [
        "uv", "run", "python", "scripts/process_dataset.py", "dataset.json",
        "--resolution-buckets", resolution_buckets,
        "--model-path", ltx_model_path,
        "--text-encoder-path", gemma_path,
        "--output-dir", "preprocessed_data"
    ]

    if not no_audio:
        process_cmd.append("--with-audio")
    
    subprocess.run(
        process_cmd,
        cwd=ltx_trainer_path,
        check=True
    )
    print("Dataset processing completed.")

    # 8. Run Training
    print("Starting training...")
    config_path = os.path.join(ltx_trainer_path, "train.yaml")
    
    # Write config file content
    with open(config_path, "w") as f:
        f.write(config_content)
    
    subprocess.run(
        ["uv", "run", "python", "scripts/train.py", "train.yaml"],
        cwd=ltx_trainer_path,
        check=True
    )
    print("Training finished successfully.")

@app.function(volumes={VOLUME_PATH: vol}, timeout=86400, gpu="H100")
def train_h100(resolution_buckets: str, config_content: str, caption_low_mem: bool, caption_gemini: str, no_audio: bool):
    _train_logic(resolution_buckets, config_content, caption_low_mem, caption_gemini, no_audio)

@app.function(volumes={VOLUME_PATH: vol}, timeout=86400, gpu="A100")
def train_a100(resolution_buckets: str, config_content: str, caption_low_mem: bool, caption_gemini: str, no_audio: bool):
    _train_logic(resolution_buckets, config_content, caption_low_mem, caption_gemini, no_audio)

@app.function(volumes={VOLUME_PATH: vol}, timeout=86400, gpu="A10G")
def train_a10g(resolution_buckets: str, config_content: str, caption_low_mem: bool, caption_gemini: str, no_audio: bool):
    _train_logic(resolution_buckets, config_content, caption_low_mem, caption_gemini, no_audio)

@app.function(volumes={VOLUME_PATH: vol}, timeout=86400, gpu="L4")
def train_l4(resolution_buckets: str, config_content: str, caption_low_mem: bool, caption_gemini: str, no_audio: bool):
    _train_logic(resolution_buckets, config_content, caption_low_mem, caption_gemini, no_audio)

@app.function(volumes={VOLUME_PATH: vol}, timeout=86400, gpu="T4")
def train_t4(resolution_buckets: str, config_content: str, caption_low_mem: bool, caption_gemini: str, no_audio: bool):
    _train_logic(resolution_buckets, config_content, caption_low_mem, caption_gemini, no_audio)

@app.function()
def build_only():
    """
    Utility function to trigger the image build without running training.
    Usage: modal run modal/ltx2_modal_traning.py ::build_only
    """
    print("Image build verified successfully!")

@app.local_entrypoint()
def main(
    config_file: str = None,
    resolution_buckets: str = None,
    gpu_type: str = "H100",
    caption_low_mem: bool = False, 
    caption_gemini: str = None,
    no_audio: bool = False,
    build_only_flag: bool = False
):
    if build_only_flag:
        print("Running build_only mode...")
        build_only.remote()
        return

    if not config_file:
        raise ValueError("Argument '--config-file' is required for training.")
    
    if not resolution_buckets:
        raise ValueError("Argument '--resolution-buckets' is required for training.")

    with open(config_file, "r") as f:
        config_content = f.read()
    
    kwargs = {
        "resolution_buckets": resolution_buckets,
        "config_content": config_content,
        "caption_low_mem": caption_low_mem,
        "caption_gemini": caption_gemini,
        "no_audio": no_audio
    }
    
    if gpu_type.upper() == "H100":
        print("Launching training on H100...")
        train_h100.remote(**kwargs)
    elif gpu_type.upper() == "A100":
        print("Launching training on A100...")
        train_a100.remote(**kwargs)
    elif gpu_type.upper() == "A10G":
        print("Launching training on A10G...")
        train_a10g.remote(**kwargs)
    elif gpu_type.upper() == "L4":
        print("Launching training on L4...")
        train_l4.remote(**kwargs)
    elif gpu_type.upper() == "T4":
        print("Launching training on T4...")
        train_t4.remote(**kwargs)
    else:
        raise ValueError(f"GPU type '{gpu_type}' not supported. Valid options: H100, A100, A10G, L4, T4.")




