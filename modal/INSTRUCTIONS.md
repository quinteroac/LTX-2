# LTX-2 Training Guide on Modal

This guide describes the steps required to configure, execute, and manage LTX-2 training using Modal.com.

## 1. Before Training (Preparation)

### Initial Modal Configuration
Ensure you have `modal` installed and configured locally:
```bash
pip install modal
modal setup
```

### Required Secrets
You must configure your Hugging Face token as a secret in Modal so the script can download the protected base models (LTX-2).
```bash
modal secret create huggingface-secret HF_TOKEN=your_huggingface_token_here
```

### Data Volume Management
The script uses a persistent volume named `training-data` acting as shared storage.

**1. Create the volume (if not exists):**
```bash
modal volume create training-data
```

**2. Upload your Training Videos:**
You must upload your videos to a folder named `videos/` within the volume.
Assuming you have your local videos in `./mymovies`:
```bash
for f mymovies/*; do modal volume put training-data "$f" videos/; done
```

**3. (Optional) Upload Reference Videos for Video-to-Video:**
If you plan to train using IC-LoRA (Video-to-Video), upload reference videos to `reference_videos/`:
```bash
for f myreferences/*; do modal volume put training-data "$f" reference_videos/; done
```

### Configuration Files
We have prepared Modal-optimized templates in `packages/ltx-trainer/configs/`. Choose the one that fits your needs:

- **Standard**: `ltx2_av_lora_modal.yaml`
- **Low Consumption (Low VRAM)**: `ltx2_av_lora_low_vram_modal.yaml`
- **Video-to-Video**: `ltx2_v2v_ic_lora_modal.yaml`

### Pre-building the Image (Optional)
If you want to prepare the environment (download models, install dependencies) without starting a training run, you can use the `--build-only-flag`:
```bash
modal run modal/ltx2_modal_traning.py --build-only-flag
```
This is useful to cache the image and ensure everything is ready before launching a potentially expensive training job.

---

## 2. During Training (Execution)

The script `ltx2_modal_traning.py` orchestrates the entire process: model downloading (at build time), captioning, dataset processing, and final training.

### Base Command
```bash
modal run modal/ltx2_modal_traning.py \
    --config-file <LOCAL_PATH_TO_YAML> \
    --resolution-buckets <BUCKETS> \
    [ADDITIONAL OPTIONS]
```

### Parameters
| Parameter | Required | Description | Example |
|-----------|-----------|-------------|---------|
| `--config-file` | YES | Local path to the `.yaml` configuration file. | `packages/ltx-trainer/configs/ltx2_av_lora_modal.yaml` |
| `--resolution-buckets` | YES | Resolution and frames for buckets. | `"960x544x49"` |
| `--gpu-type` | NO | GPU Type (`H100`, `A100`, `A10G`, `L4`, `T4`). Default: `H100`. | `--gpu-type A100` |
| `--caption-low-mem` | NO | Use 8-bit mode for captioning (less RAM). | `--caption-low-mem` |
| `--caption-gemini` | NO | Google API Key to use Gemini Flash for captioning. | `--caption-gemini AIzaSy...` |
| `--no-audio` | NO | Disables audio processing. | `--no-audio` |

### Common Examples

**1. High-End Training (H100, Audio+Video):**
```bash
modal run modal/ltx2_modal_traning.py \
    --config-file packages/ltx-trainer/configs/ltx2_av_lora_modal.yaml \
    --resolution-buckets "960x544x49"
```

**2. Economical Training (A100, Low VRAM Config):**
```bash
modal run modal/ltx2_modal_traning.py \
    --config-file packages/ltx-trainer/configs/ltx2_av_lora_low_vram_modal.yaml \
    --resolution-buckets "960x544x49" \
    --gpu-type "A100"
```

**3. Advanced Captioning with Gemini:**
```bash
modal run modal/ltx2_modal_traning.py \
    --config-file packages/ltx-trainer/configs/ltx2_av_lora_modal.yaml \
    --resolution-buckets "960x544x49" \
    --caption-gemini "YOUR_GOOGLE_API_KEY"
```

---

## 3. After Training (Results)

Outputs are automatically saved to the `training-data` volume. By default, configurations point to `/data/outputs/...`.

### Verify Results
List the files generated in the cloud:
```bash
modal volume ls training-data outputs/ltx2_av_lora
```

### Download Checkpoints and Videos
Download what you need to your local machine:

```bash
# Download the entire output folder
modal volume get training-data outputs/ltx2_av_lora ./my_local_results
```

### Cleanup (Optional)
If you want to free up space in the cloud volume:
```bash
modal volume rm training-data outputs/ltx2_av_lora -r
```
