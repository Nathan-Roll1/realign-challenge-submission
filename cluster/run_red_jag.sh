#!/bin/bash
# =============================================================================
# Red Team Pipeline — Self-Contained Cluster Job (Jagupard Node)
# =============================================================================
#
# This job is FULLY SELF-CONTAINED: it acquires its own data, runs the
# pipeline, saves results, and cleans up all temporary data.
#
# Data acquisition priority:
#   1. Check known cluster paths for ImageNet val
#   2. If not found, download only dog breed images from HuggingFace (~800MB)
#   3. Clean up all downloaded data when done
#
# ---- SUBMIT FROM sc ----
#
#   nlprun -q jag -g 1 -r 80G -c 16 -p standard \
#       'bash /nlp/scr/USER/realign/run_red_jag.sh'
#
# ---- DEPLOY FROM LOCAL MACHINE ----
#
#   scp /Users/USER/Documents/realign_challenge/red_team_pipeline.py \
#       /Users/USER/Documents/realign_challenge/run_red_jag.sh \
#       /Users/USER/Documents/realign_challenge/requirements_gpu.txt \
#       /Users/USER/Documents/realign_challenge/configs_blue_team_model_registry.txt \
#       USER@CLUSTER_ADDRESS:/nlp/scr/USER/realign/
#
# ---- CONFIGURATION ----

USERNAME="USER"
PROJECT_DIR="/nlp/scr/${USERNAME}/realign"
CONDA_DIR="/nlp/scr/${USERNAME}/miniconda3"
ENV_NAME="realign"

HF_TOKEN="hf_YOUR_TOKEN_HERE"
HF_REPO="USER/red-team-dogs-v1"

# Strategy: "dogs" (superclass) or "none" (all images)
SUPERCLASS="dogs"

# SA iterations: 50K=fast/20min, 100K=recommended/40min, 200K=thorough/80min
SA_ITERATIONS=100000

# Number of proxy models (max 12, fewer = faster)
N_PROXY_MODELS=12

# Number of candidates for Gram-based optimization
N_CANDIDATES=5000

# Phase to start from: all, extract, score, select, validate, submit
PHASE="all"

# =============================================================================

set -euo pipefail

echo "============================================================"
echo " Red Team Pipeline — Self-Contained Job"
echo " Started: $(date)"
echo " Host:    $(hostname)"
echo " User:    $(whoami)"
echo " Strategy: superclass=${SUPERCLASS}"
echo "============================================================"

# ---------------------------------------------------------------------------
# 1. GPU diagnostics
# ---------------------------------------------------------------------------
echo ""
echo "=== GPU Info ==="
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    echo ""
    nvidia-smi
else
    echo "WARNING: nvidia-smi not found"
fi

# ---------------------------------------------------------------------------
# 2. Set up conda environment
# ---------------------------------------------------------------------------
echo ""
echo "=== Setting Up Conda Environment ==="

if [ -f "${CONDA_DIR}/etc/profile.d/conda.sh" ]; then
    source "${CONDA_DIR}/etc/profile.d/conda.sh"
elif [ -f "${HOME}/.bashrc" ]; then
    source "${HOME}/.bashrc"
fi

if ! conda env list | grep -q "^${ENV_NAME} "; then
    echo "Creating conda environment '${ENV_NAME}'..."
    conda create -n "${ENV_NAME}" python=3.11 -y
fi

conda activate "${ENV_NAME}"
echo "Python: $(which python) ($(python --version))"

# Redirect caches to scratch space (sailhome quota is tiny)
export HF_HOME="/nlp/scr/${USERNAME}/.cache/huggingface"
export TORCH_HOME="/nlp/scr/${USERNAME}/.cache/torch"
export XDG_CACHE_HOME="/nlp/scr/${USERNAME}/.cache"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export HF_TOKEN="${HF_TOKEN}"
mkdir -p "${HF_HOME}/hub" "${TORCH_HOME}/hub"

echo "Installing dependencies..."
pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu121 2>/dev/null || \
    pip install -q torch torchvision
pip install -q timm>=1.0.0 scipy numpy Pillow tqdm huggingface_hub datasets

python -c "
import torch
print(f'PyTorch {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    props = torch.cuda.get_device_properties(0)
    vram = getattr(props, 'total_memory', None) or getattr(props, 'total_mem', 0)
    print(f'VRAM: {vram / 1e9:.1f} GB')
import timm; print(f'timm {timm.__version__}')
"

# ---------------------------------------------------------------------------
# 3. Stage to local SSD
# ---------------------------------------------------------------------------
echo ""
echo "=== Staging to Local SSD ==="

if [ -d "/scr-ssd" ]; then
    LOCAL_DIR="/scr-ssd/${USERNAME}_redteam_$$"
elif [ -d "/scr" ]; then
    LOCAL_DIR="/scr/${USERNAME}_redteam_$$"
else
    LOCAL_DIR="/tmp/${USERNAME}_redteam_$$"
fi

mkdir -p "${LOCAL_DIR}"
echo "Local working directory: ${LOCAL_DIR}"

# Copy project files
cp "${PROJECT_DIR}/red_team_pipeline.py" "${LOCAL_DIR}/"
cp "${PROJECT_DIR}/configs_blue_team_model_registry.txt" "${LOCAL_DIR}/"

# Copy cached embeddings if they exist (for resume)
if [ -d "${PROJECT_DIR}/red_team_embeddings" ]; then
    echo "Copying cached embeddings..."
    cp -r "${PROJECT_DIR}/red_team_embeddings" "${LOCAL_DIR}/"
    N_CACHED=$(ls "${LOCAL_DIR}/red_team_embeddings/"*.npy 2>/dev/null | wc -l || echo 0)
    echo "  ${N_CACHED} cached embedding files found"
fi

# Copy cached work directory (scores, candidates, etc.)
if [ -d "${PROJECT_DIR}/red_team_work" ]; then
    echo "Copying cached work files..."
    cp -r "${PROJECT_DIR}/red_team_work" "${LOCAL_DIR}/"
fi

# ---------------------------------------------------------------------------
# 4. Locate or download image data (SELF-CONTAINED)
# ---------------------------------------------------------------------------
echo ""
echo "=== Locating Image Data ==="

IMAGENET_VAL_DIR=""
DOWNLOADED_DATA=""

# Priority 1: Check known cluster paths
for candidate in \
    "/scr-ssd/imagenet/val" \
    "/scr-ssd/imagenet_val" \
    "/scr/imagenet/val" \
    "/u/scr/nlp/data/imagenet/val" \
    "/nlp/scr/nlp/data/imagenet/val" \
    "/nlp/scr/${USERNAME}/imagenet_val" \
    "/juice/scr/nlp/data/ImageNet/ILSVRC2012_img_val_synset"; do
    if [ -d "${candidate}" ]; then
        n_files=$(find "${candidate}" -maxdepth 2 \( -name "*.JPEG" -o -name "*.jpeg" \) 2>/dev/null | head -100 | wc -l)
        if [ "${n_files}" -gt 50 ]; then
            echo "Found ImageNet-val at: ${candidate} (~${n_files}+ images)"
            IMAGENET_VAL_DIR="${candidate}"
            break
        fi
    fi
done

# Priority 2: Check if we previously downloaded dog images
if [ -z "${IMAGENET_VAL_DIR}" ] && [ -d "${PROJECT_DIR}/downloaded_imagenet_dogs" ]; then
    n_files=$(find "${PROJECT_DIR}/downloaded_imagenet_dogs" -name "*.JPEG" 2>/dev/null | head -100 | wc -l)
    if [ "${n_files}" -gt 50 ]; then
        echo "Found previously downloaded dog images: ${PROJECT_DIR}/downloaded_imagenet_dogs"
        IMAGENET_VAL_DIR="${PROJECT_DIR}/downloaded_imagenet_dogs"
    fi
fi

# Priority 3: Will auto-download via pipeline (HuggingFace streaming, ~800MB)
if [ -z "${IMAGENET_VAL_DIR}" ]; then
    echo "No local ImageNet found. Will auto-download dog breed images from HuggingFace."
    echo "(Only downloads ~6,250 dog images via streaming, not the full 6.7GB dataset)"
    DOWNLOADED_DATA="${LOCAL_DIR}/downloaded_imagenet_dogs"
fi

cd "${LOCAL_DIR}"
echo "Working in: $(pwd)"
df -h . 2>/dev/null || true

# ---------------------------------------------------------------------------
# 5. Run the Red Team pipeline
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo " Running Red Team Pipeline"
echo " Strategy: superclass=${SUPERCLASS}"
echo " Time: $(date)"
echo "============================================================"

CMD="python red_team_pipeline.py"

# Image source flags
if [ -n "${IMAGENET_VAL_DIR}" ]; then
    CMD="${CMD} --imagenet-val-dir ${IMAGENET_VAL_DIR}"
else
    CMD="${CMD} --auto-download"
    CMD="${CMD} --download-dir ${DOWNLOADED_DATA}"
fi

CMD="${CMD} --superclass ${SUPERCLASS}"
CMD="${CMD} --n-proxy-models ${N_PROXY_MODELS}"
CMD="${CMD} --n-candidates ${N_CANDIDATES}"
CMD="${CMD} --sa-iterations ${SA_ITERATIONS}"
CMD="${CMD} --phase ${PHASE}"
CMD="${CMD} --output red_team_submission.json"

if [ -n "${HF_TOKEN}" ]; then
    CMD="${CMD} --hf-token ${HF_TOKEN}"
fi
if [ -n "${HF_REPO}" ]; then
    CMD="${CMD} --hf-repo ${HF_REPO}"
fi

echo "Command: ${CMD}"
echo ""

SECONDS=0
eval "${CMD}"
PIPELINE_EXIT=$?
ELAPSED=${SECONDS}

echo ""
echo "Pipeline finished in $((ELAPSED / 60))m $((ELAPSED % 60))s (exit code: ${PIPELINE_EXIT})"

# ---------------------------------------------------------------------------
# 6. Copy results back to persistent storage
# ---------------------------------------------------------------------------
echo ""
echo "=== Saving Results to Persistent Storage ==="

# Embeddings (for resume)
if [ -d "red_team_embeddings" ]; then
    echo "Copying embeddings back..."
    mkdir -p "${PROJECT_DIR}/red_team_embeddings"
    rsync -a red_team_embeddings/ "${PROJECT_DIR}/red_team_embeddings/"
    N_TOTAL=$(ls "${PROJECT_DIR}/red_team_embeddings/"*.npy 2>/dev/null | wc -l || echo 0)
    echo "  ${N_TOTAL} embedding files in persistent storage"
fi

# Work directory (scores, candidates, selections)
if [ -d "red_team_work" ]; then
    mkdir -p "${PROJECT_DIR}/red_team_work"
    rsync -a red_team_work/ "${PROJECT_DIR}/red_team_work/"
    echo "  Work directory saved"
fi

# Submission JSON
if [ -f "red_team_submission.json" ]; then
    cp red_team_submission.json "${PROJECT_DIR}/"
    echo "  Submission JSON saved"
    echo ""
    echo "=== SUBMISSION ==="
    python -c "
import json
d = json.load(open('red_team_submission.json'))
meta = d.get('metadata', {})
print(f'  Images: {meta.get(\"n_images\", len(d.get(\"images\", [])))}')
print(f'  Predicted CKA:   {meta.get(\"predicted_mean_cka\", \"N/A\")}')
print(f'  Predicted Score:  {meta.get(\"predicted_score\", \"N/A\")}')
imgs = d.get('images', [])
ds_counts = {}
for img in imgs:
    ds = img.get('dataset_name', 'unknown')
    ds_counts[ds] = ds_counts.get(ds, 0) + 1
for ds, c in sorted(ds_counts.items()):
    print(f'  {ds}: {c} images')
"
fi

# ---------------------------------------------------------------------------
# 7. Cleanup — remove ALL temporary/downloaded data
# ---------------------------------------------------------------------------
echo ""
echo "=== Cleaning Up ==="

# Remove downloaded images (they're only needed for embedding extraction)
if [ -n "${DOWNLOADED_DATA}" ] && [ -d "${DOWNLOADED_DATA}" ]; then
    SIZE=$(du -sh "${DOWNLOADED_DATA}" 2>/dev/null | cut -f1)
    rm -rf "${DOWNLOADED_DATA}"
    echo "  Removed downloaded images: ${DOWNLOADED_DATA} (${SIZE})"
fi

# Remove local SSD staging directory
cd /tmp
rm -rf "${LOCAL_DIR}"
echo "  Removed local staging: ${LOCAL_DIR}"

# Optionally clean up timm/HF model cache to free disk
CACHE_SIZE=$(du -sh "${HF_HOME}/hub/models--timm--*" 2>/dev/null | tail -1 | cut -f1 || echo "0")
echo "  HF model cache: ${CACHE_SIZE} (keeping for future runs)"

echo ""
echo "============================================================"
echo " Red Team Pipeline Complete!"
echo " Finished: $(date)"
echo " Duration: $((ELAPSED / 60))m $((ELAPSED % 60))s"
echo " Host:     $(hostname)"
echo " Strategy: superclass=${SUPERCLASS}"
echo "============================================================"

if [ -f "${PROJECT_DIR}/red_team_submission.json" ]; then
    echo ""
    echo "Results saved to: ${PROJECT_DIR}/"
    echo "  red_team_submission.json  — submission payload"
    echo "  red_team_embeddings/      — cached embeddings (for re-runs)"
    echo "  red_team_work/            — scores, candidates, selections"
fi

if [ -n "${HF_REPO}" ]; then
    echo ""
    echo "HuggingFace dataset: https://huggingface.co/datasets/${HF_REPO}"
    echo ""
    echo "To submit on the Re-Align Challenge:"
    echo "  1. Go to the HuggingFace Space"
    echo "  2. Click Red Team tab"
    echo "  3. Paste: ${HF_REPO}"
    echo "  4. Enter your HF access token"
    echo "  5. Click 'Generate JSON' then 'Submit'"
fi

exit ${PIPELINE_EXIT}
