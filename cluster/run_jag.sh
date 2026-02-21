#!/bin/bash
# =============================================================================
# Blue Team CKA Pipeline — Stanford NLP Cluster (Jagupard Node)
# =============================================================================
#
# This script runs the full GPU pipeline on a single jagupard node.
# It handles:  environment setup, local SSD staging, extraction, optimization,
#              and copying results back to persistent storage.
#
# ---- SUBMIT FROM sc ----
#
#   # Standard priority, 1 GPU, 80GB RAM, 16 cores:
#   nlprun -q jag -g 1 -r 80G -c 16 -p standard 'bash /nlp/scr/USER/realign/run_jag.sh'
#
#   # Low priority (won't preempt others):
#   nlprun -q jag -g 1 -r 80G -c 16 -p low 'bash /nlp/scr/USER/realign/run_jag.sh'
#
#   # Important priority (can't be preempted once started):
#   nlprun -q jag -g 1 -r 80G -c 16 -p important 'bash /nlp/scr/USER/realign/run_jag.sh'
#
#   # Exclude weak jag nodes (older GPUs with <16GB VRAM):
#   nlprun -q jag -g 1 -r 80G -c 16 -p standard \
#       -x jagupard[18-25] \
#       'bash /nlp/scr/USER/realign/run_jag.sh'
#
# ---- CONFIGURATION ----
# Edit these variables to match your setup:

USERNAME="USER"
PROJECT_DIR="/nlp/scr/${USERNAME}/realign"
CONDA_DIR="/nlp/scr/${USERNAME}/miniconda3"
ENV_NAME="realign"
N_IMAGES=1000
HF_TOKEN="hf_YOUR_TOKEN_HERE"
HF_REPO="USER/blue-team-gpu-v1"
HF_PROXY_DATASET="USER/blue-team-proxy-1000"

# =============================================================================

set -euo pipefail

echo "============================================================"
echo " Blue Team CKA Pipeline — Jagupard Node"
echo " Started: $(date)"
echo " Host:    $(hostname)"
echo " User:    $(whoami)"
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

# Check CUDA
echo ""
echo "=== CUDA Versions Available ==="
ls -d /usr/local/cuda* 2>/dev/null || echo "No CUDA found in /usr/local"

# ---------------------------------------------------------------------------
# 2. Set up conda environment
# ---------------------------------------------------------------------------
echo ""
echo "=== Setting Up Conda Environment ==="

# Source conda
if [ -f "${CONDA_DIR}/etc/profile.d/conda.sh" ]; then
    source "${CONDA_DIR}/etc/profile.d/conda.sh"
elif [ -f "${HOME}/.bashrc" ]; then
    source "${HOME}/.bashrc"
fi

# Create environment if it doesn't exist
if ! conda env list | grep -q "^${ENV_NAME} "; then
    echo "Creating conda environment '${ENV_NAME}'..."
    conda create -n "${ENV_NAME}" python=3.11 -y
fi

conda activate "${ENV_NAME}"
echo "Python: $(which python) ($(python --version))"

# Redirect model caches to scratch space (sailhome quota is tiny)
export HF_HOME="/nlp/scr/${USERNAME}/.cache/huggingface"
export TORCH_HOME="/nlp/scr/${USERNAME}/.cache/torch"
export XDG_CACHE_HOME="/nlp/scr/${USERNAME}/.cache"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
mkdir -p "${HF_HOME}/hub" "${TORCH_HOME}/hub"
echo "Cache: HF_HOME=${HF_HOME}"

# Install dependencies
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
# 3. Stage data to local SSD for speed
# ---------------------------------------------------------------------------
echo ""
echo "=== Staging Data to Local SSD ==="

# Use local SSD if available, otherwise fall back to /tmp
if [ -d "/scr-ssd" ]; then
    LOCAL_DIR="/scr-ssd/${USERNAME}_realign_$$"
elif [ -d "/scr" ]; then
    LOCAL_DIR="/scr/${USERNAME}_realign_$$"
else
    LOCAL_DIR="/tmp/${USERNAME}_realign_$$"
fi

mkdir -p "${LOCAL_DIR}"
echo "Local working directory: ${LOCAL_DIR}"

# Copy project files to local SSD
cp "${PROJECT_DIR}/gpu_pipeline.py" "${LOCAL_DIR}/"
cp "${PROJECT_DIR}/blue_team_optimizer.py" "${LOCAL_DIR}/" 2>/dev/null || true
cp "${PROJECT_DIR}/configs_blue_team_model_registry.txt" "${LOCAL_DIR}/"

# Copy cached Gram matrices if they exist (for resume)
if [ -d "${PROJECT_DIR}/gram_matrices" ]; then
    echo "Copying cached Gram matrices..."
    cp -r "${PROJECT_DIR}/gram_matrices" "${LOCAL_DIR}/"
    N_CACHED=$(ls "${LOCAL_DIR}/gram_matrices/"*.npy 2>/dev/null | wc -l)
    echo "  ${N_CACHED} cached Gram matrices found"
fi

# Copy proxy images if they exist
if [ -d "${PROJECT_DIR}/proxy_images" ]; then
    echo "Copying cached proxy images..."
    cp -r "${PROJECT_DIR}/proxy_images" "${LOCAL_DIR}/"
fi

# --- Proxy image source ---
# Priority: HF proxy dataset > local dirs > auto-download
# The HF dataset is created once locally via create_proxy_dataset.py
# and contains a stratified 50/50 ImageNet-val + ObjectNet sample (~150 MB).

IMAGE_FLAGS=""
if [ -n "${HF_PROXY_DATASET}" ]; then
    echo "Using HF proxy dataset: ${HF_PROXY_DATASET}"
    IMAGE_FLAGS="--hf-proxy-dataset ${HF_PROXY_DATASET}"
else
    # Fallback: detect local datasets on this node
    IMAGENET_VAL_DIR=""
    for candidate in \
        "/scr-ssd/imagenet/val" \
        "/scr-ssd/imagenet_val" \
        "/scr/imagenet/val" \
        "/u/scr/nlp/data/imagenet/val" \
        "/nlp/scr/nlp/data/imagenet/val" \
        "/nlp/scr/${USERNAME}/imagenet_val"; do
        if [ -d "${candidate}" ]; then
            echo "Found ImageNet-val at: ${candidate}"
            IMAGENET_VAL_DIR="${candidate}"
            break
        fi
    done

    OBJECTNET_DIR=""
    for candidate in \
        "/scr-ssd/objectnet" \
        "/scr/objectnet" \
        "/nlp/scr/${USERNAME}/objectnet" \
        "/nlp/scr/nlp/data/objectnet" \
        "/nlp/scr/${USERNAME}/objectnet/objectnet-1.0/images"; do
        if [ -d "${candidate}" ]; then
            echo "Found ObjectNet at: ${candidate}"
            OBJECTNET_DIR="${candidate}"
            break
        fi
    done

    if [ -n "${IMAGENET_VAL_DIR}" ]; then
        IMAGE_FLAGS="${IMAGE_FLAGS} --imagenet-val-dir ${IMAGENET_VAL_DIR}"
    fi
    if [ -n "${OBJECTNET_DIR}" ]; then
        IMAGE_FLAGS="${IMAGE_FLAGS} --objectnet-dir ${OBJECTNET_DIR}"
    fi
    if [ -z "${IMAGE_FLAGS}" ]; then
        IMAGE_FLAGS="--auto-download"
        echo "WARNING: No image source found. Falling back to Food-101."
    fi
fi

cd "${LOCAL_DIR}"
echo "Working in: $(pwd)"
df -h . 2>/dev/null || true

# ---------------------------------------------------------------------------
# 4. Run the pipeline
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo " Running GPU Pipeline"
echo " Time: $(date)"
echo "============================================================"

# Build the command
CMD="python gpu_pipeline.py"
CMD="${CMD} ${IMAGE_FLAGS}"
CMD="${CMD} --n-images ${N_IMAGES}"
CMD="${CMD} --gram-dir gram_matrices"
CMD="${CMD} --output submission_gpu.json"

if [ -n "${HF_TOKEN}" ] && [ -n "${HF_REPO}" ]; then
    CMD="${CMD} --hf-token ${HF_TOKEN}"
    CMD="${CMD} --hf-repo ${HF_REPO}"
fi

echo "Command: ${CMD}"
echo ""

# Run with timing
SECONDS=0
eval "${CMD}"
PIPELINE_EXIT=$?
ELAPSED=${SECONDS}

echo ""
echo "Pipeline finished in $((ELAPSED / 60))m $((ELAPSED % 60))s (exit code: ${PIPELINE_EXIT})"

# ---------------------------------------------------------------------------
# 4b. Generate strategic submissions (5 maximally informative candidates)
# ---------------------------------------------------------------------------
if [ -f "cka_matrix.npz" ]; then
    echo ""
    echo "=== Generating Strategic Submissions ==="
    STRAT_CMD="python strategic_submit.py --cka-matrix cka_matrix.npz"
    if [ -n "${HF_TOKEN}" ] && [ -n "${HF_REPO}" ]; then
        STRAT_CMD="${STRAT_CMD} --hf-token ${HF_TOKEN} --hf-repo-prefix ${HF_REPO}"
    fi
    cp "${PROJECT_DIR}/strategic_submit.py" . 2>/dev/null || true
    eval "${STRAT_CMD}" || echo "Strategic submission generation failed (non-fatal)"
fi

# ---------------------------------------------------------------------------
# 5. Copy results back to persistent storage
# ---------------------------------------------------------------------------
echo ""
echo "=== Saving Results to Persistent Storage ==="

# Always copy back Gram matrices (for resume / future runs)
if [ -d "gram_matrices" ]; then
    echo "Copying Gram matrices back..."
    mkdir -p "${PROJECT_DIR}/gram_matrices"
    cp -n gram_matrices/*.npy "${PROJECT_DIR}/gram_matrices/" 2>/dev/null || \
        rsync -a gram_matrices/ "${PROJECT_DIR}/gram_matrices/"
    N_TOTAL=$(ls "${PROJECT_DIR}/gram_matrices/"*.npy 2>/dev/null | wc -l)
    echo "  ${N_TOTAL} total Gram matrices in persistent storage"
fi

# Copy CKA matrix
if [ -f "cka_matrix.npz" ]; then
    cp cka_matrix.npz "${PROJECT_DIR}/"
    echo "  CKA matrix saved"
fi

# Copy results
if [ -d "results" ]; then
    cp -r results/ "${PROJECT_DIR}/results/"
    echo "  Results directory saved"
fi

# Copy submission
if [ -f "submission_gpu.json" ]; then
    cp submission_gpu.json "${PROJECT_DIR}/"
    echo "  Submission JSON saved"
    echo ""
    echo "=== SUBMISSION ==="
    cat submission_gpu.json
fi

# Copy strategic submissions
if [ -d "strategic_submissions" ]; then
    cp -r strategic_submissions/ "${PROJECT_DIR}/strategic_submissions/"
    echo "  Strategic submissions saved"
fi

# Copy proxy images if we downloaded them
if [ -d "proxy_images" ] && [ ! -d "${PROJECT_DIR}/proxy_images" ]; then
    echo "Copying proxy images back for future runs..."
    cp -r proxy_images/ "${PROJECT_DIR}/proxy_images/"
fi

# ---------------------------------------------------------------------------
# 6. Cleanup local SSD
# ---------------------------------------------------------------------------
echo ""
echo "=== Cleaning Up Local SSD ==="
cd /tmp
rm -rf "${LOCAL_DIR}"
echo "  Cleaned: ${LOCAL_DIR}"

# ---------------------------------------------------------------------------
# 7. Summary
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo " Pipeline Complete!"
echo " Finished: $(date)"
echo " Duration: $((ELAPSED / 60))m $((ELAPSED % 60))s"
echo " Host:     $(hostname)"
echo "============================================================"

if [ -f "${PROJECT_DIR}/submission_gpu.json" ]; then
    echo ""
    echo "Results saved to: ${PROJECT_DIR}/"
    echo "  submission_gpu.json  — submission payload"
    echo "  gram_matrices/       — cached Gram matrices (for re-runs)"
    echo "  cka_matrix.npz       — full CKA matrix"
    echo ""
    SCORE=$(python -c "import json; d=json.load(open('${PROJECT_DIR}/submission_gpu.json')); print(d.get('predicted_score', 'N/A'))" 2>/dev/null || echo "N/A")
    echo "Predicted CKA score: ${SCORE}"
fi

if [ -n "${HF_REPO}" ]; then
    echo ""
    echo "HuggingFace dataset: https://huggingface.co/datasets/${HF_REPO}"
    echo ""
    echo "To submit on the Re-Align Challenge:"
    echo "  1. Go to the HuggingFace Space"
    echo "  2. Click Blue Team tab"
    echo "  3. Paste: ${HF_REPO}"
    echo "  4. Enter your HF access token"
    echo "  5. Click 'Generate JSON' then 'Submit'"
fi

exit ${PIPELINE_EXIT}
