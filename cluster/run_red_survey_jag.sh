#!/bin/bash
# =============================================================================
# Red Team SURVEY — Superclass Divergence Exploration (Jagupard Node)
# =============================================================================
#
# Tests ~14 ImageNet superclass categories × 20 diverse models to find which
# image category produces the most representational divergence.
#
# Self-contained: downloads ImageNet val if not found on cluster.
#
# ---- SUBMIT ----
#
#   nlprun -q jag -g 1 -r 80G -c 16 -p standard \
#       'bash /nlp/scr/USER/realign/run_red_survey_jag.sh'
#
# ---- DEPLOY ----
#
#   scp /Users/USER/Documents/realign_challenge/red_team_survey.py \
#       /Users/USER/Documents/realign_challenge/run_red_survey_jag.sh \
#       /Users/USER/Documents/realign_challenge/configs_blue_team_model_registry.txt \
#       USER@CLUSTER_ADDRESS:/nlp/scr/USER/realign/
#
# ---- CONFIGURATION ----

USERNAME="USER"
PROJECT_DIR="/nlp/scr/${USERNAME}/realign"
CONDA_DIR="/nlp/scr/${USERNAME}/miniconda3"
ENV_NAME="realign"
HF_TOKEN="hf_YOUR_TOKEN_HERE"

# Max images per category (1000 = full survey, 500 = faster)
N_IMAGES=1000

# Only run specific categories (empty = all)
# CATEGORIES="dogs snakes insects primates birds"
CATEGORIES=""

# =============================================================================

set -euo pipefail

echo "============================================================"
echo " Red Team Survey — Superclass Divergence Exploration"
echo " Started: $(date)"
echo " Host:    $(hostname)"
echo "============================================================"

# --- GPU ---
echo ""
echo "=== GPU ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "No GPU"

# --- Conda ---
echo ""
echo "=== Environment ==="
if [ -f "${CONDA_DIR}/etc/profile.d/conda.sh" ]; then
    source "${CONDA_DIR}/etc/profile.d/conda.sh"
fi
if ! conda env list | grep -q "^${ENV_NAME} "; then
    conda create -n "${ENV_NAME}" python=3.11 -y
fi
conda activate "${ENV_NAME}"
echo "Python: $(python --version)"

export HF_HOME="/nlp/scr/${USERNAME}/.cache/huggingface"
export TORCH_HOME="/nlp/scr/${USERNAME}/.cache/torch"
export XDG_CACHE_HOME="/nlp/scr/${USERNAME}/.cache"
export HF_TOKEN="${HF_TOKEN}"
mkdir -p "${HF_HOME}/hub" "${TORCH_HOME}/hub"

pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu121 2>/dev/null || \
    pip install -q torch torchvision
pip install -q timm>=1.0.0 scipy numpy Pillow tqdm huggingface_hub datasets

# --- Local SSD ---
echo ""
echo "=== Staging ==="
if [ -d "/scr-ssd" ]; then
    LOCAL_DIR="/scr-ssd/${USERNAME}_survey_$$"
elif [ -d "/scr" ]; then
    LOCAL_DIR="/scr/${USERNAME}_survey_$$"
else
    LOCAL_DIR="/tmp/${USERNAME}_survey_$$"
fi
mkdir -p "${LOCAL_DIR}"
echo "Working dir: ${LOCAL_DIR}"

cp "${PROJECT_DIR}/red_team_survey.py" "${LOCAL_DIR}/"
cp "${PROJECT_DIR}/configs_blue_team_model_registry.txt" "${LOCAL_DIR}/"

# Copy cached survey results if they exist
if [ -d "${PROJECT_DIR}/red_team_survey_results" ]; then
    cp -r "${PROJECT_DIR}/red_team_survey_results" "${LOCAL_DIR}/"
    echo "Copied cached survey results"
fi

# --- Find ImageNet val ---
echo ""
echo "=== Image Data ==="
IMAGENET_VAL_DIR=""
DOWNLOADED=""

for candidate in \
    "/scr-ssd/imagenet/val" \
    "/scr-ssd/imagenet_val" \
    "/scr/imagenet/val" \
    "/u/scr/nlp/data/imagenet/val" \
    "/nlp/scr/nlp/data/imagenet/val" \
    "/nlp/scr/${USERNAME}/imagenet_val" \
    "/juice/scr/nlp/data/ImageNet/ILSVRC2012_img_val_synset"; do
    if [ -d "${candidate}" ]; then
        echo "Found ImageNet-val: ${candidate}"
        IMAGENET_VAL_DIR="${candidate}"
        break
    fi
done

# Check for previously downloaded images
if [ -z "${IMAGENET_VAL_DIR}" ] && [ -d "${PROJECT_DIR}/downloaded_imagenet_survey" ]; then
    n=$(find "${PROJECT_DIR}/downloaded_imagenet_survey" -name "*.JPEG" 2>/dev/null | head -100 | wc -l)
    if [ "$n" -gt 50 ]; then
        echo "Found previously downloaded: ${PROJECT_DIR}/downloaded_imagenet_survey"
        IMAGENET_VAL_DIR="${PROJECT_DIR}/downloaded_imagenet_survey"
    fi
fi

cd "${LOCAL_DIR}"

# --- Run Survey ---
echo ""
echo "============================================================"
echo " Running Survey"
echo " Time: $(date)"
echo "============================================================"

CMD="python red_team_survey.py"

if [ -n "${IMAGENET_VAL_DIR}" ]; then
    CMD="${CMD} --imagenet-val-dir ${IMAGENET_VAL_DIR}"
else
    CMD="${CMD} --auto-download --hf-token ${HF_TOKEN}"
    CMD="${CMD} --download-dir ${LOCAL_DIR}/downloaded_imagenet_survey"
    DOWNLOADED="${LOCAL_DIR}/downloaded_imagenet_survey"
fi

CMD="${CMD} --n-images ${N_IMAGES}"

if [ -n "${CATEGORIES}" ]; then
    CMD="${CMD} --categories ${CATEGORIES}"
fi

echo "Command: ${CMD}"
echo ""

SECONDS=0
eval "${CMD}"
EXIT_CODE=$?
ELAPSED=${SECONDS}

echo ""
echo "Survey finished in $((ELAPSED / 60))m $((ELAPSED % 60))s"

# --- Save results ---
echo ""
echo "=== Saving Results ==="
if [ -d "red_team_survey_results" ]; then
    mkdir -p "${PROJECT_DIR}/red_team_survey_results"
    rsync -a red_team_survey_results/ "${PROJECT_DIR}/red_team_survey_results/"
    echo "Survey results saved to ${PROJECT_DIR}/red_team_survey_results/"
fi

# --- Cleanup ---
echo ""
echo "=== Cleanup ==="
if [ -n "${DOWNLOADED}" ] && [ -d "${DOWNLOADED}" ]; then
    SIZE=$(du -sh "${DOWNLOADED}" 2>/dev/null | cut -f1)
    rm -rf "${DOWNLOADED}"
    echo "Removed downloaded images (${SIZE})"
fi
cd /tmp
rm -rf "${LOCAL_DIR}"
echo "Removed staging dir"

echo ""
echo "============================================================"
echo " Survey Complete!"
echo " Duration: $((ELAPSED / 60))m $((ELAPSED % 60))s"
echo " Results:  ${PROJECT_DIR}/red_team_survey_results/"
echo "============================================================"
echo ""
echo "To view results:"
echo "  python ${PROJECT_DIR}/red_team_survey.py --phase report"

exit ${EXIT_CODE}
