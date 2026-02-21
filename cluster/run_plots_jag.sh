#!/bin/bash
# =============================================================================
# Generate Real Plots — Self-Contained Cluster Job (Jagupard Node)
# =============================================================================
#
# ---- SUBMIT FROM sc ----
#
#   nlprun -q jag -g 1 -r 80G -c 16 -p standard \
#       'bash /nlp/scr/USER/realign/run_plots_jag.sh'
#
# ---- DEPLOY FROM LOCAL MACHINE ----
#
#   scp /Users/USER/Documents/realign_challenge/generate_real_plots_cluster.py \
#       /Users/USER/Documents/realign_challenge/run_plots_jag.sh \
#       USER@CLUSTER_ADDRESS:/nlp/scr/USER/realign/
#
# =============================================================================

USERNAME="USER"
PROJECT_DIR="/nlp/scr/${USERNAME}/realign"
CONDA_DIR="/nlp/scr/${USERNAME}/miniconda3"
ENV_NAME="realign"

set -euo pipefail

echo "============================================================"
echo " Plot Generation Job"
echo " Started: $(date)"
echo " Host:    $(hostname)"
echo " User:    $(whoami)"
echo "============================================================"

# 1. Set up conda environment
if [ -f "${CONDA_DIR}/etc/profile.d/conda.sh" ]; then
    source "${CONDA_DIR}/etc/profile.d/conda.sh"
elif [ -f "${HOME}/.bashrc" ]; then
    source "${HOME}/.bashrc"
fi

conda activate "${ENV_NAME}"

# Set HF Cache paths
export HF_HOME="/nlp/scr/${USERNAME}/.cache/huggingface"
export TORCH_HOME="/nlp/scr/${USERNAME}/.cache/torch"
export XDG_CACHE_HOME="/nlp/scr/${USERNAME}/.cache"

# Ensure timm and matplotlib are installed
pip install -q timm matplotlib seaborn datasets

# 2. Stage to local SSD
if [ -d "/scr-ssd" ]; then
    LOCAL_DIR="/scr-ssd/${USERNAME}_plots_$$"
elif [ -d "/scr" ]; then
    LOCAL_DIR="/scr/${USERNAME}_plots_$$"
else
    LOCAL_DIR="/tmp/${USERNAME}_plots_$$"
fi

mkdir -p "${LOCAL_DIR}"
cp "${PROJECT_DIR}/generate_real_plots_cluster.py" "${LOCAL_DIR}/"

cd "${LOCAL_DIR}"
echo "Working in: $(pwd)"

# 3. Run the plot generation
echo "Running generate_real_plots_cluster.py..."
python generate_real_plots_cluster.py
EXIT_CODE=$?

# 4. Copy results back
echo "Copying results back to persistent storage..."
cp *.png "${PROJECT_DIR}/" 2>/dev/null || true
cp *.npz "${PROJECT_DIR}/" 2>/dev/null || true

# 5. Cleanup
cd /tmp
rm -rf "${LOCAL_DIR}"

echo "============================================================"
echo " Plot Generation Complete! (Exit code: ${EXIT_CODE})"
echo "============================================================"
exit ${EXIT_CODE}
