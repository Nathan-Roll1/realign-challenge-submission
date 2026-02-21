#!/bin/bash
# Deploy Red Team pipeline to Stanford NLP cluster
#
# Usage:
#   ./deploy_red_team.sh              # deploy to /nlp/scr/USER/realign
#   ./deploy_red_team.sh --dry-run    # show what would be copied

set -euo pipefail

REMOTE_HOST="CLUSTER_ADDRESS"
REMOTE_USER="USER"
REMOTE_DIR="/nlp/scr/${REMOTE_USER}/realign"
LOCAL_DIR="/Users/USER/Documents/realign_challenge"

DRY_RUN=""
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN="echo [DRY RUN]"
fi

FILES=(
    "red_team_pipeline.py"
    "run_red_jag.sh"
    "requirements_gpu.txt"
    "configs_blue_team_model_registry.txt"
)

echo "Deploying Red Team pipeline"
echo "  From: ${LOCAL_DIR}"
echo "  To:   ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}"
echo ""

# Create remote directory
${DRY_RUN} ssh "${REMOTE_USER}@${REMOTE_HOST}" "mkdir -p ${REMOTE_DIR}" || {
    echo "SSH failed. Try:"
    echo "  ssh-copy-id ${REMOTE_USER}@${REMOTE_HOST}"
    echo ""
    echo "Or copy manually:"
    echo "  scp ${LOCAL_DIR}/red_team_pipeline.py \\"
    echo "      ${LOCAL_DIR}/run_red_jag.sh \\"
    echo "      ${LOCAL_DIR}/requirements_gpu.txt \\"
    echo "      ${LOCAL_DIR}/configs_blue_team_model_registry.txt \\"
    echo "      ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/"
    exit 1
}

# Copy files using full absolute paths
for f in "${FILES[@]}"; do
    if [ -f "${LOCAL_DIR}/${f}" ]; then
        echo "  Copying ${f}..."
        ${DRY_RUN} scp "${LOCAL_DIR}/${f}" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/${f}"
    else
        echo "  SKIP: ${f} (not found at ${LOCAL_DIR}/${f})"
    fi
done

# Make scripts executable
${DRY_RUN} ssh "${REMOTE_USER}@${REMOTE_HOST}" "chmod +x ${REMOTE_DIR}/run_red_jag.sh"

echo ""
echo "Done! To run on the cluster:"
echo ""
echo "  ssh ${REMOTE_USER}@${REMOTE_HOST}"
echo ""
echo "  nlprun -q jag -g 1 -r 80G -c 16 -p standard \\"
echo "      'bash ${REMOTE_DIR}/run_red_jag.sh'"
