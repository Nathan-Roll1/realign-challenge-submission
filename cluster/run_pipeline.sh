#!/bin/bash
# =============================================================================
# Blue Team GPU Pipeline — End-to-End Orchestrator
# =============================================================================
#
# Usage:
#   ./run_pipeline.sh                                    # full pipeline
#   ./run_pipeline.sh --image-dir /path/to/imagenet_val  # use ImageNet-val
#   ./run_pipeline.sh --modal                            # run on Modal cloud
#
# Environment variables:
#   HF_TOKEN  — HuggingFace token for submission
#   HF_REPO   — HuggingFace repo (e.g. user/blue-team-gpu-v1)
#
# =============================================================================

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

IMAGE_DIR=""
N_IMAGES=500
USE_MODAL=false
HF_TOKEN="${HF_TOKEN:-}"
HF_REPO="${HF_REPO:-}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --image-dir)   IMAGE_DIR="$2"; shift 2 ;;
        --n-images)    N_IMAGES="$2"; shift 2 ;;
        --modal)       USE_MODAL=true; shift ;;
        --hf-token)    HF_TOKEN="$2"; shift 2 ;;
        --hf-repo)     HF_REPO="$2"; shift 2 ;;
        *)             echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN} Blue Team CKA Pipeline${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""

# Step 1: Check dependencies
echo -e "${YELLOW}Checking dependencies...${NC}"
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')" 2>/dev/null || {
    echo -e "${RED}PyTorch not found. Installing...${NC}"
    pip install -r requirements_gpu.txt
}
python3 -c "import timm; print(f'timm: {timm.__version__}')" 2>/dev/null || {
    echo -e "${RED}timm not found. Installing...${NC}"
    pip install timm>=1.0.0
}

# Step 2: Check GPU
echo ""
echo -e "${YELLOW}Checking GPU...${NC}"
python3 -c "
import torch
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    props = torch.cuda.get_device_properties(0)
    vram = getattr(props, 'total_memory', None) or getattr(props, 'total_mem', 0)
    print(f'VRAM: {vram / 1e9:.1f} GB')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print('Apple MPS available')
else:
    print('WARNING: No GPU detected — this will be very slow!')
"

# Step 3: Run pipeline
echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN} Running Pipeline${NC}"
echo -e "${GREEN}============================================${NC}"

if [ "$USE_MODAL" = true ]; then
    echo -e "${YELLOW}Deploying to Modal cloud...${NC}"
    CMD="python3 run_on_modal.py --n-images $N_IMAGES"
    [ -n "$HF_TOKEN" ] && CMD="$CMD --hf-token $HF_TOKEN"
    [ -n "$HF_REPO" ] && CMD="$CMD --hf-repo $HF_REPO"
    eval "$CMD"
else
    CMD="python3 gpu_pipeline.py --n-images $N_IMAGES"

    if [ -n "$IMAGE_DIR" ]; then
        CMD="$CMD --image-dir $IMAGE_DIR"
    else
        CMD="$CMD --auto-download"
    fi

    [ -n "$HF_TOKEN" ] && CMD="$CMD --hf-token $HF_TOKEN"
    [ -n "$HF_REPO" ] && CMD="$CMD --hf-repo $HF_REPO"

    echo "Command: $CMD"
    echo ""
    eval "$CMD"
fi

echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN} Pipeline Complete!${NC}"
echo -e "${GREEN}============================================${NC}"

if [ -f "submission_gpu.json" ]; then
    echo ""
    echo "Submission file: submission_gpu.json"
    echo "Predicted score:"
    python3 -c "import json; d=json.load(open('submission_gpu.json')); print(f'  {d.get(\"predicted_score\", \"N/A\")}')"
fi

if [ -n "$HF_REPO" ]; then
    echo ""
    echo "HuggingFace dataset: https://huggingface.co/datasets/$HF_REPO"
    echo ""
    echo "To submit:"
    echo "  1. Go to the Re-Align Challenge HuggingFace Space"
    echo "  2. Click on the Blue Team tab"
    echo "  3. Paste: $HF_REPO"
    echo "  4. Enter your HF access token"
    echo "  5. Click 'Generate JSON' then 'Submit'"
fi
