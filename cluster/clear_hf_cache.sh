#!/bin/bash
# One-time cleanup: remove HF/torch model caches from sailhome
# (sailhome has a tiny quota; these should live on /nlp/scr instead)
#
# Run from the login node (sc):
#   bash clear_hf_cache.sh

set -euo pipefail

echo "=== Clearing HuggingFace / Torch caches from sailhome ==="

SAILHOME="${HOME}"
HF_CACHE="${SAILHOME}/.cache/huggingface"
TORCH_CACHE="${SAILHOME}/.cache/torch"

for DIR in "${HF_CACHE}" "${TORCH_CACHE}"; do
    if [ -d "${DIR}" ]; then
        SIZE=$(du -sh "${DIR}" 2>/dev/null | cut -f1)
        echo "Removing ${DIR} (${SIZE})..."
        rm -rf "${DIR}"
        echo "  Done."
    else
        echo "${DIR} — not found, skipping."
    fi
done

echo ""
echo "=== Disk usage after cleanup ==="
du -sh "${SAILHOME}/.cache" 2>/dev/null || echo "  ~/.cache is clean"
echo ""
quota -s 2>/dev/null || true
echo ""
echo "Done. Future runs will use /nlp/scr/ for model caches."
