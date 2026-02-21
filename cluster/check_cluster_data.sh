#!/bin/bash
# Quick check for available datasets on the cluster
# Run: bash check_cluster_data.sh

echo "=== Checking for ImageNet Validation Set ==="
for dir in \
    "/scr-ssd/imagenet/val" \
    "/scr-ssd/imagenet_val" \
    "/scr/imagenet/val" \
    "/u/scr/nlp/data/imagenet/val" \
    "/nlp/scr/nlp/data/imagenet/val" \
    "/juice/scr/nlp/data/ImageNet/ILSVRC2012_img_val_synset" \
    "/nlp/scr/$(whoami)/imagenet_val"; do
    if [ -d "${dir}" ]; then
        n_files=$(find "${dir}" -maxdepth 2 -name "*.JPEG" -o -name "*.jpeg" -o -name "*.jpg" 2>/dev/null | head -100 | wc -l)
        n_dirs=$(ls -d "${dir}"/*/ 2>/dev/null | head -5)
        echo "  FOUND: ${dir}  (~${n_files}+ images)"
        if [ -n "${n_dirs}" ]; then
            echo "    Structure: organized in folders (synset-based)"
            echo "    Sample: $(ls -d ${dir}/*/ 2>/dev/null | head -3)"
        else
            echo "    Structure: flat directory"
        fi
    fi
done

echo ""
echo "=== Checking for ObjectNet ==="
for dir in \
    "/scr-ssd/objectnet" \
    "/scr/objectnet" \
    "/nlp/scr/$(whoami)/objectnet" \
    "/nlp/scr/nlp/data/objectnet"; do
    if [ -d "${dir}" ]; then
        echo "  FOUND: ${dir}"
    fi
done

echo ""
echo "=== GPU Available ==="
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "  nvidia-smi failed"
else
    echo "  No nvidia-smi (run on a GPU node)"
fi

echo ""
echo "=== Disk Space ==="
df -h /nlp/scr/$(whoami) 2>/dev/null || echo "  /nlp/scr not available"
df -h /scr-ssd 2>/dev/null || echo "  /scr-ssd not available"
