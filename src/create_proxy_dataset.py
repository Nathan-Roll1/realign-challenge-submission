#!/usr/bin/env python3
"""
One-time local script: create a private HuggingFace dataset with 1000
stratified proxy images matching the Blue Team evaluation distribution.

The evaluation universe is ~50% ImageNet-val + ~50% ObjectNet.
This samples proportionally, packages as a private HF dataset, and pushes.
On the cluster, the pipeline just does `load_dataset(...)` (~150 MB).

Prerequisites:
  pip install datasets Pillow huggingface_hub numpy

  ImageNet-val requires accepting the license on HuggingFace:
    https://huggingface.co/datasets/ILSVRC/imagenet-1k
  (Click "Access repository" — one-time, takes 10 seconds)

Usage:
  python create_proxy_dataset.py

Then on the cluster:
  python gpu_pipeline.py --hf-proxy-dataset USER/blue-team-proxy-1000 ...
"""
from __future__ import annotations

import os
import sys
from collections import Counter

import numpy as np

# Force unbuffered output for progress visibility
os.environ["PYTHONUNBUFFERED"] = "1"

# ---- Configuration ----
HF_TOKEN = "hf_YOUR_TOKEN_HERE"
HF_REPO = "USER/blue-team-proxy-1000"

N_TOTAL = 1000
# Catalog distribution: imagenet_val=50000 (49.9%), objectnet=50273 (50.1%)
N_IMAGENET = 499
N_OBJECTNET = 501

IMAGENET_VAL_SIZE = 50_000
OBJECTNET_SIZE = 50_273


def systematic_sample(dataset_iter, total_est: int, n_sample: int,
                      image_key: str = "image", seed: int = 42,
                      timeout_sec: int = 120):
    """Sample n_sample images at even intervals from a streaming dataset.

    Uses systematic sampling (every K-th item with random offset) so we get
    good class diversity while streaming — only the sampled images are kept
    in memory.  Aborts if no progress within timeout_sec.
    """
    import time
    from PIL import Image

    rng = np.random.RandomState(seed)
    step = max(1, total_est // n_sample)
    offset = rng.randint(0, step)

    images = []
    last_progress = time.time()
    for i, row in enumerate(dataset_iter):
        # Abort if no image collected for too long (dataset might be huge/stuck)
        if time.time() - last_progress > timeout_sec:
            print(f"    Timeout after {timeout_sec}s with {len(images)} images "
                  f"(iterated {i} rows)")
            break
        if (i - offset) % step == 0:
            img = None
            for key in (image_key, "image", "jpg", "png", "webp"):
                if key in row:
                    img = row[key]
                    break
            if img is None:
                continue
            if not isinstance(img, Image.Image):
                continue
            images.append(img.convert("RGB"))
            last_progress = time.time()
            if len(images) % 100 == 0:
                print(f"    {len(images)}/{n_sample} collected...")
            if len(images) >= n_sample:
                break
    return images


def load_imagenet_val(n_sample: int, token: str):
    """Load ImageNet validation images via HF streaming."""
    from datasets import load_dataset

    name = "ILSVRC/imagenet-1k"
    try:
        print(f"  Streaming {name} validation split...")
        ds = load_dataset(name, split="validation",
                          streaming=True, token=token)
        images = systematic_sample(ds, IMAGENET_VAL_SIZE, n_sample, seed=42)
        if images:
            print(f"  Got {len(images)} ImageNet-val images")
            return images
    except Exception as e:
        err = str(e).lower()
        if "gated" in err or "access" in err or "401" in err or "403" in err:
            print(f"  ACCESS DENIED — please accept the license first:")
            print(f"    https://huggingface.co/datasets/{name}")
            print(f"  (Click 'Access repository', takes 10 seconds)")
        else:
            print(f"  Failed: {e}")
    return None


def load_objectnet(n_sample: int, token: str):
    """Load ObjectNet images, trying multiple HF sources."""
    from datasets import load_dataset

    sources = [
        ("clip-benchmark/wds_objectnet", "test", "jpg"),
        ("timm/objectnet", "test", "image"),
        ("visual-layer/objectnet", "test", "image"),
    ]
    for name, split, img_key in sources:
        try:
            print(f"  Trying {name} [{split}]...")
            ds = load_dataset(name, split=split, streaming=True,
                              token=token)
            images = systematic_sample(ds, OBJECTNET_SIZE, n_sample,
                                       image_key=img_key, seed=43)
            if images:
                print(f"  Got {len(images)} ObjectNet images")
                return images
        except Exception as e:
            print(f"    not available: {e}")
    return None


def main():
    from datasets import Dataset, Image
    from huggingface_hub import login

    print("=" * 60)
    print("Creating Stratified Proxy Image Dataset")
    print(f"  Target:  {N_TOTAL} images")
    print(f"           {N_IMAGENET} ImageNet-val  (49.9%)")
    print(f"           {N_OBJECTNET} ObjectNet     (50.1%)")
    print(f"  Repo:    {HF_REPO}")
    print("=" * 60)

    login(token=HF_TOKEN)

    all_images = []
    all_sources = []

    # ---- ImageNet-val ----
    print("\n--- ImageNet Validation ---")
    inet = load_imagenet_val(N_IMAGENET, HF_TOKEN)
    if inet:
        all_images.extend(inet)
        all_sources.extend(["imagenet_val"] * len(inet))

    # ---- ObjectNet ----
    print("\n--- ObjectNet ---")
    objnet = load_objectnet(N_OBJECTNET, HF_TOKEN)
    if objnet:
        all_images.extend(objnet)
        all_sources.extend(["objectnet"] * len(objnet))

    # ---- Handle missing datasets ----
    if not all_images:
        print("\nERROR: Could not load any images.")
        print("  1. Accept ImageNet license at:")
        print("     https://huggingface.co/datasets/ILSVRC/imagenet-1k")
        print("  2. Re-run this script.")
        sys.exit(1)

    shortage = N_TOTAL - len(all_images)
    if shortage > 0:
        available = "ImageNet" if inet else "ObjectNet"
        print(f"\n  Compensating for missing dataset with {shortage} "
              f"extra {available} images...")
        if inet:
            extra = load_imagenet_val(shortage, HF_TOKEN)
        else:
            extra = load_objectnet(shortage, HF_TOKEN)
        if extra:
            all_images.extend(extra[:shortage])
            all_sources.extend([all_sources[0]] * min(len(extra), shortage))

    # ---- Deterministic shuffle ----
    rng = np.random.RandomState(44)
    idx = list(range(len(all_images)))
    rng.shuffle(idx)
    all_images = [all_images[i] for i in idx]
    all_sources = [all_sources[i] for i in idx]

    # ---- Stats ----
    counts = Counter(all_sources)
    print(f"\n--- Dataset Composition ---")
    for src, n in counts.most_common():
        print(f"  {src}: {n} ({100 * n / len(all_images):.1f}%)")
    print(f"  Total: {len(all_images)}")

    # ---- Create and push ----
    print(f"\nBuilding HuggingFace dataset...")
    ds = Dataset.from_dict({
        "image": all_images,
        "source": all_sources,
    })
    ds = ds.cast_column("image", Image())

    print(f"Pushing to {HF_REPO} (private)...")
    ds.push_to_hub(HF_REPO, private=True, token=HF_TOKEN)

    print(f"\n{'=' * 60}")
    print(f"Done! Dataset: https://huggingface.co/datasets/{HF_REPO}")
    print(f"\nCluster usage:")
    print(f"  python gpu_pipeline.py --hf-proxy-dataset {HF_REPO} \\")
    print(f"    --hf-token {HF_TOKEN}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
