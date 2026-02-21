#!/usr/bin/env python3
"""
GPU Pipeline for Blue Team — ICLR 2026 Re-Align Hackathon

Computes the *real* CKA matrix between all 141 models, then finds the optimal
20-model subset to maximize mean pairwise CKA.

Strategy: CKA(K, L) = <K_c, L_c>_F / (||K_c||_F * ||L_c||_F)
where K_c is the centered Gram matrix. This is just cosine similarity of
flattened centered Gram matrices — so we:
  1. Extract centered Gram matrix per model (500x500 each, ~2MB)
  2. Compute full 141x141 CKA via one matrix multiply
  3. Run comprehensive optimization on the real CKA matrix

Supports checkpointing: if interrupted, resumes from where it left off.

Usage:
  python gpu_pipeline.py --hf-proxy-dataset USER/proxy-1000 --hf-token hf_...
  python gpu_pipeline.py --imagenet-val-dir /path/to/val --objectnet-dir /path/to/obj
  python gpu_pipeline.py --image-dir /path/to/images
  python gpu_pipeline.py --phase cka      # skip extraction, recompute CKA
  python gpu_pipeline.py --phase optimize  # skip CKA, re-run optimization
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import gc
import warnings
from pathlib import Path

import numpy as np

REGISTRY_PATH = "configs_blue_team_model_registry.txt"
GRAM_DIR = "gram_matrices"
CKA_MATRIX_PATH = "cka_matrix.npz"
RESULTS_DIR = "results"

# Evaluation universe distribution (from red_team_stimuli_catalog.jsonl)
# Total: 100,273 images — 50,273 ObjectNet (50.1%) + 50,000 ImageNet-val (49.9%)
CATALOG_WEIGHTS = {"imagenet_val": 50000, "objectnet": 50273}


# ---------------------------------------------------------------------------
# Phase 1: Embedding Extraction → Centered Gram Matrices
# ---------------------------------------------------------------------------

def get_adaptive_batch_size(crop_size: int, base_vram_gb: float = 16.0) -> int:
    """Choose batch size based on crop size and available VRAM."""
    if crop_size <= 224:
        return 64
    elif crop_size <= 256:
        return 48
    elif crop_size <= 288:
        return 32
    elif crop_size <= 384:
        return 16
    elif crop_size <= 512:
        return 8
    elif crop_size <= 768:
        return 4
    else:
        return 2


def _find_images(directory: str) -> list[str]:
    """Recursively find all image files in a directory."""
    exts = {'.jpg', '.jpeg', '.png', '.JPEG', '.JPG', '.PNG',
            '.bmp', '.tiff', '.webp'}
    paths = []
    for root, _, files in os.walk(directory):
        for f in sorted(files):
            if Path(f).suffix in exts:
                paths.append(os.path.join(root, f))
    return sorted(paths)


def _compute_image_hash(image_paths: list[str]) -> str:
    """Compute a short hash of the image set for cache invalidation."""
    content = json.dumps(sorted(image_paths)[:200] + [str(len(image_paths))])
    return hashlib.md5(content.encode()).hexdigest()[:12]


def check_image_manifest(gram_dir: str, image_paths: list[str]) -> bool:
    """Check if cached Gram matrices match the current image set.

    Returns True if cache is valid, False if stale (clears stale caches).
    """
    manifest_path = os.path.join(gram_dir, "image_manifest.json")
    current_hash = _compute_image_hash(image_paths)

    if os.path.exists(manifest_path):
        try:
            saved = json.loads(Path(manifest_path).read_text())
            if saved.get("hash") == current_hash:
                print(f"  Image set matches cache (hash={current_hash})")
                return True
            stale_count = len(list(Path(gram_dir).glob("*.npy")))
            if stale_count > 0:
                print(f"  WARNING: Image set changed! "
                      f"(old={saved.get('hash')}, new={current_hash})")
                print(f"  Clearing {stale_count} stale Gram matrices...")
                for f in Path(gram_dir).glob("*.npy"):
                    f.unlink()
        except Exception:
            pass

    os.makedirs(gram_dir, exist_ok=True)
    Path(manifest_path).write_text(json.dumps({
        "hash": current_hash,
        "n_images": len(image_paths),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, indent=2))
    return False


def _load_stratified(n_images: int, imagenet_val_dir: str | None,
                     objectnet_dir: str | None) -> list[str] | None:
    """Load a stratified proxy set matching the evaluation distribution.

    The evaluation universe is ~50% ImageNet-val + ~50% ObjectNet.
    Samples proportionally from whichever datasets are available.
    """
    sources = {}

    if imagenet_val_dir and os.path.isdir(imagenet_val_dir):
        paths = _find_images(imagenet_val_dir)
        if paths:
            sources['imagenet_val'] = paths
            print(f"  Found {len(paths):,} ImageNet-val images")

    if objectnet_dir and os.path.isdir(objectnet_dir):
        paths = _find_images(objectnet_dir)
        if paths:
            sources['objectnet'] = paths
            print(f"  Found {len(paths):,} ObjectNet images")

    if not sources:
        return None

    # Proportional allocation matching the catalog distribution
    available_weights = {k: CATALOG_WEIGHTS[k] for k in sources
                         if k in CATALOG_WEIGHTS}
    if not available_weights:
        available_weights = {k: 1 for k in sources}
    total_weight = sum(available_weights.values())

    allocation = {}
    assigned = 0
    names = list(available_weights.keys())
    for i, name in enumerate(names):
        if i == len(names) - 1:
            allocation[name] = n_images - assigned
        else:
            n = round(n_images * available_weights[name] / total_weight)
            allocation[name] = min(n, n_images - assigned)
            assigned += allocation[name]

    # Sample from each dataset
    rng = np.random.RandomState(42)
    all_paths = []

    for dataset_name in names:
        n_sample = allocation[dataset_name]
        pool = sources[dataset_name]
        if len(pool) > n_sample:
            idx = rng.choice(len(pool), n_sample, replace=False)
            sampled = [pool[i] for i in sorted(idx)]
        else:
            sampled = pool
        all_paths.extend(sampled)
        print(f"  Sampled {len(sampled):,} from {dataset_name} "
              f"(target: {n_sample:,})")

    # Deterministic shuffle to interleave datasets
    rng_shuffle = np.random.RandomState(44)
    shuffle_idx = np.arange(len(all_paths))
    rng_shuffle.shuffle(shuffle_idx)
    all_paths = [all_paths[i] for i in shuffle_idx]

    print(f"  Stratified proxy set: {len(all_paths):,} images "
          f"({', '.join(f'{k}={v}' for k, v in allocation.items())})")
    return all_paths


def _load_from_hf_dataset(dataset_name: str, hf_token: str | None,
                          n_images: int,
                          cache_dir: str = "proxy_images") -> list[str] | None:
    """Load proxy images from a private HuggingFace dataset.

    Downloads once and caches as local JPEGs.  Subsequent runs reuse the cache.
    """
    from PIL import Image

    os.makedirs(cache_dir, exist_ok=True)

    # Fast path: already cached from a prior run?
    manifest_path = os.path.join(cache_dir, "hf_manifest.json")
    if os.path.exists(manifest_path):
        try:
            saved = json.loads(Path(manifest_path).read_text())
            if saved.get("dataset") == dataset_name and saved.get("n") >= n_images:
                existing = _find_images(cache_dir)
                if len(existing) >= n_images:
                    print(f"  Using {len(existing)} cached images "
                          f"from {dataset_name}")
                    return existing[:n_images]
        except Exception:
            pass

    try:
        from datasets import load_dataset

        print(f"  Downloading {dataset_name} ...")
        ds = load_dataset(dataset_name, token=hf_token, split="train")

        paths = []
        sources: dict[str, int] = {}
        for i, row in enumerate(ds):
            if i >= n_images:
                break
            img = row["image"]
            if not isinstance(img, Image.Image):
                continue
            img = img.convert("RGB")
            src = row.get("source", "unknown")
            sources[src] = sources.get(src, 0) + 1
            out_path = os.path.join(cache_dir, f"proxy_{i:05d}.jpg")
            img.save(out_path, quality=95)
            paths.append(out_path)
            if (i + 1) % 200 == 0:
                print(f"    {i + 1}/{min(n_images, len(ds))} saved ...")

        Path(manifest_path).write_text(json.dumps({
            "dataset": dataset_name,
            "n": len(paths),
            "sources": sources,
        }, indent=2))

        print(f"  Loaded {len(paths)} images from {dataset_name}")
        for src, n in sorted(sources.items(), key=lambda x: -x[1]):
            print(f"    {src}: {n}")
        return paths
    except Exception as e:
        print(f"  WARNING: Could not load HF dataset {dataset_name}: {e}")
        return None


def load_proxy_images(image_dir: str | None, n_images: int,
                      auto_download: bool,
                      imagenet_val_dir: str | None = None,
                      objectnet_dir: str | None = None,
                      hf_proxy_dataset: str | None = None,
                      hf_token: str | None = None) -> list[str]:
    """Load proxy images for CKA computation.

    Priority:
      1. Private HF dataset  (--hf-proxy-dataset, most portable)
      2. Stratified from ImageNet-val + ObjectNet dirs
      3. Single --image-dir
      4. Auto-download fallback (Food-101)
    """
    # 1. Private HF proxy dataset (best: tiny download, matches eval dist)
    if hf_proxy_dataset:
        paths = _load_from_hf_dataset(hf_proxy_dataset, hf_token, n_images)
        if paths and len(paths) >= 100:
            return paths
        print("  HF dataset loading failed, trying fallbacks...")

    # 2. Stratified from local directories
    if imagenet_val_dir or objectnet_dir:
        paths = _load_stratified(n_images, imagenet_val_dir, objectnet_dir)
        if paths and len(paths) >= 100:
            return paths
        print("  Stratified loading insufficient, trying fallbacks...")

    # 3. Single image directory
    if image_dir and os.path.isdir(image_dir):
        paths = _find_images(image_dir)
        if len(paths) == 0:
            raise ValueError(f"No images found in {image_dir}")
        np.random.seed(42)
        if len(paths) > n_images:
            idx = np.random.choice(len(paths), n_images, replace=False)
            paths = [paths[i] for i in sorted(idx)]
        print(f"  Using {len(paths)} images from {image_dir}")
        return paths

    # 4. Auto-download fallback
    if auto_download:
        return download_proxy_images(n_images)

    raise ValueError(
        "No image source provided. Use:\n"
        "  --hf-proxy-dataset USER/REPO                  (best, see create_proxy_dataset.py)\n"
        "  --imagenet-val-dir PATH --objectnet-dir PATH  (local stratified)\n"
        "  --image-dir PATH                              (single directory)\n"
        "  --auto-download                               (Food-101 fallback)"
    )


def download_proxy_images(n_images: int) -> list[str]:
    """Download proxy images from a freely available dataset (fallback)."""
    download_dir = "proxy_images"
    os.makedirs(download_dir, exist_ok=True)

    existing = sorted(Path(download_dir).glob("*.jpg")) + \
               sorted(Path(download_dir).glob("*.png")) + \
               sorted(Path(download_dir).glob("*.JPEG"))
    if len(existing) >= n_images:
        print(f"  Found {len(existing)} cached proxy images")
        return [str(p) for p in existing[:n_images]]

    print("  Downloading proxy images (Food-101 test set)...")
    print("  NOTE: For better CKA estimates, use --imagenet-val-dir and "
          "--objectnet-dir instead")
    try:
        import torchvision.datasets as dsets
        food = dsets.Food101(root=download_dir, split="test", download=True)
        indices = np.random.RandomState(42).choice(
            len(food), min(n_images, len(food)), replace=False
        )
        paths = []
        for i, idx in enumerate(indices):
            img, _ = food[idx]
            out_path = os.path.join(download_dir, f"food_{i:05d}.jpg")
            img.save(out_path)
            paths.append(out_path)
            if (i + 1) % 100 == 0:
                print(f"    Saved {i + 1}/{len(indices)} images")
        print(f"  Downloaded {len(paths)} proxy images")
        return paths
    except Exception as e:
        print(f"  Food-101 download failed: {e}")
        raise RuntimeError(
            "Could not download proxy images. Please provide\n"
            "  --imagenet-val-dir / --objectnet-dir or --image-dir"
        )


def extract_gram_matrix(model_info: dict, image_paths: list[str],
                        device: str, gram_dir: str) -> str | None:
    """Extract the centered Gram matrix for a single model.

    Returns the path to the saved .npy file, or None on failure.
    """
    import torch
    import timm
    from PIL import Image
    from torchvision import transforms

    name = model_info["model_name"]
    layer = model_info["layer"]
    pp = model_info["preprocess"]

    out_path = os.path.join(gram_dir, f"{name}.npy")
    if os.path.exists(out_path):
        return out_path

    crop = pp["crop"]
    batch_size = get_adaptive_batch_size(crop)
    n = len(image_paths)

    try:
        model = timm.create_model(name, pretrained=True)
        model.eval()
        model.to(device)
        if device == "cuda":
            model.half()

        transform = transforms.Compose([
            transforms.Resize(pp["resize"], interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(crop),
            transforms.ToTensor(),
            transforms.Normalize(mean=pp["mean"], std=pp["std"]),
        ])

        # Find target layer
        modules = dict(model.named_modules())
        if layer not in modules:
            print(f"    WARNING: layer '{layer}' not found in {name}")
            available = [k for k in modules.keys() if k]
            suggestions = [k for k in available if any(
                x in k for x in ['norm', 'pool', 'head', 'fc']
            )]
            if suggestions:
                print(f"    Possible layers: {suggestions[:5]}")
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
            return None

        target_module = modules[layer]

        all_features = []
        features_buf = {}

        def hook_fn(module, inp, out):
            features_buf["out"] = out

        handle = target_module.register_forward_hook(hook_fn)

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device == "cuda")):
            for i in range(0, n, batch_size):
                batch_paths = image_paths[i:i + batch_size]
                imgs = []
                for p in batch_paths:
                    try:
                        img = Image.open(p).convert("RGB")
                        imgs.append(transform(img))
                    except Exception:
                        imgs.append(torch.zeros(3, crop, crop))
                if not imgs:
                    continue
                batch = torch.stack(imgs).to(device)
                if device == "cuda":
                    batch = batch.half()

                try:
                    model(batch)
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        torch.cuda.empty_cache()
                        half_bs = max(1, len(imgs) // 2)
                        for sub_i in range(0, len(imgs), half_bs):
                            sub_batch = torch.stack(imgs[sub_i:sub_i + half_bs]).to(device)
                            if device == "cuda":
                                sub_batch = sub_batch.half()
                            model(sub_batch)
                            feat = features_buf["out"]
                            if feat.dim() > 2:
                                feat = feat.flatten(1)
                            all_features.append(feat.float().cpu().numpy())
                            del sub_batch
                            torch.cuda.empty_cache()
                        continue
                    else:
                        raise

                feat = features_buf["out"]
                if feat.dim() > 2:
                    feat = feat.flatten(1)
                all_features.append(feat.float().cpu().numpy())

                del batch
                if device == "cuda" and i % (batch_size * 4) == 0:
                    torch.cuda.empty_cache()

        handle.remove()

        if not all_features:
            print(f"    ERROR: No features extracted for {name}")
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
            return None

        X = np.concatenate(all_features, axis=0).astype(np.float64)

        # Column-center
        X -= X.mean(axis=0, keepdims=True)

        # Compute Gram matrix
        K = X @ X.T  # (N, N)

        # Center Gram matrix: K_c = H @ K @ H where H = I - 1/n
        n_samples = K.shape[0]
        row_mean = K.mean(axis=1, keepdims=True)
        col_mean = K.mean(axis=0, keepdims=True)
        grand_mean = K.mean()
        K_c = K - row_mean - col_mean + grand_mean

        np.save(out_path, K_c)

        del model, X, K, K_c, all_features
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

        return out_path

    except Exception as e:
        print(f"    ERROR processing {name}: {e}")
        if device == "cuda":
            import torch
            torch.cuda.empty_cache()
        gc.collect()
        return None


def _cleanup_model_cache(model_name: str):
    """Remove downloaded model weights from HF cache after extraction.

    Each model is ~100-400 MB.  Since we only need the Gram matrix after
    extraction, purging the weights keeps disk usage bounded to ~1 model
    at a time instead of accumulating 30-50 GB for all 141 models.
    """
    hf_home = os.environ.get(
        "HF_HOME",
        os.path.join(os.path.expanduser("~"), ".cache", "huggingface"),
    )
    model_cache = os.path.join(hf_home, "hub", f"models--timm--{model_name}")
    if os.path.isdir(model_cache):
        try:
            shutil.rmtree(model_cache)
        except OSError:
            pass


def run_extraction(registry: list[dict], image_paths: list[str],
                   device: str, gram_dir: str) -> dict[str, str]:
    """Extract centered Gram matrices for all models."""
    os.makedirs(gram_dir, exist_ok=True)
    n_models = len(registry)
    results = {}
    skipped = 0
    failed = 0

    for i, model_info in enumerate(registry):
        name = model_info["model_name"]
        out_path = os.path.join(gram_dir, f"{name}.npy")

        if os.path.exists(out_path):
            results[name] = out_path
            skipped += 1
            continue

        elapsed_str = ""
        t0 = time.time()
        print(f"[{i + 1:3d}/{n_models}] {name}...", end="", flush=True)

        path = extract_gram_matrix(model_info, image_paths, device, gram_dir)
        dt = time.time() - t0

        if path:
            results[name] = path
            print(f" done ({dt:.1f}s, dim={np.load(path).shape})")
        else:
            failed += 1
            print(f" FAILED ({dt:.1f}s)")

        # Free model weights from disk cache (we only need the Gram matrix)
        _cleanup_model_cache(name)

    print(f"\nExtraction complete: {len(results)} success, "
          f"{skipped} cached, {failed} failed")
    return results


# ---------------------------------------------------------------------------
# Phase 2: CKA Matrix Computation
# ---------------------------------------------------------------------------

def compute_cka_matrix(gram_dir: str, registry: list[dict],
                       out_path: str) -> tuple[np.ndarray, list[str]]:
    """Compute the full CKA matrix from saved centered Gram matrices.

    CKA(K_c, L_c) = <K_c, L_c>_F / (||K_c||_F * ||L_c||_F)
    = cosine similarity of flattened centered Gram matrices.
    """
    model_names = []
    gram_flat = []

    for model_info in registry:
        name = model_info["model_name"]
        gram_path = os.path.join(gram_dir, f"{name}.npy")
        if not os.path.exists(gram_path):
            continue
        K_c = np.load(gram_path)
        flat = K_c.ravel()
        norm = np.linalg.norm(flat)
        # Skip degenerate models (all-zero, inf, or NaN Gram matrices)
        if not np.isfinite(norm) or norm < 1e-8:
            print(f"  WARNING: Skipping {name} (degenerate Gram, norm={norm})")
            continue
        gram_flat.append(flat)
        model_names.append(name)

    print(f"  Loaded {len(model_names)} centered Gram matrices")
    if len(model_names) < 20:
        raise ValueError(
            f"Only {len(model_names)} models available, need at least 20"
        )

    G = np.stack(gram_flat, axis=0)  # (n_models, N^2)
    norms = np.linalg.norm(G, axis=1, keepdims=True)
    G_normed = G / norms

    # CKA matrix = cosine similarity
    S = G_normed @ G_normed.T

    # Sanitize: replace any residual NaN/inf with 0, clamp to [0, 1]
    np.nan_to_num(S, copy=False, nan=0.0, posinf=1.0, neginf=0.0)
    np.clip(S, 0.0, 1.0, out=S)
    np.fill_diagonal(S, 1.0)

    np.savez(out_path, cka=S, model_names=model_names)
    print(f"  CKA matrix shape: {S.shape}")
    print(f"  Mean off-diagonal CKA: {(S.sum() - len(model_names)) / (len(model_names) * (len(model_names) - 1)):.4f}")
    print(f"  Max off-diagonal CKA: {np.max(S - np.eye(len(model_names))):.4f}")
    print(f"  Min off-diagonal CKA: {np.min(S + np.eye(len(model_names)) * 999):.4f}")

    return S, model_names


# ---------------------------------------------------------------------------
# Phase 3: Optimization (imported from blue_team_optimizer.py + enhancements)
# ---------------------------------------------------------------------------

def _obj(S: np.ndarray, selected: list[int], k: int) -> float:
    """Mean pairwise CKA for a selection."""
    sub = S[np.ix_(selected, selected)]
    return (sub.sum() - k) / (k * (k - 1))


def greedy_densest_k(S: np.ndarray, k: int) -> tuple[list[int], float]:
    """Greedy: iteratively add model maximizing sum of CKA to selected."""
    n = S.shape[0]
    off_diag = S.copy()
    np.fill_diagonal(off_diag, -np.inf)
    best_pair = np.unravel_index(np.argmax(off_diag), S.shape)
    selected = list(best_pair)
    remaining = set(range(n)) - set(selected)

    while len(selected) < k:
        scores = S[list(remaining)][:, selected].sum(axis=1)
        remaining_list = list(remaining)
        best_idx = remaining_list[np.argmax(scores)]
        selected.append(best_idx)
        remaining.remove(best_idx)

    return selected, _obj(S, selected, k)


def spectral_densest_k(S: np.ndarray, k: int,
                       n_restarts: int = 500) -> tuple[list[int], float]:
    """Spectral method: top eigenvectors + randomized rounding."""
    from scipy.linalg import eigh
    eigenvalues, eigenvectors = eigh(S)

    best_obj = -np.inf
    best_set = None

    v1 = eigenvectors[:, -1]
    top_k = np.argsort(np.abs(v1))[-k:]
    obj = _obj(S, top_k.tolist(), k)
    if obj > best_obj:
        best_obj = obj
        best_set = top_k.tolist()

    rng = np.random.RandomState(42)
    for r in [1, 2, 3, 5, 10, 20, 30, 50]:
        if r > S.shape[0]:
            continue
        V = eigenvectors[:, -r:]
        for _ in range(n_restarts):
            w = rng.randn(r)
            scores = V @ w
            top_k = np.argsort(scores)[-k:]
            obj = _obj(S, top_k.tolist(), k)
            if obj > best_obj:
                best_obj = obj
                best_set = top_k.tolist()

    return best_set, best_obj


def truncated_power_method(S: np.ndarray, k: int,
                           max_iter: int = 2000) -> tuple[list[int], float]:
    """Truncated power method for sparse leading eigenvector."""
    n = S.shape[0]
    best_obj = -np.inf
    best_set = None

    for seed in range(10):
        rng = np.random.RandomState(seed)
        x = rng.randn(n)
        x /= np.linalg.norm(x)

        for _ in range(max_iter):
            x_new = S @ x
            if k < n:
                threshold = np.sort(np.abs(x_new))[-k]
                x_new[np.abs(x_new) < threshold] = 0
            norm = np.linalg.norm(x_new)
            if norm < 1e-12:
                x = rng.randn(n)
                x /= np.linalg.norm(x)
                continue
            x_new /= norm
            if np.linalg.norm(x_new - x) < 1e-8:
                break
            x = x_new

        selected = np.argsort(np.abs(x))[-k:]
        obj = _obj(S, selected.tolist(), k)
        if obj > best_obj:
            best_obj = obj
            best_set = selected.tolist()

    return best_set, best_obj


def frank_wolfe_dks(S: np.ndarray, k: int, lam: float = 1.0,
                    max_iter: int = 1000) -> tuple[list[int], float]:
    """Frank-Wolfe with diagonal loading."""
    n = S.shape[0]
    S_diag = S + lam * np.eye(n)
    L_norm = np.linalg.norm(S_diag, ord=2)

    x = np.ones(n) * (k / n)

    for t in range(max_iter):
        grad = S_diag @ x
        top_k = np.argpartition(grad, -k)[-k:]
        s = np.zeros(n)
        s[top_k] = 1.0
        d = s - x
        fw_gap = grad @ d
        if fw_gap <= 1e-10:
            break
        gamma = min(1.0, fw_gap / (L_norm * np.linalg.norm(d) ** 2 + 1e-12))
        x = x + gamma * d

    selected = np.argpartition(x, -k)[-k:]
    return selected.tolist(), _obj(S, selected.tolist(), k)


def local_search(S: np.ndarray, k: int, initial_set: list[int],
                 max_no_improve: int = 2000,
                 max_iter: int = 50000) -> tuple[list[int], float]:
    """Vectorized steepest-ascent local search with 1-swap neighborhood."""
    n = S.shape[0]
    sel = np.array(sorted(initial_set[:k]))
    best_obj = _obj(S, sel.tolist(), k)
    best_set = sel.tolist()
    no_improve = 0
    total_iter = 0
    min_delta = 1e-10
    rng = np.random.RandomState(42)

    while no_improve < max_no_improve and total_iter < max_iter:
        total_iter += 1
        unsel = np.setdiff1d(np.arange(n), sel)
        S_sel = S[np.ix_(sel, sel)]
        out_scores = S_sel.sum(axis=1) - 1.0
        in_scores_full = S[np.ix_(unsel, sel)]

        found = False
        best_swap_delta = min_delta
        best_out_idx = -1
        best_in_idx = -1

        for oi in range(k):
            loss = out_scores[oi]
            gains = in_scores_full.sum(axis=1) - in_scores_full[:, oi]
            deltas = gains - loss
            max_idx = np.argmax(deltas)
            if deltas[max_idx] > best_swap_delta:
                best_swap_delta = deltas[max_idx]
                best_out_idx = oi
                best_in_idx = max_idx
                found = True

        if found:
            sel[best_out_idx] = unsel[best_in_idx]
            sel.sort()
            obj = _obj(S, sel.tolist(), k)
            if obj > best_obj:
                best_obj = obj
                best_set = sel.tolist()
            no_improve = 0
        else:
            no_improve += 1
            oi = rng.randint(k)
            unsel = np.setdiff1d(np.arange(n), sel)
            ii = rng.randint(len(unsel))
            sel[oi] = unsel[ii]
            sel.sort()

    return best_set, best_obj


def simulated_annealing(S: np.ndarray, k: int, initial_set: list[int],
                        T_start: float = 0.1, T_end: float = 1e-7,
                        n_iter: int = 2_000_000,
                        seed: int = 0) -> tuple[list[int], float]:
    """Simulated annealing with 1-swap moves."""
    n = S.shape[0]
    rng = np.random.RandomState(seed)
    in_set = np.zeros(n, dtype=bool)
    for i in initial_set[:k]:
        in_set[i] = True

    sel_indices = np.where(in_set)[0]
    current_obj = _obj(S, sel_indices.tolist(), k)
    best_obj = current_obj
    best_set = sel_indices.tolist()

    alpha = (T_end / T_start) ** (1.0 / n_iter)
    T = T_start

    for it in range(n_iter):
        sel = np.where(in_set)[0]
        unsel = np.where(~in_set)[0]
        v_out = rng.choice(sel)
        v_in = rng.choice(unsel)

        T_minus = sel[sel != v_out]
        gain = S[v_in, T_minus].sum() - S[v_out, T_minus].sum()
        delta_obj = gain * 2 / (k * (k - 1))

        if delta_obj > 0 or rng.rand() < np.exp(delta_obj / max(T, 1e-15)):
            in_set[v_out] = False
            in_set[v_in] = True
            current_obj += delta_obj
            if current_obj > best_obj:
                best_obj = current_obj
                best_set = np.where(in_set)[0].tolist()

        T *= alpha

    return best_set, best_obj


def comprehensive_optimize(S: np.ndarray, k: int = 20,
                           model_names: list[str] | None = None,
                           verbose: bool = True) -> tuple[list[int], float]:
    """Run all optimization algorithms and return the best result."""
    results = {}

    # 1. Row-sum heuristic
    print("  Running row-sum heuristic...")
    row_means = (S.sum(axis=1) - 1) / (S.shape[0] - 1)
    top_k = np.argsort(row_means)[-k:]
    results["rowsum"] = (top_k.tolist(), _obj(S, top_k.tolist(), k))

    # 2. Greedy
    print("  Running greedy...")
    results["greedy"] = greedy_densest_k(S, k)

    # 3. Spectral (heavy randomized rounding)
    print("  Running spectral (500 restarts)...")
    results["spectral"] = spectral_densest_k(S, k, n_restarts=500)

    # 4. Truncated power
    print("  Running truncated power method...")
    results["truncated_power"] = truncated_power_method(S, k)

    # 5. Frank-Wolfe with multiple lambdas
    print("  Running Frank-Wolfe...")
    best_fw_obj = -np.inf
    best_fw_sel = None
    for lam in [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
        sel, obj = frank_wolfe_dks(S, k, lam=lam)
        if obj > best_fw_obj:
            best_fw_obj = obj
            best_fw_sel = sel
    results["frank_wolfe"] = (best_fw_sel, best_fw_obj)

    if verbose:
        print("\n  === Initial Algorithm Results ===")
        for method, (sel, obj) in sorted(results.items(), key=lambda x: -x[1][1]):
            print(f"    {method:20s}: {obj:.6f}")

    # 6. Local search from each initial solution
    print("\n  Running local search refinement...")
    refined = {}
    for method, (sel, obj) in results.items():
        sel_ls, obj_ls = local_search(S, k, sel, max_no_improve=5000)
        refined[f"{method}+LS"] = (sel_ls, obj_ls)

    if verbose:
        print("\n  === After Local Search ===")
        for method, (sel, obj) in sorted(refined.items(), key=lambda x: -x[1][1]):
            print(f"    {method:20s}: {obj:.6f}")

    # 7. Simulated annealing from best + random restarts
    all_results = {**results, **refined}
    best_method = max(all_results, key=lambda m: all_results[m][1])
    best_init = all_results[best_method][0]

    print("\n  Running simulated annealing (8 restarts, 2M iterations each)...")
    sa_results = {}
    for i in range(8):
        if i == 0:
            init = best_init
        elif i <= 3:
            # Perturb best solution
            init = list(best_init)
            rng = np.random.RandomState(i)
            n_perturb = max(1, k // 4)
            for _ in range(n_perturb):
                out_idx = rng.randint(k)
                available = list(set(range(S.shape[0])) - set(init))
                init[out_idx] = rng.choice(available)
        else:
            init = np.random.RandomState(i + 100).choice(
                S.shape[0], k, replace=False
            ).tolist()

        print(f"    SA restart {i}...", end="", flush=True)
        t0 = time.time()
        sel_sa, obj_sa = simulated_annealing(
            S, k, init, T_start=0.1, T_end=1e-7,
            n_iter=2_000_000, seed=i
        )
        # Polish with local search
        sel_sa, obj_sa = local_search(S, k, sel_sa, max_no_improve=3000)
        dt = time.time() - t0
        sa_results[f"SA_restart_{i}"] = (sel_sa, obj_sa)
        print(f" {obj_sa:.6f} ({dt:.1f}s)")

    if verbose:
        print("\n  === Simulated Annealing Results ===")
        for method, (sel, obj) in sorted(sa_results.items(), key=lambda x: -x[1][1]):
            print(f"    {method:20s}: {obj:.6f}")

    # Overall best
    all_final = {**all_results, **refined, **sa_results}
    best_method = max(all_final, key=lambda m: all_final[m][1])
    best_sel, best_obj = all_final[best_method]

    if verbose:
        print(f"\n  {'=' * 60}")
        print(f"  BEST METHOD: {best_method}")
        print(f"  BEST SCORE:  {best_obj:.6f}")
        if model_names:
            print(f"\n  Selected models ({k}):")
            for idx in sorted(best_sel):
                print(f"    {model_names[idx]}")
        print(f"  {'=' * 60}")

    return best_sel, best_obj, all_final


# ---------------------------------------------------------------------------
# Phase 4: Submission
# ---------------------------------------------------------------------------

def generate_submission(best_sel: list[int], model_names: list[str],
                        registry: list[dict]) -> list[dict]:
    """Generate the submission JSON payload."""
    name_to_info = {m["model_name"]: m for m in registry}
    submission = []
    for idx in sorted(best_sel):
        name = model_names[idx]
        info = name_to_info[name]
        submission.append({
            "model_name": name,
            "layer_name": info["layer"],
        })
    return submission


def push_to_huggingface(submission: list[dict], hf_token: str,
                        repo_name: str) -> str:
    """Push submission to HuggingFace as a private dataset."""
    from datasets import Dataset
    from huggingface_hub import login

    login(token=hf_token)
    rows = [{"model_name": m["model_name"], "layer_name": m["layer_name"]}
            for m in submission]
    ds = Dataset.from_list(rows)
    ds.push_to_hub(repo_name, private=True)
    return f"https://huggingface.co/datasets/{repo_name}"


# ---------------------------------------------------------------------------
# Diagnostic Utilities
# ---------------------------------------------------------------------------

def print_cka_analysis(S: np.ndarray, model_names: list[str],
                       registry: list[dict]):
    """Print diagnostic analysis of the CKA matrix."""
    from blue_team_optimizer import classify_architecture

    n = len(model_names)
    name_to_arch = {}
    for name in model_names:
        name_to_arch[name] = classify_architecture(name)

    archs = sorted(set(name_to_arch.values()))
    print("\n  === CKA Analysis by Architecture Family ===")
    for arch in archs:
        members = [i for i, name in enumerate(model_names)
                   if name_to_arch[name] == arch]
        if len(members) < 2:
            continue
        sub = S[np.ix_(members, members)]
        mean_cka = (sub.sum() - len(members)) / (len(members) * (len(members) - 1))
        print(f"    {arch:25s}: {len(members):3d} models, "
              f"mean intra-CKA = {mean_cka:.4f}")

    # Top-20 most similar pairs
    print("\n  === Top 20 Most Similar Model Pairs ===")
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((S[i, j], model_names[i], model_names[j]))
    pairs.sort(reverse=True)
    for score, m1, m2 in pairs[:20]:
        print(f"    {score:.4f}  {m1[:35]:35s} <-> {m2[:35]}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="GPU pipeline for Blue Team CKA optimization"
    )
    parser.add_argument("--hf-proxy-dataset", type=str, default=None,
                        help="Private HF dataset with proxy images (best option)")
    parser.add_argument("--imagenet-val-dir", type=str, default=None,
                        help="Path to ImageNet validation images (for stratified proxy)")
    parser.add_argument("--objectnet-dir", type=str, default=None,
                        help="Path to ObjectNet images (for stratified proxy)")
    parser.add_argument("--image-dir", type=str, default=None,
                        help="Directory with proxy images (fallback if no stratified dirs)")
    parser.add_argument("--auto-download", action="store_true",
                        help="Auto-download proxy images if no dirs provided")
    parser.add_argument("--n-images", type=int, default=1000,
                        help="Number of proxy images (default: 1000, more=stabler CKA)")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "cpu", "mps"],
                        help="Device for inference")
    parser.add_argument("--registry", type=str, default=REGISTRY_PATH,
                        help="Path to model registry JSON")
    parser.add_argument("--gram-dir", type=str, default=GRAM_DIR,
                        help="Directory for cached Gram matrices")
    parser.add_argument("--phase", type=str, default="all",
                        choices=["all", "extract", "cka", "optimize", "submit"],
                        help="Which phase to run")
    parser.add_argument("--k", type=int, default=20,
                        help="Number of models to select")
    parser.add_argument("--hf-token", type=str, default=None,
                        help="HuggingFace token for submission")
    parser.add_argument("--hf-repo", type=str, default=None,
                        help="HuggingFace repo name (e.g. user/blue-team-v2)")
    parser.add_argument("--output", type=str, default="submission_gpu.json",
                        help="Output JSON file")
    args = parser.parse_args()

    # Detect device
    if args.device == "auto":
        import torch
        if torch.cuda.is_available():
            device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            gpu_mem = getattr(props, 'total_memory', None) or getattr(props, 'total_mem', 0)
            gpu_mem /= 1e9
            print(f"Using GPU: {gpu_name} ({gpu_mem:.1f} GB)")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
            print("Using Apple MPS")
        else:
            device = "cpu"
            print("WARNING: No GPU detected, using CPU (will be very slow)")
    else:
        device = args.device

    # Load registry
    print("\n=== Loading Model Registry ===")
    registry = json.loads(Path(args.registry).read_text())
    print(f"  {len(registry)} models in registry")

    # Phase 1: Extract Gram matrices
    if args.phase in ("all", "extract"):
        print("\n=== Phase 1: Extracting Centered Gram Matrices ===")
        image_paths = load_proxy_images(
            args.image_dir, args.n_images, args.auto_download,
            imagenet_val_dir=args.imagenet_val_dir,
            objectnet_dir=args.objectnet_dir,
            hf_proxy_dataset=args.hf_proxy_dataset,
            hf_token=args.hf_token,
        )
        print(f"  Using {len(image_paths)} proxy images")

        # Detect image set changes and invalidate stale Gram caches
        check_image_manifest(args.gram_dir, image_paths)

        t0 = time.time()
        gram_results = run_extraction(registry, image_paths, device, args.gram_dir)
        dt = time.time() - t0
        print(f"\n  Phase 1 complete in {dt / 60:.1f} minutes")
        print(f"  {len(gram_results)} / {len(registry)} models extracted")

    # Phase 2: Compute CKA matrix
    if args.phase in ("all", "extract", "cka"):
        print("\n=== Phase 2: Computing CKA Matrix ===")
        t0 = time.time()
        S, model_names = compute_cka_matrix(
            args.gram_dir, registry, CKA_MATRIX_PATH
        )
        dt = time.time() - t0
        print(f"  Phase 2 complete in {dt:.1f} seconds")

    # Load pre-computed CKA matrix for optimize/submit phases
    if args.phase in ("optimize", "submit"):
        if not os.path.exists(CKA_MATRIX_PATH):
            print(f"ERROR: {CKA_MATRIX_PATH} not found. Run extract/cka first.")
            sys.exit(1)
        data = np.load(CKA_MATRIX_PATH, allow_pickle=True)
        S = data["cka"]
        model_names = list(data["model_names"])
        print(f"  Loaded CKA matrix: {S.shape}")

    # Phase 3: Optimize
    if args.phase in ("all", "extract", "cka", "optimize"):
        print("\n=== Phase 3: Optimization ===")

        # Print diagnostic analysis
        print_cka_analysis(S, model_names, registry)

        print(f"\n  Finding optimal {args.k} models...")
        t0 = time.time()
        best_sel, best_obj, all_results = comprehensive_optimize(
            S, k=args.k, model_names=model_names, verbose=True
        )
        dt = time.time() - t0
        print(f"\n  Phase 3 complete in {dt / 60:.1f} minutes")

        # Save results
        os.makedirs(RESULTS_DIR, exist_ok=True)
        submission = generate_submission(best_sel, model_names, registry)

        result_data = {
            "models": submission,
            "predicted_score": best_obj,
            "n_proxy_images": args.n_images,
            "n_models_available": len(model_names),
        }
        with open(args.output, "w") as f:
            json.dump(result_data, f, indent=2)
        print(f"\n  Submission saved to {args.output}")

        # Also save full results for analysis
        np.savez(
            os.path.join(RESULTS_DIR, "optimization_results.npz"),
            best_sel=np.array(best_sel),
            best_obj=best_obj,
            cka_matrix=S,
            model_names=np.array(model_names),
        )

        print("\n=== SUBMISSION JSON ===")
        print(json.dumps({"models": submission}, indent=2))

    # Phase 4: Submit to HuggingFace
    if args.phase in ("all", "submit"):
        if args.hf_token and args.hf_repo:
            print("\n=== Phase 4: Pushing to HuggingFace ===")
            if not os.path.exists(args.output):
                print(f"ERROR: {args.output} not found")
                sys.exit(1)
            with open(args.output) as f:
                result_data = json.load(f)
            url = push_to_huggingface(
                result_data["models"], args.hf_token, args.hf_repo
            )
            print(f"  Dataset pushed to: {url}")
        else:
            print("\n  Skipping HuggingFace push (no --hf-token / --hf-repo)")
            print("  To submit, run:")
            print(f"    python gpu_pipeline.py --phase submit "
                  f"--hf-token YOUR_TOKEN --hf-repo YOUR_USER/blue-team-gpu-v1")

    print("\nDone!")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()
