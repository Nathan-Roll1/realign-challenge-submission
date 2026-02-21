#!/usr/bin/env python3
"""
Red Team Exploration Survey — ICLR 2026 Re-Align Hackathon

Tests which ImageNet superclass categories produce the most representational
divergence across models.  For each category, samples images and computes the
full pairwise CKA matrix across 20 diverse proxy models, reporting:

  - Mean pairwise CKA (lower = more divergent = better for Red Team)
  - Which model pairs diverge most within each category
  - Ranking of categories by divergence potential

Usage:
  python red_team_survey.py --imagenet-val-dir /data/imagenet/val
  python red_team_survey.py --auto-download --hf-token hf_...
  python red_team_survey.py --phase report   # re-analyze cached embeddings
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import urllib.request
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REGISTRY_PATH = "configs_blue_team_model_registry.txt"
SURVEY_DIR = "red_team_survey_results"

# ---------------------------------------------------------------------------
# 20 Proxy Models — maximum architectural + training diversity
# ---------------------------------------------------------------------------
# Mix of crop sizes intentional: different preprocessing = extra divergence.

SURVEY_MODELS = [
    # ---- Pure CNNs (4) ----
    "vgg11.tv_in1k",                              # shallow classic CNN
    "resnet101.a1_in1k",                          # deep residual
    "densenet121.ra_in1k",                        # dense connections
    "convnext_atto.d2_in1k",                      # modern ConvNeXt

    # ---- Vision Transformers (4) ----
    "deit3_base_patch16_224.fb_in1k",             # supervised ViT
    "beit_base_patch16_224.in22k_ft_in22k",       # MIM pretrained
    "eva02_base_patch14_224.mim_in22k",           # EVA02 MIM
    "cait_m36_384.fb_dist_in1k",                  # class-attention (384px)

    # ---- CLIP / vision-language (2) ----
    "vit_base_mci_224.apple_mclip",               # Apple MCLIP
    "vitamin_base_224.datacomp1b_clip",            # DataComp CLIP

    # ---- Self-supervised (2) ----
    "resmlp_12_224.fb_dino",                      # DINO contrastive
    "hiera_base_224.mae",                         # MAE reconstructive

    # ---- Exotic architectures (3) ----
    "mambaout_base.in1k",                         # Mamba/SSM-derived
    "mixer_b16_224.goog_in21k",                   # MLP-Mixer
    "sequencer2d_l.in1k",                         # LSTM-based

    # ---- Hybrids + other ViT flavors (3) ----
    "maxvit_base_tf_224.in1k",                    # Multi-axis attention
    "swin_base_patch4_window12_384.ms_in1k",      # Swin (384px)
    "xcit_large_24_p16_224.fb_dist_in1k",         # Cross-covariance

    # ---- Efficient / Mobile (2) ----
    "efficientnet_b0.ra4_e3600_r224_in1k",        # NAS-designed
    "ghostnetv2_100.in1k",                        # Ghost modules
]

# ---------------------------------------------------------------------------
# ImageNet Superclass Definitions (by synset ID ranges)
# ---------------------------------------------------------------------------
# Each group is a set of synset IDs.  We define them via the well-known
# ImageNet-1K ordering where class i (0-indexed) maps to a synset.

SUPERCLASS_DEFINITIONS = {
    "dogs": {
        "description": "Domestic dogs + wild canids (118+7 classes)",
        "class_range": (151, 275),
    },
    "snakes": {
        "description": "Snakes (17 classes)",
        "class_range": (52, 68),
    },
    "insects": {
        "description": "Insects and arachnids (~27 classes)",
        "class_range": (300, 326),
    },
    "primates": {
        "description": "Monkeys and apes (~18 classes)",
        "class_range": (365, 382),
    },
    "birds": {
        "description": "Birds — cocks/hens + songbirds (~24 classes)",
        "class_range": (7, 24),
        "extra_ranges": [(80, 100)],
    },
    "cats_bears": {
        "description": "Cats (wild+domestic) + bears (19 classes)",
        "class_range": (281, 299),
    },
    "reptiles_amphibians": {
        "description": "Turtles, lizards, frogs, crocs (~33 classes)",
        "class_range": (25, 51),
        "extra_ranges": [(69, 76)],
    },
    "fish_aquatic": {
        "description": "Fish + aquatic creatures (~15 classes)",
        "class_range": (0, 6),
        "extra_ranges": [(389, 397)],
    },
    "wheeled_vehicles": {
        "description": "Cars, trucks, bikes, buses (~30 classes)",
        "class_range": (407, 437),
    },
    "containers_furniture": {
        "description": "Chairs, tables, cabinets, containers (~20 classes)",
        "class_range": (520, 545),
    },
    "clothing": {
        "description": "Clothing items (~20 classes)",
        "class_range": (474, 498),
    },
    "food_fruit": {
        "description": "Food, fruit, vegetables (~20 classes)",
        "class_range": (924, 969),
    },
    "random_diverse": {
        "description": "Uniform random from all 1000 classes (baseline)",
        "class_range": (0, 999),
        "sample_mode": "uniform",
    },
    "anti_dogs": {
        "description": "Everything EXCEPT dogs (control)",
        "class_range": (0, 999),
        "exclude_range": (151, 275),
    },
}


# ---------------------------------------------------------------------------
# Synset / Label Utilities
# ---------------------------------------------------------------------------

def download_synset_labels(cache_dir: str = SURVEY_DIR) -> list[str]:
    """Download the ordered list of 1000 ImageNet-1K synset IDs.

    Returns list where index i = synset ID for class i (0-indexed).
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "imagenet_synsets_ordered.txt")

    if os.path.exists(cache_path):
        with open(cache_path) as f:
            synsets = [line.strip() for line in f if line.strip()]
        if len(synsets) == 1000:
            return synsets

    url = ("https://raw.githubusercontent.com/tensorflow/models/master/"
           "research/slim/datasets/imagenet_2012_validation_synset_labels.txt")
    print(f"  Downloading synset labels...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        all_synsets = resp.read().decode("utf-8").strip().split("\n")

    assert len(all_synsets) == 50000, f"Expected 50000 lines, got {len(all_synsets)}"

    # Build ordered list: for each image 1-50000, the synset.
    # Images are 50 per class, classes appear in order.
    # Deduplicate while preserving order to get the 1000 unique synsets.
    seen = set()
    ordered = []
    for s in all_synsets:
        s = s.strip()
        if s not in seen:
            seen.add(s)
            ordered.append(s)

    # Also cache the per-image labels
    val_labels_path = os.path.join(cache_dir, "imagenet_val_labels.txt")
    if not os.path.exists(val_labels_path):
        with open(val_labels_path, "w") as f:
            for i, synset in enumerate(all_synsets):
                fname = f"ILSVRC2012_val_{i+1:08d}.JPEG"
                f.write(f"{fname} {synset.strip()}\n")

    with open(cache_path, "w") as f:
        for s in ordered:
            f.write(s + "\n")

    print(f"  Got {len(ordered)} unique synsets (expected ~1000)")
    return ordered


def download_class_names(cache_dir: str = SURVEY_DIR) -> dict[str, str]:
    """Download synset → human-readable class name mapping."""
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "synset_to_name.json")

    if os.path.exists(cache_path):
        return json.loads(Path(cache_path).read_text())

    url = ("https://raw.githubusercontent.com/anishathalye/"
           "imagenet-simple-labels/master/imagenet-simple-labels.json")
    print(f"  Downloading class names...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        names = json.loads(resp.read().decode("utf-8"))

    # names[i] = human-readable name for class i (0-indexed)
    # We need to pair with synsets
    synsets = download_synset_labels(cache_dir)
    mapping = {}
    for i, name in enumerate(names):
        if i < len(synsets):
            mapping[synsets[i]] = name

    Path(cache_path).write_text(json.dumps(mapping, indent=2))
    return mapping


def build_image_groups(
    val_labels_path: str,
    synsets_ordered: list[str],
    max_per_group: int = 1000,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Build image groups for each superclass.

    Returns dict: superclass_name → list of image filenames.
    """
    rng = np.random.RandomState(seed)

    # Load val labels: filename → synset
    file_to_synset: dict[str, str] = {}
    with open(val_labels_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                file_to_synset[parts[0]] = parts[1]

    # Build synset → class index mapping
    synset_to_idx: dict[str, int] = {}
    for i, s in enumerate(synsets_ordered):
        synset_to_idx[s] = i

    # Build class index → list of filenames
    class_to_files: dict[int, list[str]] = {}
    for fname, synset in file_to_synset.items():
        idx = synset_to_idx.get(synset)
        if idx is not None:
            class_to_files.setdefault(idx, []).append(fname)

    groups: dict[str, list[str]] = {}

    for name, spec in SUPERCLASS_DEFINITIONS.items():
        lo, hi = spec["class_range"]
        class_indices = set(range(lo, hi + 1))

        for extra_lo, extra_hi in spec.get("extra_ranges", []):
            class_indices.update(range(extra_lo, extra_hi + 1))

        # Handle exclusions
        if "exclude_range" in spec:
            ex_lo, ex_hi = spec["exclude_range"]
            exclude = set(range(ex_lo, ex_hi + 1))
            # For anti_dogs: start with full range, remove excluded
            if name == "anti_dogs":
                class_indices = set(range(0, 1000)) - set(range(151, 276))

        # Collect images for this group
        candidate_files = []
        for cls_idx in sorted(class_indices):
            candidate_files.extend(class_to_files.get(cls_idx, []))

        n_classes = len([c for c in class_indices if c in class_to_files])

        if spec.get("sample_mode") == "uniform" or name == "random_diverse":
            # Uniform random: pick max_per_group from all candidates
            if len(candidate_files) > max_per_group:
                idx = rng.choice(len(candidate_files), max_per_group, replace=False)
                candidate_files = [candidate_files[i] for i in sorted(idx)]
        elif name == "anti_dogs":
            if len(candidate_files) > max_per_group:
                idx = rng.choice(len(candidate_files), max_per_group, replace=False)
                candidate_files = [candidate_files[i] for i in sorted(idx)]
        else:
            # Take all images from the superclass (up to max)
            if len(candidate_files) > max_per_group:
                idx = rng.choice(len(candidate_files), max_per_group, replace=False)
                candidate_files = [candidate_files[i] for i in sorted(idx)]

        groups[name] = sorted(candidate_files)
        print(f"  {name:25s}: {len(candidate_files):5d} images from {n_classes:3d} classes "
              f"({spec['description']})")

    return groups


# ---------------------------------------------------------------------------
# Embedding Extraction (reused logic)
# ---------------------------------------------------------------------------

def extract_embeddings(
    model_info: dict,
    image_paths: list[str],
    device: str,
    batch_size: int = 64,
) -> np.ndarray | None:
    """Extract embeddings for all images through one model.

    Returns (N, D) float32 array, or None on failure.
    """
    import torch
    import timm
    from PIL import Image
    from torchvision import transforms

    name = model_info["model_name"]
    layer = model_info["layer"]
    pp = model_info["preprocess"]
    crop = pp["crop"]
    n = len(image_paths)

    # Adaptive batch size for large crop sizes
    if crop <= 224:
        bs = batch_size
    elif crop <= 256:
        bs = min(48, batch_size)
    elif crop <= 384:
        bs = min(16, batch_size)
    else:
        bs = min(8, batch_size)

    try:
        model = timm.create_model(name, pretrained=True)
        model.eval().to(device)
        if device == "cuda":
            model.half()

        transform = transforms.Compose([
            transforms.Resize(pp["resize"],
                              interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(crop),
            transforms.ToTensor(),
            transforms.Normalize(mean=pp["mean"], std=pp["std"]),
        ])

        modules = dict(model.named_modules())
        if layer not in modules:
            print(f"    WARNING: layer '{layer}' not found in {name}")
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
            return None

        features_buf = {}
        handle = modules[layer].register_forward_hook(
            lambda mod, inp, out: features_buf.update({"out": out})
        )

        all_features = []
        with torch.no_grad(), torch.amp.autocast(device_type=device, enabled=(device == "cuda")):
            for i in range(0, n, bs):
                batch_paths = image_paths[i:i + bs]
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
                    if "out of memory" in str(e).lower():
                        torch.cuda.empty_cache()
                        for si in range(len(imgs)):
                            sub = imgs[si].unsqueeze(0).to(device)
                            if device == "cuda":
                                sub = sub.half()
                            model(sub)
                            feat = features_buf["out"]
                            if feat.dim() > 2:
                                feat = feat.flatten(1)
                            all_features.append(feat.float().cpu().numpy())
                            del sub
                        torch.cuda.empty_cache()
                        continue
                    raise

                feat = features_buf["out"]
                if feat.dim() > 2:
                    feat = feat.flatten(1)
                all_features.append(feat.float().cpu().numpy())
                del batch
                if device == "cuda" and i % (bs * 4) == 0:
                    torch.cuda.empty_cache()

        handle.remove()
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

        if not all_features:
            return None

        return np.concatenate(all_features, axis=0).astype(np.float32)

    except Exception as e:
        print(f"    ERROR: {name}: {e}")
        if device == "cuda":
            import torch
            torch.cuda.empty_cache()
        gc.collect()
        return None


# ---------------------------------------------------------------------------
# CKA Computation
# ---------------------------------------------------------------------------

def compute_cka_matrix(embeddings: dict[str, np.ndarray]) -> tuple[np.ndarray, list[str]]:
    """Compute pairwise CKA matrix from a dict of model_name → (N, D) embeddings.

    Returns (M×M CKA matrix, list of model names).
    """
    names = sorted(embeddings.keys())
    M = len(names)
    vecs = []

    for name in names:
        E = embeddings[name].astype(np.float64)
        E -= E.mean(axis=0, keepdims=True)
        K = E @ E.T
        # Double-center
        rm = K.mean(axis=1, keepdims=True)
        cm = K.mean(axis=0, keepdims=True)
        gm = K.mean()
        K_c = K - rm - cm + gm
        v = K_c.ravel()
        norm = np.linalg.norm(v)
        if norm < 1e-12:
            vecs.append(np.zeros_like(v))
        else:
            vecs.append(v / norm)

    G = np.stack(vecs)
    C = G @ G.T
    np.clip(C, 0.0, 1.0, out=C)
    np.fill_diagonal(C, 1.0)
    return C, names


# ---------------------------------------------------------------------------
# Main Survey
# ---------------------------------------------------------------------------

def run_survey(
    image_groups: dict[str, list[str]],
    imagenet_val_dir: str,
    registry: list[dict],
    survey_models: list[str],
    device: str,
    cache_dir: str,
    n_images: int = 1000,
) -> dict:
    """Run the full survey: extract embeddings + compute CKA for each group."""

    os.makedirs(cache_dir, exist_ok=True)
    results = {}

    # Resolve model infos
    name_to_info = {m["model_name"]: m for m in registry}
    valid_models = [m for m in survey_models if m in name_to_info]
    print(f"\nSurvey models: {len(valid_models)}/{len(survey_models)} available")

    for group_name, filenames in sorted(image_groups.items()):
        if not filenames:
            print(f"\n--- {group_name}: SKIPPED (no images) ---")
            continue

        # Sample down to n_images if needed
        if len(filenames) > n_images:
            rng = np.random.RandomState(42)
            idx = rng.choice(len(filenames), n_images, replace=False)
            filenames = [filenames[i] for i in sorted(idx)]

        print(f"\n{'='*70}")
        print(f"Category: {group_name} ({len(filenames)} images)")
        print(f"{'='*70}")

        # Resolve to full paths
        image_paths = []
        for fname in filenames:
            # Try direct path, then search in subdirectories
            direct = os.path.join(imagenet_val_dir, fname)
            if os.path.exists(direct):
                image_paths.append(direct)
            else:
                # Search in synset subfolders
                found = False
                for root, _, files in os.walk(imagenet_val_dir):
                    if fname in files:
                        image_paths.append(os.path.join(root, fname))
                        found = True
                        break
                if not found:
                    image_paths.append(direct)  # will fail gracefully

        # Check cache for this group
        cache_path = os.path.join(cache_dir, f"embeddings_{group_name}.npz")
        embeddings = {}

        if os.path.exists(cache_path):
            data = np.load(cache_path, allow_pickle=True)
            cached_models = list(data.get("model_names", []))
            cached_n = int(data.get("n_images", 0))
            if cached_n == len(image_paths) and set(cached_models) >= set(valid_models):
                print(f"  Using cached embeddings ({len(cached_models)} models)")
                for mname in valid_models:
                    if mname in data:
                        embeddings[mname] = data[mname]

        # Extract missing models
        missing = [m for m in valid_models if m not in embeddings]
        if missing:
            print(f"  Extracting {len(missing)} models...")
            for i, model_name in enumerate(missing):
                info = name_to_info[model_name]
                t0 = time.time()
                print(f"    [{i+1}/{len(missing)}] {model_name}...", end="", flush=True)
                emb = extract_embeddings(info, image_paths, device)
                dt = time.time() - t0
                if emb is not None and emb.shape[0] == len(image_paths):
                    embeddings[model_name] = emb
                    print(f" OK ({emb.shape}, {dt:.1f}s)")
                else:
                    print(f" FAILED ({dt:.1f}s)")

                # Free model weights
                import shutil
                hf_home = os.environ.get("HF_HOME",
                    os.path.join(os.path.expanduser("~"), ".cache", "huggingface"))
                cache = os.path.join(hf_home, "hub", f"models--timm--{model_name}")
                if os.path.isdir(cache):
                    try:
                        shutil.rmtree(cache)
                    except OSError:
                        pass

            # Save cache
            save_dict = {mname: emb for mname, emb in embeddings.items()}
            save_dict["model_names"] = np.array(list(embeddings.keys()))
            save_dict["n_images"] = len(image_paths)
            save_dict["filenames"] = np.array(filenames[:20])  # sample for verification
            np.savez_compressed(cache_path, **save_dict)

        # Compute CKA
        if len(embeddings) < 2:
            print(f"  Only {len(embeddings)} models succeeded, skipping CKA")
            continue

        C, model_names = compute_cka_matrix(embeddings)
        M = len(model_names)
        mean_cka = (C.sum() - M) / (M * (M - 1))
        score = 1.0 - mean_cka

        # Find most/least divergent pairs
        pairs = []
        for i in range(M):
            for j in range(i + 1, M):
                pairs.append((C[i, j], model_names[i], model_names[j]))
        pairs.sort()

        print(f"\n  === {group_name} Results ===")
        print(f"  Models:         {M}")
        print(f"  Images:         {len(filenames)}")
        print(f"  Mean CKA:       {mean_cka:.6f}")
        print(f"  RED TEAM SCORE: {score:.6f}")
        print(f"\n  Most divergent pairs:")
        for cka, m1, m2 in pairs[:5]:
            print(f"    {cka:.4f}  {m1[:40]} <-> {m2[:40]}")
        print(f"  Most aligned pairs:")
        for cka, m1, m2 in pairs[-5:]:
            print(f"    {cka:.4f}  {m1[:40]} <-> {m2[:40]}")

        results[group_name] = {
            "n_images": len(filenames),
            "n_models": M,
            "mean_cka": float(mean_cka),
            "score": float(score),
            "min_cka_pair": (float(pairs[0][0]), pairs[0][1], pairs[0][2]),
            "max_cka_pair": (float(pairs[-1][0]), pairs[-1][1], pairs[-1][2]),
            "cka_matrix": C.tolist(),
            "model_names": model_names,
        }

        del embeddings
        gc.collect()

    return results


def print_summary(results: dict, class_names: dict[str, str] | None = None):
    """Print final summary ranking all categories by divergence."""
    print("\n" + "=" * 70)
    print(" SURVEY SUMMARY — Category Ranking by Divergence")
    print("=" * 70)
    print(f"\n  {'Category':<25s} {'Images':>6s} {'Models':>6s} "
          f"{'Mean CKA':>10s} {'Score':>10s}")
    print("  " + "-" * 63)

    ranked = sorted(results.items(), key=lambda x: x[1]["mean_cka"])
    for name, r in ranked:
        print(f"  {name:<25s} {r['n_images']:>6d} {r['n_models']:>6d} "
              f"{r['mean_cka']:>10.6f} {r['score']:>10.6f}")

    best = ranked[0]
    worst = ranked[-1]
    print(f"\n  BEST for Red Team:  {best[0]} (score={best[1]['score']:.4f})")
    print(f"  WORST for Red Team: {worst[0]} (score={worst[1]['score']:.4f})")
    print(f"  Gap: {best[1]['score'] - worst[1]['score']:.4f}")

    # Cross-category model divergence analysis
    print(f"\n  === Per-Model Analysis ===")
    all_model_ckas: dict[str, list[float]] = {}
    for name, r in results.items():
        C = np.array(r["cka_matrix"])
        mnames = r["model_names"]
        M = len(mnames)
        for i, m in enumerate(mnames):
            row_mean = (C[i].sum() - 1.0) / (M - 1)
            all_model_ckas.setdefault(m, []).append((name, row_mean))

    # Find models that consistently have low CKA (most divergent)
    model_avg_cka = {}
    for m, entries in all_model_ckas.items():
        model_avg_cka[m] = np.mean([v for _, v in entries])

    print(f"\n  {'Model':<50s} {'Avg CKA across categories':>25s}")
    print("  " + "-" * 75)
    for m, avg in sorted(model_avg_cka.items(), key=lambda x: x[1]):
        print(f"  {m:<50s} {avg:>25.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Red Team Survey — find best superclass for divergence"
    )
    parser.add_argument("--imagenet-val-dir", type=str, default=None)
    parser.add_argument("--auto-download", action="store_true")
    parser.add_argument("--hf-token", type=str, default=None)
    parser.add_argument("--download-dir", type=str, default="downloaded_imagenet_survey")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "cpu", "mps"])
    parser.add_argument("--registry", type=str, default=REGISTRY_PATH)
    parser.add_argument("--n-images", type=int, default=1000,
                        help="Max images per category (default: 1000)")
    parser.add_argument("--survey-dir", type=str, default=SURVEY_DIR)
    parser.add_argument("--phase", type=str, default="all",
                        choices=["all", "report"],
                        help="'all' = extract+report, 'report' = re-analyze cached results")
    parser.add_argument("--categories", type=str, nargs="+", default=None,
                        help="Only run specific categories (e.g. dogs snakes insects)")
    args = parser.parse_args()

    os.makedirs(args.survey_dir, exist_ok=True)

    # --- Report-only mode ---
    results_path = os.path.join(args.survey_dir, "survey_results.json")
    if args.phase == "report" and os.path.exists(results_path):
        results = json.loads(Path(results_path).read_text())
        class_names = download_class_names(args.survey_dir)
        print_summary(results, class_names)
        return

    # --- Device ---
    if args.device == "auto":
        import torch
        if torch.cuda.is_available():
            device = "cuda"
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
            print("WARNING: No GPU — will be very slow")
    else:
        device = args.device

    # --- Registry ---
    registry = json.loads(Path(args.registry).read_text())
    print(f"Registry: {len(registry)} models")

    # --- Synset labels ---
    synsets = download_synset_labels(args.survey_dir)
    class_names = download_class_names(args.survey_dir)

    # --- Val labels ---
    val_labels_path = os.path.join(args.survey_dir, "imagenet_val_labels.txt")
    if not os.path.exists(val_labels_path):
        # download_synset_labels already creates this
        synsets = download_synset_labels(args.survey_dir)

    # --- Image source ---
    imagenet_val_dir = args.imagenet_val_dir

    if not imagenet_val_dir or not os.path.isdir(str(imagenet_val_dir)):
        if args.auto_download:
            print("\nAuto-downloading full ImageNet val for survey...")
            print("(Survey needs all classes, not just dogs)")
            try:
                from datasets import load_dataset
                dl_dir = args.download_dir
                os.makedirs(dl_dir, exist_ok=True)

                # Check cache
                existing = list(Path(dl_dir).rglob("*.JPEG"))
                if len(existing) >= 40000:
                    print(f"  Found {len(existing)} cached images")
                    imagenet_val_dir = dl_dir
                else:
                    print("  Streaming from HuggingFace imagenet-1k...")
                    ds = load_dataset("ILSVRC/imagenet-1k", split="validation",
                                      streaming=True, token=args.hf_token,
                                      trust_remote_code=True)
                    count = 0
                    for i, row in enumerate(ds):
                        img = row["image"].convert("RGB")
                        fname = f"ILSVRC2012_val_{i+1:08d}.JPEG"
                        out = os.path.join(dl_dir, fname)
                        img.save(out)
                        count += 1
                        if count % 5000 == 0:
                            print(f"    {count}/50000 images...")
                    print(f"  Downloaded {count} images")
                    imagenet_val_dir = dl_dir
            except Exception as e:
                print(f"  Download failed: {e}")
                sys.exit(1)
        else:
            print("ERROR: Provide --imagenet-val-dir or --auto-download")
            sys.exit(1)

    # --- Build image groups ---
    print("\nBuilding image groups...")
    groups = build_image_groups(val_labels_path, synsets, max_per_group=args.n_images)

    # Filter to requested categories
    if args.categories:
        groups = {k: v for k, v in groups.items() if k in args.categories}
        print(f"\nFiltered to {len(groups)} categories: {list(groups.keys())}")

    # --- Run survey ---
    t0 = time.time()
    results = run_survey(
        groups, imagenet_val_dir, registry, SURVEY_MODELS,
        device, args.survey_dir, args.n_images,
    )
    dt = time.time() - t0
    print(f"\nSurvey complete in {dt/60:.1f} minutes")

    # Save results
    Path(results_path).write_text(json.dumps(results, indent=2, default=str))
    print(f"Results saved: {results_path}")

    # Print summary
    print_summary(results, class_names)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()
