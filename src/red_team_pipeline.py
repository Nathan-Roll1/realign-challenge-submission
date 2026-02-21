#!/usr/bin/env python3
"""
Red Team Pipeline — ICLR 2026 Re-Align Hackathon

Selects exactly 1000 images from imagenet_val + objectnet to MINIMIZE
average pairwise CKA across the fixed set of ~141 vision models.
Score = 1 - avg_CKA  (higher divergence ranks higher).

Strategy:
  1. Extract embeddings for ALL candidate images through 12 diverse proxy models
  2. Score each image by cross-model representation divergence
  3. Pre-compute Gram matrices for top candidates
  4. Optimize selection via greedy initialization + simulated annealing
  5. Validate with full CKA computation
  6. Submit to HuggingFace

Computational requirements (A100 GPU):
  Phase 1 (extract):   ~25 min  — 100K images × 12 models, batch=64
  Phase 2 (score):     ~5 min   — reference-profile divergence on CPU
  Phase 3 (select):    ~30 min  — SA with 200K iterations
  Phase 4 (validate):  ~2 min   — CKA on 1000 images
  Total:               ~60 min

Usage:
  # Full pipeline
  python red_team_pipeline.py \\
      --imagenet-val-dir /data/imagenet_val \\
      --objectnet-dir /data/objectnet

  # Resume from specific phase
  python red_team_pipeline.py --phase score
  python red_team_pipeline.py --phase select
  python red_team_pipeline.py --phase validate

  # Submit to HuggingFace
  python red_team_pipeline.py --phase submit \\
      --hf-token hf_xxx --hf-repo USER/red-team-v1
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
import time
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REGISTRY_PATH = "configs_blue_team_model_registry.txt"

# Working directories (created automatically)
EMBED_DIR = "red_team_embeddings"
WORK_DIR = "red_team_work"

# 12 proxy models chosen for maximum architectural diversity, all crop=224
# for uniform batch processing. Spans: CNN, ViT, CLIP, DINO, MAE, MLP-Mixer,
# Mamba, ConvMixer, XCiT, DenseNet — covering every major inductive bias.
PROXY_MODELS = [
    # --- Pure CNNs (3): texture bias, local features ---
    "vgg11.tv_in1k",                              # shallow classic CNN
    "resnet101.a1_in1k",                          # deep ResNet + skip connections
    "densenet121.ra_in1k",                        # dense connections, feature reuse
    # --- Pure Vision Transformers (3): global attention, shape bias ---
    "deit3_base_patch16_224.fb_in1k",             # DeiT3, supervised ViT
    "beit_base_patch16_224.in22k_ft_in22k",       # BEiT, masked image modeling pretrain
    "xcit_large_24_p16_224.fb_dist_in1k",         # XCiT, cross-covariance attention
    # --- CLIP / vision-language (2): semantic organization ---
    "vit_base_mci_224.apple_mclip",               # Apple MCLIP
    "vitamin_base_224.datacomp1b_clip",            # DataComp CLIP
    # --- Self-supervised (2): different training signals ---
    "resmlp_12_224.fb_dino",                      # DINO (contrastive, clusters)
    "hiera_base_224.mae",                         # MAE (reconstructive)
    # --- Exotic architectures (2): unusual inductive biases ---
    "mambaout_base.in1k",                         # Mamba/SSM-derived
    "mixer_b16_224.goog_in21k",                   # MLP-Mixer (fixed token mixing)
]

# How many top candidates to keep for Gram-based optimization
N_CANDIDATES = 5000

# ---------------------------------------------------------------------------
# Superclass Strategy — Dog Breed Synsets
# ---------------------------------------------------------------------------
# ImageNet-1K dog breed synsets (0-indexed classes 151-268 = 118 breeds)
# plus wild canids (wolves, coyote, dingo, dhole, African wild dog) for
# a total of ~125 classes × 50 images = ~6,250 candidates.
#
# The mathematical insight: CKA uses centered Gram matrices. When images span
# many dissimilar classes, the block-diagonal between-class structure dominates
# and ALL models agree on it → high CKA. Within a single superclass (dogs),
# the between-class signal vanishes, leaving only within-class variation
# (pose, texture, background) where different architectures DISAGREE → low CKA.

DOG_SYNSETS = {
    # Toy breeds
    "n02085620", "n02085782", "n02085936", "n02086079", "n02086240",
    "n02086646", "n02086910", "n02087046",
    # Hounds
    "n02087394", "n02088094", "n02088238", "n02088364", "n02088466",
    "n02088632", "n02089078", "n02089867", "n02089973", "n02090379",
    "n02090622", "n02090721", "n02091032", "n02091134", "n02091244",
    "n02091467", "n02091635", "n02091831", "n02092002", "n02092339",
    # Terriers
    "n02093256", "n02093428", "n02093647", "n02093754", "n02093859",
    "n02093991", "n02094114", "n02094258", "n02094433", "n02095314",
    "n02095570", "n02095889", "n02096051", "n02096177", "n02096294",
    "n02096437", "n02096585", "n02097047", "n02097130", "n02097209",
    "n02097298", "n02097474", "n02097658",
    # Sporting / working / herding
    "n02098105", "n02098286", "n02098413", "n02099267", "n02099429",
    "n02099601", "n02099712", "n02099849", "n02100236", "n02100583",
    "n02100735", "n02100877", "n02101006", "n02101388", "n02101556",
    "n02102040", "n02102177", "n02102318", "n02102480", "n02102973",
    "n02104029", "n02104365", "n02105056", "n02105162", "n02105251",
    "n02105412", "n02105505", "n02105641", "n02105855", "n02106030",
    "n02106166", "n02106382", "n02106550", "n02106662", "n02107142",
    "n02107312", "n02107574", "n02107683", "n02107908", "n02108000",
    "n02108089", "n02108422", "n02108551", "n02108915", "n02109047",
    "n02109525", "n02109961", "n02110063", "n02110185", "n02110341",
    "n02110627", "n02110806", "n02110958", "n02111129", "n02111277",
    "n02111500", "n02111889", "n02112018", "n02112137", "n02112350",
    "n02112706", "n02113023", "n02113186", "n02113624", "n02113712",
    "n02113799", "n02113978",
    # Wild canids (visually similar to dogs, no strong between-class signal)
    "n02114367", "n02114548", "n02114712", "n02114855",  # wolves, coyote
    "n02115641", "n02115913", "n02116738",                # dingo, dhole, African wild dog
}

# Other fine-grained superclasses (available as alternatives)
BIRD_SYNSETS_RANGE = ("n01514668", "n01860187")  # approximate range

SUPERCLASS_OPTIONS = {
    "dogs": DOG_SYNSETS,
    "none": None,
}

# Simulated annealing parameters
# Per-iteration cost: ~50ms (laptop) / ~25ms (cloud server) — dominated by
# 12 × (1000x1000) Gram centering. On a cloud GPU server with fast CPU:
#   50K  iterations → ~20 min    (quick, captures ~80% of SA benefit)
#   100K iterations → ~40 min    (recommended default)
#   200K iterations → ~80 min    (diminishing returns)
SA_ITERATIONS = 100_000
SA_T_START = 0.005
SA_T_END = 1e-7


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class ImageInfo:
    """Tracks an image's filesystem path and submission identifier."""
    path: str               # local path for loading
    dataset_name: str       # "imagenet_val" or "objectnet"
    image_identifier: str   # identifier for submission


# ---------------------------------------------------------------------------
# Phase 0: Load Candidate Images
# ---------------------------------------------------------------------------

def detect_synset_from_path(filepath: str) -> str | None:
    """Extract the ImageNet synset ID (e.g. 'n02085620') from a file path.

    Works when images are organized in synset folders like:
        .../val/n02085620/ILSVRC2012_val_00000044.JPEG
    """
    parts = Path(filepath).parts
    for part in parts:
        if len(part) == 9 and part.startswith("n") and part[1:].isdigit():
            return part
    return None


def download_imagenet_val_labels(cache_dir: str = "red_team_work") -> dict[str, int] | None:
    """Download ImageNet validation labels: filename → 0-indexed class ID.

    Tries multiple sources. Returns a dict mapping e.g.
    'ILSVRC2012_val_00000001.JPEG' → 65 (class index).
    """
    import urllib.request

    os.makedirs(cache_dir, exist_ok=True)
    labels_path = os.path.join(cache_dir, "imagenet_val_labels.txt")

    if os.path.exists(labels_path):
        labels = {}
        with open(labels_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    labels[parts[0]] = int(parts[1])
        if labels:
            return labels

    # Try downloading the ground truth (1-indexed class IDs, one per line)
    gt_urls = [
        "https://raw.githubusercontent.com/tensorflow/models/master/research/slim/datasets/imagenet_2012_validation_synset_labels.txt",
    ]

    for url in gt_urls:
        try:
            print(f"  Downloading val labels from {url[:80]}...")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                content = resp.read().decode("utf-8").strip().split("\n")
            if len(content) == 50000:
                # This file contains synset IDs, one per image
                labels = {}
                for i, synset in enumerate(content):
                    fname = f"ILSVRC2012_val_{i+1:08d}.JPEG"
                    labels[fname] = synset.strip()
                # Save for caching
                with open(labels_path, "w") as f:
                    for fname, synset in sorted(labels.items()):
                        f.write(f"{fname} {synset}\n")
                print(f"  Downloaded labels for {len(labels)} images")
                return labels
        except Exception as e:
            print(f"  Failed to download from {url[:60]}: {e}")
            continue

    return None


def filter_to_superclass(
    images: list[ImageInfo],
    synset_set: set[str],
    val_labels: dict[str, str] | None = None,
) -> list[ImageInfo]:
    """Filter images to those belonging to a specific superclass.

    Uses two strategies:
      1. Synset detected from folder structure (images in synset dirs)
      2. Downloaded val labels (flat image directories)

    Args:
        images: full image list
        synset_set: set of synset IDs to keep (e.g. DOG_SYNSETS)
        val_labels: optional dict mapping filename → synset ID

    Returns: filtered image list
    """
    filtered = []
    match_method = {"folder": 0, "labels": 0, "objectnet_skip": 0, "no_match": 0}

    for img in images:
        if img.dataset_name == "objectnet":
            match_method["objectnet_skip"] += 1
            continue

        # Method 1: detect synset from folder structure
        synset = detect_synset_from_path(img.path)
        if synset:
            if synset in synset_set:
                filtered.append(img)
                match_method["folder"] += 1
            continue

        # Method 2: use downloaded labels
        if val_labels:
            fname = img.image_identifier
            synset = val_labels.get(fname)
            if synset and synset in synset_set:
                filtered.append(img)
                match_method["labels"] += 1
            continue

        match_method["no_match"] += 1

    print(f"  Superclass filtering: {len(filtered)} images kept from {len(images)}")
    print(f"    Matched by folder structure: {match_method['folder']}")
    print(f"    Matched by downloaded labels: {match_method['labels']}")
    print(f"    ObjectNet skipped: {match_method['objectnet_skip']}")
    print(f"    No match info: {match_method['no_match']}")
    return filtered


def download_dog_images_hf(
    hf_token: str | None,
    output_dir: str = "downloaded_imagenet_dogs",
    synset_set: set[str] | None = None,
) -> str | None:
    """Download ONLY dog breed images from HuggingFace imagenet-1k (streaming).

    Instead of downloading all 50K val images (~6.7GB), streams and saves only
    dog breeds (~6,250 images, ~800MB). Uses the 'label' column to filter.

    ImageNet-1K class indices for dogs/canids: 151-275 (0-indexed).
    Returns the output directory path, or None on failure.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Check if already downloaded
    existing = list(Path(output_dir).rglob("*.JPEG")) + list(Path(output_dir).rglob("*.jpeg"))
    if len(existing) >= 5000:
        print(f"  Found {len(existing)} cached dog images in {output_dir}")
        return output_dir

    # Dog breed label range in ImageNet-1K (0-indexed)
    # 151-268 = domestic dogs, 269-275 = wolves/canids
    dog_labels = set(range(151, 276))

    try:
        from datasets import load_dataset
        print(f"  Downloading dog breed images from HuggingFace imagenet-1k...")
        print(f"  (streaming — only downloading ~6,250 dog images, not all 50K)")

        ds = load_dataset(
            "ILSVRC/imagenet-1k",
            split="validation",
            streaming=True,
            token=hf_token,
            trust_remote_code=True,
        )

        count = 0
        for i, row in enumerate(ds):
            label = row["label"]
            if label not in dog_labels:
                continue

            img = row["image"]
            if not hasattr(img, "save"):
                continue

            img = img.convert("RGB")
            # Use the standard ImageNet val naming convention
            fname = f"ILSVRC2012_val_{i+1:08d}.JPEG"
            # Organize into synset-like folders for superclass detection
            synset_dir = os.path.join(output_dir, f"label_{label:04d}")
            os.makedirs(synset_dir, exist_ok=True)
            out_path = os.path.join(synset_dir, fname)
            img.save(out_path)
            count += 1

            if count % 500 == 0:
                print(f"    {count} dog images saved...")

        print(f"  Downloaded {count} dog breed images → {output_dir}")
        return output_dir if count >= 1000 else None

    except Exception as e:
        print(f"  HuggingFace download failed: {e}")
        print(f"  You may need to accept the imagenet-1k terms at:")
        print(f"  https://huggingface.co/datasets/ILSVRC/imagenet-1k")
        return None


def download_imagenet_val_full(
    output_dir: str,
    imagenet_username: str | None = None,
    imagenet_password: str | None = None,
) -> str | None:
    """Download full ImageNet validation set from image-net.org.

    Requires ImageNet credentials. Returns output dir path or None on failure.
    """
    import urllib.request
    import tarfile

    username = imagenet_username or os.environ.get("IMAGENET_USERNAME")
    password = imagenet_password or os.environ.get("IMAGENET_PASSWORD")
    if not username or not password:
        print("  ImageNet credentials not available.")
        print("  Set IMAGENET_USERNAME and IMAGENET_PASSWORD env vars,")
        print("  or use --imagenet-username / --imagenet-password flags.")
        return None

    os.makedirs(output_dir, exist_ok=True)
    tar_path = os.path.join(output_dir, "ILSVRC2012_img_val.tar")
    val_url = "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar"

    if not os.path.exists(tar_path):
        print(f"  Downloading ImageNet val from {val_url}...")
        try:
            password_mgr = urllib.request.HTTPPasswordMgrWithDefaultRealm()
            password_mgr.add_password(None, val_url, username, password)
            handler = urllib.request.HTTPBasicAuthHandler(password_mgr)
            opener = urllib.request.build_opener(handler)
            urllib.request.install_opener(opener)
            urllib.request.urlretrieve(val_url, tar_path)
        except Exception as e:
            print(f"  Download failed: {e}")
            return None

    print(f"  Extracting {tar_path}...")
    try:
        with tarfile.open(tar_path, "r") as tar:
            tar.extractall(output_dir)
        os.remove(tar_path)
        print(f"  Extracted to {output_dir}")
        return output_dir
    except Exception as e:
        print(f"  Extraction failed: {e}")
        return None


def load_candidate_images(imagenet_val_dir: str | None,
                          objectnet_dir: str | None,
                          catalog_path: str | None = None,
                          auto_download: bool = False,
                          hf_token: str | None = None,
                          download_dir: str | None = None,
                          ) -> list[ImageInfo]:
    """Discover all candidate images from both datasets.

    Extracts the correct submission identifiers:
      - imagenet_val:  filename only (e.g. "ILSVRC2012_val_00000001.JPEG")
      - objectnet:     relative path (e.g. "objectnet-1.0/images/cup/abc.png")

    If auto_download is True and no local images found, downloads dog breed
    images from HuggingFace (streaming, ~800MB instead of full 6.7GB).
    """
    EXTS = {'.jpg', '.jpeg', '.png', '.JPEG', '.JPG', '.PNG', '.bmp', '.webp'}
    images: list[ImageInfo] = []

    if imagenet_val_dir and os.path.isdir(imagenet_val_dir):
        count = 0
        for root, _, files in os.walk(imagenet_val_dir):
            for f in sorted(files):
                if Path(f).suffix in EXTS:
                    full_path = os.path.join(root, f)
                    images.append(ImageInfo(
                        path=full_path,
                        dataset_name="imagenet_val",
                        image_identifier=f,
                    ))
                    count += 1
        print(f"  Found {count:,} ImageNet-val images")

    if objectnet_dir and os.path.isdir(objectnet_dir):
        count = 0
        objectnet_root = os.path.abspath(objectnet_dir)
        for root, _, files in os.walk(objectnet_dir):
            for f in sorted(files):
                if Path(f).suffix in EXTS:
                    full_path = os.path.join(root, f)
                    rel = os.path.relpath(full_path, objectnet_root)
                    # The identifier should start with "objectnet-1.0/..."
                    # Adjust based on actual directory structure
                    if not rel.startswith("objectnet"):
                        # If objectnet_dir already contains objectnet-1.0/
                        parent_name = os.path.basename(objectnet_root)
                        if parent_name.startswith("objectnet"):
                            rel = os.path.join(parent_name, rel)
                        else:
                            rel = os.path.join("objectnet-1.0", rel)
                    images.append(ImageInfo(
                        path=full_path,
                        dataset_name="objectnet",
                        image_identifier=rel,
                    ))
                    count += 1
        print(f"  Found {count:,} ObjectNet images")

    # Auto-download if no local images found
    if not images and auto_download:
        dl_dir = download_dir or "downloaded_imagenet_dogs"
        print(f"\n  No local images found. Auto-downloading dog breeds...")

        # Try HuggingFace streaming first (only downloads dogs, ~800MB)
        result = download_dog_images_hf(hf_token, output_dir=dl_dir)
        if result:
            imagenet_val_dir = result
            for root, _, files in os.walk(result):
                for f in sorted(files):
                    if Path(f).suffix in EXTS:
                        full_path = os.path.join(root, f)
                        images.append(ImageInfo(
                            path=full_path,
                            dataset_name="imagenet_val",
                            image_identifier=f,
                        ))
            print(f"  Loaded {len(images):,} downloaded images")
        else:
            # Fallback: try full ImageNet val download
            result = download_imagenet_val_full(dl_dir)
            if result:
                for root, _, files in os.walk(result):
                    for f in sorted(files):
                        if Path(f).suffix in EXTS:
                            full_path = os.path.join(root, f)
                            images.append(ImageInfo(
                                path=full_path,
                                dataset_name="imagenet_val",
                                image_identifier=f,
                            ))
                print(f"  Loaded {len(images):,} downloaded images")

    if not images:
        raise ValueError(
            "No images found. Options:\n"
            "  --imagenet-val-dir PATH    (local ImageNet val)\n"
            "  --objectnet-dir PATH       (local ObjectNet)\n"
            "  --auto-download            (download dog breeds from HuggingFace)\n"
            "\n"
            "For HuggingFace download, you may need to:\n"
            "  1. Accept terms at https://huggingface.co/datasets/ILSVRC/imagenet-1k\n"
            "  2. Provide --hf-token hf_..."
        )

    print(f"  Total candidate images: {len(images):,}")
    return images


def save_image_manifest(images: list[ImageInfo], path: str):
    """Save image list to JSON for reproducibility and phase resumption."""
    rows = [asdict(img) for img in images]
    Path(path).write_text(json.dumps(rows, indent=1))
    print(f"  Saved image manifest ({len(images):,} images) → {path}")


def load_image_manifest(path: str) -> list[ImageInfo]:
    """Reload image list from a saved manifest."""
    rows = json.loads(Path(path).read_text())
    images = [ImageInfo(**r) for r in rows]
    print(f"  Loaded image manifest: {len(images):,} images")
    return images


# ---------------------------------------------------------------------------
# Phase 1: Embedding Extraction
# ---------------------------------------------------------------------------

def get_model_info(model_name: str, registry: list[dict]) -> dict | None:
    """Look up a model's config from the registry."""
    for m in registry:
        if m["model_name"] == model_name:
            return m
    return None


def extract_embeddings_single_model(
    model_info: dict,
    image_paths: list[str],
    device: str,
    output_path: str,
    batch_size: int = 64,
) -> bool:
    """Extract embeddings for all images through one model. Saves to .npy.

    Returns True on success, False on failure.
    The output array has shape (N_images, embedding_dim), dtype float32.
    """
    import torch
    import timm
    from PIL import Image
    from torchvision import transforms

    if os.path.exists(output_path):
        emb = np.load(output_path, mmap_mode='r')
        if emb.shape[0] == len(image_paths):
            print(f"    Cached: {output_path} ({emb.shape})")
            return True
        print(f"    Stale cache ({emb.shape[0]} vs {len(image_paths)}), re-extracting")

    name = model_info["model_name"]
    layer = model_info["layer"]
    pp = model_info["preprocess"]
    crop = pp["crop"]
    n = len(image_paths)

    try:
        model = timm.create_model(name, pretrained=True)
        model.eval()
        model.to(device)
        if device == "cuda":
            model.half()

        transform = transforms.Compose([
            transforms.Resize(
                pp["resize"],
                interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.CenterCrop(crop),
            transforms.ToTensor(),
            transforms.Normalize(mean=pp["mean"], std=pp["std"]),
        ])

        # Register hook on target layer
        modules = dict(model.named_modules())
        if layer not in modules:
            print(f"    WARNING: layer '{layer}' not found in {name}")
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
            return False

        features_buf = {}

        def hook_fn(module, inp, out):
            features_buf["out"] = out

        handle = modules[layer].register_forward_hook(hook_fn)

        all_features = []
        with torch.no_grad(), torch.amp.autocast(device_type=device, enabled=(device == "cuda")):
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
                    if "out of memory" in str(e).lower():
                        torch.cuda.empty_cache()
                        half = max(1, len(imgs) // 2)
                        for si in range(0, len(imgs), half):
                            sub = torch.stack(imgs[si:si + half]).to(device)
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
                if device == "cuda" and i % (batch_size * 8) == 0:
                    torch.cuda.empty_cache()

                if (i // batch_size) % 100 == 0 and i > 0:
                    print(f"      {i:,}/{n:,} images processed")

        handle.remove()

        if not all_features:
            print(f"    ERROR: No features for {name}")
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
            return False

        X = np.concatenate(all_features, axis=0).astype(np.float32)
        np.save(output_path, X)
        print(f"    Saved: {output_path} → shape {X.shape}")

        del model, X, all_features
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        return True

    except Exception as e:
        print(f"    ERROR extracting {name}: {e}")
        if device == "cuda":
            import torch
            torch.cuda.empty_cache()
        gc.collect()
        return False


def cleanup_model_cache(model_name: str):
    """Remove model weights from HuggingFace cache to save disk space."""
    hf_home = os.environ.get(
        "HF_HOME",
        os.path.join(os.path.expanduser("~"), ".cache", "huggingface"),
    )
    cache_path = os.path.join(hf_home, "hub", f"models--timm--{model_name}")
    if os.path.isdir(cache_path):
        try:
            shutil.rmtree(cache_path)
        except OSError:
            pass


def run_extraction(registry: list[dict], images: list[ImageInfo],
                   proxy_models: list[str], device: str,
                   embed_dir: str) -> list[str]:
    """Extract embeddings for all proxy models. Returns list of successful model names."""
    os.makedirs(embed_dir, exist_ok=True)
    image_paths = [img.path for img in images]
    successful = []

    for i, model_name in enumerate(proxy_models):
        model_info = get_model_info(model_name, registry)
        if model_info is None:
            print(f"  [{i+1}/{len(proxy_models)}] {model_name} — NOT IN REGISTRY, skipping")
            continue

        out_path = os.path.join(embed_dir, f"{model_name}.npy")
        print(f"  [{i+1}/{len(proxy_models)}] {model_name}...")
        t0 = time.time()

        ok = extract_embeddings_single_model(
            model_info, image_paths, device, out_path,
            batch_size=64,
        )
        dt = time.time() - t0

        if ok:
            successful.append(model_name)
            print(f"    Done in {dt:.1f}s")
        else:
            print(f"    FAILED ({dt:.1f}s)")

        cleanup_model_cache(model_name)

    print(f"\n  Extraction complete: {len(successful)}/{len(proxy_models)} models")
    return successful


# ---------------------------------------------------------------------------
# Phase 2: Per-Image Divergence Scoring
# ---------------------------------------------------------------------------

def compute_divergence_scores(
    embed_dir: str,
    model_names: list[str],
    n_images: int,
    n_reference: int = 1000,
    seed: int = 42,
) -> np.ndarray:
    """Score each image by cross-model representation divergence.

    Algorithm (Reference-Profile Divergence):
      1. For each model, load embeddings and L2-normalize rows
      2. Sample a reference set of n_reference images
      3. For each model, compute similarity of every image to the reference
         set → (N, n_reference) profile matrix
      4. For each image, measure how much these profiles DISAGREE across
         models using mean pairwise (1 - Pearson correlation)

    Images with high divergence have similarity structures that differ
    across models — exactly what drives low CKA.

    Returns: divergence scores array of shape (n_images,), higher = more divergent
    """
    rng = np.random.RandomState(seed)
    ref_idx = rng.choice(n_images, min(n_reference, n_images), replace=False)

    M = len(model_names)
    print(f"  Computing divergence scores: {n_images:,} images × {M} models")
    print(f"  Reference set: {len(ref_idx)} images")

    # Compute similarity profiles for each model
    profiles = []  # list of (N, |ref|) arrays
    for model_name in model_names:
        emb_path = os.path.join(embed_dir, f"{model_name}.npy")
        E = np.load(emb_path).astype(np.float32)
        assert E.shape[0] == n_images, (
            f"{model_name}: expected {n_images} rows, got {E.shape[0]}"
        )

        # Center and L2-normalize
        E -= E.mean(axis=0, keepdims=True)
        norms = np.linalg.norm(E, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        E /= norms

        # Similarity to reference set: (N, d) @ (d, |ref|) → (N, |ref|)
        E_ref = E[ref_idx]
        sim = E @ E_ref.T  # cosine similarities
        profiles.append(sim)
        print(f"    {model_name}: emb_dim={E.shape[1]}, profile computed")
        del E

    # Compute per-image divergence: mean pairwise (1 - correlation) across models
    # For each model pair, compute row-wise Pearson correlation
    print(f"  Computing pairwise profile correlations ({M*(M-1)//2} pairs)...")
    divergence = np.zeros(n_images, dtype=np.float64)
    n_pairs = 0

    for a in range(M):
        for b in range(a + 1, M):
            Pa = profiles[a]  # (N, |ref|)
            Pb = profiles[b]

            # Row-wise Pearson correlation (vectorized)
            Pa_centered = Pa - Pa.mean(axis=1, keepdims=True)
            Pb_centered = Pb - Pb.mean(axis=1, keepdims=True)
            Pa_std = np.linalg.norm(Pa_centered, axis=1)
            Pb_std = np.linalg.norm(Pb_centered, axis=1)

            denom = Pa_std * Pb_std
            denom = np.maximum(denom, 1e-8)
            corr = np.sum(Pa_centered * Pb_centered, axis=1) / denom

            # Divergence contribution: 1 - correlation
            divergence += (1.0 - corr)
            n_pairs += 1

    divergence /= n_pairs  # average over model pairs
    del profiles

    # Print statistics
    print(f"\n  Divergence score statistics:")
    print(f"    Mean:   {divergence.mean():.4f}")
    print(f"    Std:    {divergence.std():.4f}")
    print(f"    Min:    {divergence.min():.4f}")
    print(f"    Max:    {divergence.max():.4f}")
    pcts = [50, 75, 90, 95, 99]
    vals = np.percentile(divergence, pcts)
    for p, v in zip(pcts, vals):
        print(f"    P{p:02d}:    {v:.4f}")

    return divergence


# ---------------------------------------------------------------------------
# Phase 3: CKA-Aware Selection Optimization
# ---------------------------------------------------------------------------

def precompute_gram_matrices(
    embed_dir: str,
    model_names: list[str],
    candidate_indices: np.ndarray,
) -> list[np.ndarray]:
    """Compute uncentered Gram matrices for candidate images.

    For each model: K[i,j] = e(i) · e(j)  (dot product of centered embeddings)
    Returns list of (N_cand, N_cand) float32 arrays.
    """
    N = len(candidate_indices)
    print(f"  Pre-computing {len(model_names)} Gram matrices ({N}×{N})...")
    grams = []

    for model_name in model_names:
        emb_path = os.path.join(embed_dir, f"{model_name}.npy")
        E_full = np.load(emb_path).astype(np.float32)
        E = E_full[candidate_indices]

        # Column-center the embeddings
        E -= E.mean(axis=0, keepdims=True)

        # Gram matrix
        K = E @ E.T
        grams.append(K)
        mem_mb = K.nbytes / 1e6
        print(f"    {model_name}: emb_dim={E.shape[1]}, gram {K.shape}, {mem_mb:.0f}MB")
        del E_full, E

    total_gb = sum(K.nbytes for K in grams) / 1e9
    print(f"  Total Gram storage: {total_gb:.2f} GB")
    return grams


def compute_avg_cka(grams: list[np.ndarray], indices: np.ndarray) -> float:
    """Compute mean pairwise CKA for a subset of images.

    Args:
        grams: list of M uncentered Gram matrices (N_cand × N_cand)
        indices: 1D array of indices into the candidate set (size K)

    Returns:
        Mean off-diagonal CKA across all model pairs
    """
    M = len(grams)
    idx = indices
    vecs = []

    for K_full in grams:
        # Extract sub-Gram matrix
        K_sub = K_full[np.ix_(idx, idx)]
        # Center
        row_mean = K_sub.mean(axis=1, keepdims=True)
        col_mean = K_sub.mean(axis=0, keepdims=True)
        grand_mean = K_sub.mean()
        K_c = K_sub - row_mean - col_mean + grand_mean
        # Flatten and normalize
        v = K_c.ravel().astype(np.float64)
        norm = np.linalg.norm(v)
        if norm < 1e-12:
            v = np.zeros_like(v)
        else:
            v /= norm
        vecs.append(v)

    # CKA matrix via cosine similarity of vectorized centered Grams
    G = np.stack(vecs)  # (M, K^2)
    C = G @ G.T         # (M, M) CKA matrix
    np.clip(C, 0.0, 1.0, out=C)

    # Mean off-diagonal
    mean_cka = (C.sum() - np.trace(C)) / (M * (M - 1))
    return float(mean_cka)


def greedy_initial_selection(
    grams: list[np.ndarray],
    divergence_scores: np.ndarray,
    n_select: int = 1000,
    n_greedy_from_top: int = 200,
) -> np.ndarray:
    """Build initial selection: top divergence + greedy CKA-minimizing additions.

    Strategy:
      1. Start with top n_greedy_from_top images by divergence score
      2. Greedily add images that minimize avg CKA (evaluated every 50 additions)
      3. Fill remaining slots from top divergence scores with diversity sampling

    Returns: array of candidate indices (into the Gram matrices)
    """
    N_cand = grams[0].shape[0]
    n_select = min(n_select, N_cand)

    # Rank by divergence score (within candidate set)
    ranked = np.argsort(-divergence_scores)  # highest first

    # Phase A: Seed with top-scoring images
    seed_size = min(n_greedy_from_top, n_select)
    selected = list(ranked[:seed_size])
    remaining = set(range(N_cand)) - set(selected)

    print(f"    Seeded with top {seed_size} by divergence score")
    sel_arr = np.array(selected, dtype=np.int64)
    current_cka = compute_avg_cka(grams, sel_arr)
    print(f"    Initial CKA (seed): {current_cka:.6f}")

    # Phase B: Greedy additions, checking CKA periodically
    # Adding in chunks for speed (check CKA every chunk)
    chunk_size = 50
    while len(selected) < n_select:
        needed = n_select - len(selected)
        # Take next chunk from ranked list
        chunk = []
        for idx in ranked:
            if idx not in set(selected) and idx in remaining:
                chunk.append(idx)
                if len(chunk) >= min(chunk_size, needed):
                    break

        if not chunk:
            break

        selected.extend(chunk)
        remaining -= set(chunk)

        if len(selected) % 200 == 0 or len(selected) >= n_select:
            sel_arr = np.array(selected[:n_select], dtype=np.int64)
            cka = compute_avg_cka(grams, sel_arr)
            print(f"    {len(selected):,} images selected, CKA = {cka:.6f}")

    selected = selected[:n_select]
    return np.array(selected, dtype=np.int64)


def _cka_from_gram_subs(
    gram_subs: list[np.ndarray], M: int, N: int,
) -> float:
    """Fast CKA computation from maintained sub-Gram matrices.

    Uses the identity: mean pairwise CKA = (||S||^2 - M) / (M*(M-1))
    where S = sum of unit-normalized vectorized centered Gram matrices.
    This avoids the expensive (M, N^2) @ (N^2, M) matmul by accumulating
    a running sum vector and computing a single dot product.

    Stays in float32 for speed (~2x faster than float64 on large vectors).
    """
    N2 = N * N
    S = np.zeros(N2, dtype=np.float64)

    for m in range(M):
        # Copy to avoid modifying the maintained sub-Gram
        K = gram_subs[m].astype(np.float64, copy=True)
        # Center: K_c = H @ K @ H
        rm = K.mean(axis=1, keepdims=True)
        cm = K.mean(axis=0, keepdims=True)
        gm = K.mean()
        K -= rm
        K -= cm
        K += gm
        # Normalize to unit length and accumulate
        v = K.ravel()
        norm = np.linalg.norm(v)
        if norm > 1e-12:
            v /= norm
            S += v

    # mean CKA = (||S||^2 - M) / (M*(M-1))
    mean_cka = (np.dot(S, S) - M) / (M * (M - 1))
    return float(np.clip(mean_cka, 0.0, 1.0))


def simulated_annealing_selection(
    grams: list[np.ndarray],
    initial_selection: np.ndarray,
    n_candidates: int,
    n_select: int = 1000,
    n_iter: int = SA_ITERATIONS,
    T_start: float = SA_T_START,
    T_end: float = SA_T_END,
    seed: int = 0,
    log_interval: int = 5000,
) -> tuple[np.ndarray, float]:
    """Optimized simulated annealing to minimize avg CKA by swapping images.

    Key optimization: maintains sub-Gram matrices incrementally. Instead of
    re-extracting 1000x1000 submatrices from 5000x5000 via expensive fancy
    indexing on each iteration, we update only the affected row and column
    when a swap is proposed (~12K ops vs ~12M ops per iteration).

    Returns: (best_selection_indices, best_cka)
    """
    rng = np.random.RandomState(seed)
    M = len(grams)
    N = n_select

    # Initialize ordered selection and complement
    sel_list = list(initial_selection[:N])
    sel_set = set(sel_list)
    unsel_list = [i for i in range(n_candidates) if i not in sel_set]

    # Build initial sub-Gram matrices (one-time fancy index)
    sel_arr = np.array(sel_list, dtype=np.int64)
    gram_subs = [K[np.ix_(sel_arr, sel_arr)].astype(np.float32).copy()
                 for K in grams]

    # Pre-allocate row backup buffers
    old_rows = [np.empty(N, dtype=np.float32) for _ in range(M)]

    # Compute initial CKA
    current_cka = _cka_from_gram_subs(gram_subs, M, N)
    best_cka = current_cka
    best_sel_list = list(sel_list)

    alpha = (T_end / T_start) ** (1.0 / max(n_iter, 1))
    T = T_start
    accepted = 0
    improved = 0
    t0 = time.time()

    print(f"  SA: {n_iter:,} iterations, T={T_start:.4f}→{T_end:.1e}")
    print(f"  Initial CKA: {current_cka:.6f}  (target: minimize)")

    for it in range(n_iter):
        # Pick random position to swap out and random candidate to swap in
        p = rng.randint(N)
        ui = rng.randint(len(unsel_list))
        v_out = sel_list[p]
        v_in = unsel_list[ui]

        # Update selection array (tentative)
        sel_arr[p] = v_in

        # Update sub-Gram matrices: replace row and column p
        for m in range(M):
            old_rows[m][:] = gram_subs[m][p, :]
            new_row = grams[m][v_in, sel_arr]  # (N,) — uses updated sel_arr
            gram_subs[m][p, :] = new_row
            gram_subs[m][:, p] = new_row

        # Compute CKA on updated sub-Grams
        new_cka = _cka_from_gram_subs(gram_subs, M, N)
        delta = new_cka - current_cka

        if delta < 0 or rng.rand() < np.exp(-delta / max(T, 1e-15)):
            # Accept swap
            sel_list[p] = v_in
            sel_set.discard(v_out)
            sel_set.add(v_in)
            unsel_list[ui] = v_out
            current_cka = new_cka
            accepted += 1
            if new_cka < best_cka:
                best_cka = new_cka
                best_sel_list = list(sel_list)
                improved += 1
        else:
            # Reject: restore sub-Gram matrices and sel_arr
            sel_arr[p] = v_out
            for m in range(M):
                gram_subs[m][p, :] = old_rows[m]
                gram_subs[m][:, p] = old_rows[m]

        T *= alpha

        if (it + 1) % log_interval == 0:
            elapsed = time.time() - t0
            rate = (it + 1) / elapsed
            eta = (n_iter - it - 1) / rate
            print(
                f"    [{it+1:>7,}/{n_iter:,}] "
                f"CKA={current_cka:.6f} best={best_cka:.6f} "
                f"T={T:.2e} acc={accepted}/{it+1} imp={improved} "
                f"({rate:.0f} it/s, ETA {eta:.0f}s)"
            )

    elapsed = time.time() - t0
    print(f"\n  SA complete: {n_iter:,} iterations in {elapsed:.1f}s "
          f"({n_iter/elapsed:.0f} it/s)")
    print(f"  Best CKA: {best_cka:.6f}  (score = {1 - best_cka:.6f})")
    print(f"  Accepted: {accepted:,}  Improved: {improved:,}")

    return np.array(best_sel_list, dtype=np.int64), best_cka


def local_search_polish(
    grams: list[np.ndarray],
    selection: np.ndarray,
    n_candidates: int,
    max_no_improve: int = 500,
    max_iter: int = 10000,
    sample_size: int = 30,
) -> tuple[np.ndarray, float]:
    """Steepest-descent local search with incremental Gram updates.

    For each position in the selection, tries swapping with a sample of
    unselected candidates. Accepts the first improvement found.
    Terminates after max_no_improve consecutive failures.
    """
    M = len(grams)
    N = len(selection)

    sel_list = list(selection)
    sel_set = set(sel_list)
    unsel_list = [i for i in range(n_candidates) if i not in sel_set]

    # Build sub-Gram matrices
    sel_arr = np.array(sel_list, dtype=np.int64)
    gram_subs = [K[np.ix_(sel_arr, sel_arr)].astype(np.float32).copy()
                 for K in grams]

    best_cka = _cka_from_gram_subs(gram_subs, M, N)
    best_sel_list = list(sel_list)
    no_improve = 0
    total_iter = 0
    total_improved = 0

    print(f"  Local search: starting CKA = {best_cka:.6f}")
    rng = np.random.RandomState(42)

    while no_improve < max_no_improve and total_iter < max_iter:
        total_iter += 1
        p = total_iter % N  # cycle through positions
        v_out = sel_list[p]

        # Sample candidates to try
        n_try = min(sample_size, len(unsel_list))
        try_indices = rng.choice(len(unsel_list), n_try, replace=False)

        found_better = False
        for ui in try_indices:
            v_in = unsel_list[ui]

            # Tentative swap
            sel_arr[p] = v_in
            old_rows = [gram_subs[m][p, :].copy() for m in range(M)]
            for m in range(M):
                new_row = grams[m][v_in, sel_arr]
                gram_subs[m][p, :] = new_row
                gram_subs[m][:, p] = new_row

            new_cka = _cka_from_gram_subs(gram_subs, M, N)

            if new_cka < best_cka - 1e-8:
                # Accept
                sel_list[p] = v_in
                sel_set.discard(v_out)
                sel_set.add(v_in)
                unsel_list[ui] = v_out
                best_cka = new_cka
                best_sel_list = list(sel_list)
                found_better = True
                no_improve = 0
                total_improved += 1
                break
            else:
                # Revert
                sel_arr[p] = v_out
                for m in range(M):
                    gram_subs[m][p, :] = old_rows[m]
                    gram_subs[m][:, p] = old_rows[m]

        if not found_better:
            no_improve += 1

        if total_iter % 200 == 0:
            print(f"    iter {total_iter}: CKA={best_cka:.6f}, "
                  f"no_improve={no_improve}/{max_no_improve}, "
                  f"improved={total_improved}")

    print(f"  Local search: {total_iter} iters, {total_improved} improvements, "
          f"final CKA = {best_cka:.6f}")
    return np.array(best_sel_list, dtype=np.int64), best_cka


# ---------------------------------------------------------------------------
# Phase 4: Validation
# ---------------------------------------------------------------------------

def validate_selection(
    embed_dir: str,
    model_names: list[str],
    image_indices: np.ndarray,
    all_model_names: list[str] | None = None,
) -> dict:
    """Compute full CKA matrix for the selected images across proxy models.

    Returns a dict with CKA statistics and the full CKA matrix.
    """
    M = len(model_names)
    N = len(image_indices)
    print(f"  Validating: {N} images × {M} models")

    # Load embeddings and compute centered Gram matrices
    vecs = []
    valid_names = []
    for model_name in model_names:
        emb_path = os.path.join(embed_dir, f"{model_name}.npy")
        if not os.path.exists(emb_path):
            continue
        E_full = np.load(emb_path, mmap_mode='r')
        E = E_full[image_indices].astype(np.float64)

        # Center
        E -= E.mean(axis=0, keepdims=True)

        # Gram matrix (already centered because E is centered)
        K = E @ E.T

        # Double-center the Gram matrix (centering is idempotent but ensures correctness)
        n = K.shape[0]
        rm = K.mean(axis=1, keepdims=True)
        cm = K.mean(axis=0, keepdims=True)
        gm = K.mean()
        K_c = K - rm - cm + gm

        v = K_c.ravel()
        norm = np.linalg.norm(v)
        if norm < 1e-12:
            continue
        vecs.append(v / norm)
        valid_names.append(model_name)
        del E_full, E

    G = np.stack(vecs)
    C = G @ G.T
    np.clip(C, 0.0, 1.0, out=C)
    np.fill_diagonal(C, 1.0)

    M_valid = len(valid_names)
    mean_cka = (C.sum() - M_valid) / (M_valid * (M_valid - 1))
    score = 1.0 - mean_cka

    # Per-pair analysis
    pairs = []
    for i in range(M_valid):
        for j in range(i + 1, M_valid):
            pairs.append((C[i, j], valid_names[i], valid_names[j]))
    pairs.sort()

    print(f"\n  === CKA Validation Results ===")
    print(f"  Models validated:  {M_valid}")
    print(f"  Images:            {N}")
    print(f"  Mean pairwise CKA: {mean_cka:.6f}")
    print(f"  RED TEAM SCORE:    {score:.6f}")
    print(f"\n  Lowest CKA pairs (most divergent):")
    for cka, m1, m2 in pairs[:10]:
        print(f"    {cka:.4f}  {m1} <-> {m2}")
    print(f"\n  Highest CKA pairs (most aligned):")
    for cka, m1, m2 in pairs[-10:]:
        print(f"    {cka:.4f}  {m1} <-> {m2}")

    return {
        "mean_cka": mean_cka,
        "score_1_minus_cka": score,
        "cka_matrix": C,
        "model_names": valid_names,
        "n_images": N,
    }


# ---------------------------------------------------------------------------
# Phase 5: Submission
# ---------------------------------------------------------------------------

def generate_submission_rows(
    images: list[ImageInfo],
    selected_indices: np.ndarray,
) -> list[dict]:
    """Generate submission rows with dataset_name and image_identifier."""
    rows = []
    for idx in sorted(selected_indices):
        img = images[idx]
        rows.append({
            "dataset_name": img.dataset_name,
            "image_identifier": img.image_identifier,
        })
    return rows


def push_to_huggingface(rows: list[dict], hf_token: str, repo_name: str):
    """Push Red Team submission to HuggingFace as a private dataset."""
    from datasets import Dataset
    from huggingface_hub import login

    login(token=hf_token)
    ds = Dataset.from_list(rows)
    ds.push_to_hub(repo_name, private=True)
    url = f"https://huggingface.co/datasets/{repo_name}"
    print(f"  Submission pushed: {url}")
    return url


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Red Team Pipeline — select 1000 images to maximize "
                    "representational divergence (minimize avg CKA)"
    )

    # Image sources
    parser.add_argument("--imagenet-val-dir", type=str, default=None,
                        help="Path to ImageNet validation images")
    parser.add_argument("--objectnet-dir", type=str, default=None,
                        help="Path to ObjectNet images")
    parser.add_argument("--auto-download", action="store_true",
                        help="Auto-download dog breed images from HuggingFace if no local data")
    parser.add_argument("--download-dir", type=str, default=None,
                        help="Directory for downloaded images (default: downloaded_imagenet_dogs)")
    parser.add_argument("--imagenet-username", type=str, default=None,
                        help="ImageNet username (fallback download)")
    parser.add_argument("--imagenet-password", type=str, default=None,
                        help="ImageNet password (fallback download)")

    # Computation
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "cpu", "mps"])
    parser.add_argument("--phase", type=str, default="all",
                        choices=["all", "extract", "score", "select",
                                 "validate", "submit"],
                        help="Which pipeline phase to run")

    # Proxy model configuration
    parser.add_argument("--n-proxy-models", type=int, default=12,
                        help="Number of proxy models (uses first N from list)")
    parser.add_argument("--proxy-models", type=str, nargs="+", default=None,
                        help="Override proxy model list")

    # Superclass strategy
    parser.add_argument("--superclass", type=str, default="dogs",
                        choices=list(SUPERCLASS_OPTIONS.keys()),
                        help="Superclass filter for image selection (default: dogs)")
    parser.add_argument("--no-superclass", action="store_true",
                        help="Disable superclass filtering (use all images)")

    # Selection parameters
    parser.add_argument("--n-select", type=int, default=1000,
                        help="Number of images to select (must be 1000 for submission)")
    parser.add_argument("--n-candidates", type=int, default=N_CANDIDATES,
                        help="Top-N candidates for Gram-based optimization")
    parser.add_argument("--sa-iterations", type=int, default=SA_ITERATIONS,
                        help="Simulated annealing iterations")

    # Output and submission
    parser.add_argument("--registry", type=str, default=REGISTRY_PATH)
    parser.add_argument("--embed-dir", type=str, default=EMBED_DIR)
    parser.add_argument("--work-dir", type=str, default=WORK_DIR)
    parser.add_argument("--output", type=str, default="red_team_submission.json")
    parser.add_argument("--hf-token", type=str, default=None)
    parser.add_argument("--hf-repo", type=str, default=None)

    args = parser.parse_args()

    os.makedirs(args.work_dir, exist_ok=True)
    os.makedirs(args.embed_dir, exist_ok=True)

    # Paths for intermediate results
    manifest_path = os.path.join(args.work_dir, "image_manifest.json")
    scores_path = os.path.join(args.work_dir, "divergence_scores.npy")
    candidates_path = os.path.join(args.work_dir, "candidate_indices.npy")
    selection_path = os.path.join(args.work_dir, "selection_indices.npy")
    models_path = os.path.join(args.work_dir, "successful_models.json")

    # Detect device
    if args.device == "auto":
        import torch
        if torch.cuda.is_available():
            device = "cuda"
            name = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"GPU: {name} ({mem:.1f} GB)")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
            print("Using Apple MPS")
        else:
            device = "cpu"
            print("WARNING: No GPU — extraction will be very slow")
    else:
        device = args.device

    # Load registry
    registry = json.loads(Path(args.registry).read_text())
    print(f"Model registry: {len(registry)} models")

    # Determine proxy models
    proxy_models = args.proxy_models or PROXY_MODELS[:args.n_proxy_models]
    print(f"Proxy models ({len(proxy_models)}):")
    for m in proxy_models:
        info = get_model_info(m, registry)
        status = "OK" if info else "NOT FOUND"
        print(f"  {m:50s} [{status}]")

    # =====================================================================
    # PHASE 1: Image Discovery + Embedding Extraction
    # =====================================================================
    if args.phase in ("all", "extract"):
        print("\n" + "=" * 70)
        print("PHASE 1: Image Discovery + Embedding Extraction")
        print("=" * 70)

        images = load_candidate_images(
            args.imagenet_val_dir,
            args.objectnet_dir,
            auto_download=args.auto_download,
            hf_token=args.hf_token,
            download_dir=args.download_dir,
        )

        # Apply superclass filtering
        superclass = None if args.no_superclass else args.superclass
        if superclass and superclass != "none":
            synset_set = SUPERCLASS_OPTIONS.get(superclass)
            if synset_set:
                print(f"\n  Applying superclass filter: '{superclass}' "
                      f"({len(synset_set)} synsets)")

                # Try folder-based detection first; fall back to downloading labels
                test_synset = detect_synset_from_path(images[0].path) if images else None
                val_labels = None
                if not test_synset:
                    print("  Images not in synset folders — downloading labels...")
                    val_labels = download_imagenet_val_labels(args.work_dir)
                    if not val_labels:
                        print("  WARNING: Could not get labels. "
                              "Proceeding without superclass filter.")
                        synset_set = None

                if synset_set:
                    images = filter_to_superclass(images, synset_set, val_labels)
                    if len(images) < args.n_select:
                        print(f"  WARNING: Only {len(images)} images after filtering "
                              f"(need {args.n_select}). Disabling superclass filter.")
                        images = load_candidate_images(
                            args.imagenet_val_dir, args.objectnet_dir,
                            auto_download=args.auto_download,
                            hf_token=args.hf_token,
                            download_dir=args.download_dir)
            else:
                print(f"  No superclass filter applied (superclass='{superclass}')")

        save_image_manifest(images, manifest_path)

        print(f"\n  Extracting embeddings to {args.embed_dir}/")
        t0 = time.time()
        successful = run_extraction(
            registry, images, proxy_models, device, args.embed_dir,
        )
        dt = time.time() - t0
        print(f"\n  Phase 1 complete: {dt / 60:.1f} minutes")

        Path(models_path).write_text(json.dumps(successful))

    # =====================================================================
    # PHASE 2: Divergence Scoring
    # =====================================================================
    if args.phase in ("all", "extract", "score"):
        print("\n" + "=" * 70)
        print("PHASE 2: Per-Image Divergence Scoring")
        print("=" * 70)

        # Load state from Phase 1
        images = load_image_manifest(manifest_path)
        if os.path.exists(models_path):
            successful = json.loads(Path(models_path).read_text())
        else:
            successful = [m for m in proxy_models
                          if os.path.exists(os.path.join(args.embed_dir, f"{m}.npy"))]

        t0 = time.time()
        scores = compute_divergence_scores(
            args.embed_dir, successful, len(images),
            n_reference=min(1000, len(images)),
        )
        np.save(scores_path, scores)
        dt = time.time() - t0
        print(f"\n  Phase 2 complete: {dt:.1f}s")

        # Save top candidates
        n_cand = min(args.n_candidates, len(images))
        candidate_idx = np.argsort(-scores)[:n_cand]
        np.save(candidates_path, candidate_idx)
        print(f"  Top {n_cand} candidates saved")

        # Show dataset distribution of top candidates
        ds_counts = {"imagenet_val": 0, "objectnet": 0}
        for ci in candidate_idx:
            ds_counts[images[ci].dataset_name] += 1
        print(f"  Dataset distribution of top-{n_cand}:")
        for ds, c in ds_counts.items():
            print(f"    {ds}: {c} ({100*c/n_cand:.1f}%)")

    # =====================================================================
    # PHASE 3: Selection Optimization
    # =====================================================================
    if args.phase in ("all", "extract", "score", "select"):
        print("\n" + "=" * 70)
        print("PHASE 3: CKA-Aware Selection Optimization")
        print("=" * 70)

        images = load_image_manifest(manifest_path)
        scores = np.load(scores_path)
        candidate_idx = np.load(candidates_path)
        if os.path.exists(models_path):
            successful = json.loads(Path(models_path).read_text())
        else:
            successful = [m for m in proxy_models
                          if os.path.exists(os.path.join(args.embed_dir, f"{m}.npy"))]

        n_cand = len(candidate_idx)
        n_select = min(args.n_select, n_cand)

        # Divergence scores within candidate set
        cand_scores = scores[candidate_idx]

        # Pre-compute Gram matrices for candidates
        t0 = time.time()
        grams = precompute_gram_matrices(
            args.embed_dir, successful, candidate_idx,
        )

        # Step A: Greedy initialization
        print(f"\n  --- Step A: Greedy Initialization ---")
        initial_sel = greedy_initial_selection(
            grams, cand_scores, n_select=n_select,
        )
        init_cka = compute_avg_cka(grams, initial_sel)
        print(f"  Greedy selection CKA: {init_cka:.6f}")

        # Step B: Simulated annealing
        print(f"\n  --- Step B: Simulated Annealing ---")
        sa_sel, sa_cka = simulated_annealing_selection(
            grams, initial_sel, n_candidates=n_cand,
            n_select=n_select, n_iter=args.sa_iterations,
        )

        # Step C: Local search polish
        print(f"\n  --- Step C: Local Search Polish ---")
        final_sel, final_cka = local_search_polish(
            grams, sa_sel, n_candidates=n_cand,
        )

        dt = time.time() - t0
        print(f"\n  Phase 3 complete: {dt / 60:.1f} minutes")
        print(f"  Final CKA:   {final_cka:.6f}")
        print(f"  Final Score: {1 - final_cka:.6f}")

        # Map back to global image indices
        global_selection = candidate_idx[final_sel]
        np.save(selection_path, global_selection)
        print(f"  Selection saved ({len(global_selection)} images)")

        # Dataset distribution
        ds_counts = {"imagenet_val": 0, "objectnet": 0}
        for gi in global_selection:
            ds_counts[images[gi].dataset_name] += 1
        print(f"  Dataset distribution of selection:")
        for ds, c in ds_counts.items():
            print(f"    {ds}: {c} ({100*c/len(global_selection):.1f}%)")

        del grams
        gc.collect()

    # =====================================================================
    # PHASE 4: Validation
    # =====================================================================
    if args.phase in ("all", "extract", "score", "select", "validate"):
        print("\n" + "=" * 70)
        print("PHASE 4: Full CKA Validation")
        print("=" * 70)

        images = load_image_manifest(manifest_path)
        global_selection = np.load(selection_path)
        if os.path.exists(models_path):
            successful = json.loads(Path(models_path).read_text())
        else:
            successful = [m for m in proxy_models
                          if os.path.exists(os.path.join(args.embed_dir, f"{m}.npy"))]

        results = validate_selection(
            args.embed_dir, successful, global_selection,
        )

        # Save submission
        submission_rows = generate_submission_rows(images, global_selection)
        submission_data = {
            "images": submission_rows,
            "metadata": {
                "n_images": len(submission_rows),
                "predicted_mean_cka": results["mean_cka"],
                "predicted_score": results["score_1_minus_cka"],
                "proxy_models": successful,
                "sa_iterations": args.sa_iterations,
                "n_candidates": args.n_candidates,
            },
        }
        with open(args.output, "w") as f:
            json.dump(submission_data, f, indent=2)
        print(f"\n  Submission JSON saved → {args.output}")
        print(f"  Contains {len(submission_rows)} images")

    # =====================================================================
    # PHASE 5: Submit to HuggingFace
    # =====================================================================
    if args.phase in ("all", "submit"):
        print("\n" + "=" * 70)
        print("PHASE 5: HuggingFace Submission")
        print("=" * 70)

        if not args.hf_token or not args.hf_repo:
            print("  Skipping (no --hf-token / --hf-repo)")
            print("  To submit:")
            print(f"    python red_team_pipeline.py --phase submit \\")
            print(f"      --hf-token YOUR_TOKEN --hf-repo USER/red-team-v1")
        else:
            if not os.path.exists(args.output):
                print(f"  ERROR: {args.output} not found")
                sys.exit(1)

            with open(args.output) as f:
                data = json.load(f)

            rows = data["images"]
            assert len(rows) == 1000, f"Expected 1000 rows, got {len(rows)}"

            url = push_to_huggingface(rows, args.hf_token, args.hf_repo)
            print(f"\n  Dataset URL: {url}")
            print("  Paste this URL in the Red Team submission tab.")

    print("\nDone!")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()
