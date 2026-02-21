#!/usr/bin/env python3
"""
Blue Team Optimizer for ICLR 2026 Re-Align Hackathon.

Two-phase approach:
  Phase 1 (zero-compute): Analytical model selection from registry metadata
  Phase 2 (with GPU): Compute CKA matrix on proxy images, then run combinatorial optimization

The optimization problem:
  Given 141 models, select 20 to maximize mean pairwise CKA.
  Equivalent to finding the densest k-subgraph in a weighted complete graph.

Papers informing the approach:
  - Cortes et al. (2012): CKA ~ kernel alignment QP, spectral structure
  - Zhou et al. (2024): CKA = cosine similarity on unit sphere of Gram matrices
  - Lu et al. (2014): CKA-based clustering for tightest cluster identification
  - Kornblith et al. (2019): Original CKA paper, linear CKA formula
"""

import json
import numpy as np
from itertools import combinations
from pathlib import Path

# ---------------------------------------------------------------------------
# Phase 1: Analytical (Zero-Compute) Model Selection
# ---------------------------------------------------------------------------

def load_model_registry(path: str) -> list[dict]:
    data = json.loads(Path(path).read_text())
    return data


def extract_model_features(registry: list[dict]) -> dict:
    """Extract features relevant to CKA alignment from model metadata."""
    features = {}
    for model in registry:
        name = model["model_name"]
        preprocess = model.get("preprocess", {})
        features[name] = {
            "source": model.get("source", ""),
            "weights": model.get("weights", ""),
            "layer": model.get("layer", ""),
            "embedding": model.get("embedding", ""),
            "resize": preprocess.get("resize", 0),
            "crop": preprocess.get("crop", 0),
            "mean": tuple(preprocess.get("mean", [])),
            "std": tuple(preprocess.get("std", [])),
            "arch_family": classify_architecture(name),
            "training_data": classify_training_data(name),
            "training_procedure": classify_training_procedure(name),
        }
    return features


def classify_architecture(name: str) -> str:
    """Classify model into architecture family from its name."""
    n = name.lower()
    if any(x in n for x in ["resnet", "resnext", "resnetv2", "resnetrs",
                             "resnetblur", "resnetaa", "wide_resnet",
                             "res2net", "resnest", "seresnet", "seresnext",
                             "skresnet", "skresnext", "ecaresnet",
                             "gcresnet", "gcresnext", "cspresnet", "cspresnext",
                             "tresnet"]):
        return "resnet_family"
    if any(x in n for x in ["darknet", "cspdarknet", "cs3darknet"]):
        return "darknet_family"
    if any(x in n for x in ["botnet", "halonet", "halo2bot", "lamhalobot",
                             "sebotnet", "sehalonet", "lambda_resnet",
                             "bat_resnext", "eca_botnext"]):
        return "attention_conv_hybrid"
    if any(x in n for x in ["densenet"]):
        return "densenet"
    if any(x in n for x in ["vgg"]):
        return "vgg"
    if any(x in n for x in ["efficientnet", "tf_efficientnet", "gc_efficientnet",
                             "tinynet"]):
        return "efficientnet"
    if any(x in n for x in ["mobilenet", "mobilenetv1", "mobileone",
                             "mobilevit", "mobilevitv2"]):
        return "mobilenet"
    if any(x in n for x in ["regnet", "regnety", "regnetx", "regnetz",
                             "nf_regnet", "haloregnetz"]):
        return "regnet"
    if any(x in n for x in ["nfnet", "dm_nfnet"]):
        return "nfnet"
    if any(x in n for x in ["ghostnet", "repghostnet"]):
        return "ghostnet"
    if any(x in n for x in ["fbnet", "mnasnet", "semnasnet", "spnasnet",
                             "hardcorenas"]):
        return "nas_mobile"
    if any(x in n for x in ["nasnetalarge", "pnasnet"]):
        return "nasnet_large"
    if any(x in n for x in ["senet", "legacy_senet"]):
        return "senet"
    if any(x in n for x in ["convnext"]):
        return "convnext"
    if any(x in n for x in ["poolformer", "caformer", "convformer"]):
        return "metaformer"
    if any(x in n for x in ["swin"]):
        return "swin"
    if any(x in n for x in ["vit_base", "flexivit", "deit", "beit",
                             "cait", "xcit", "crossvit", "pit_",
                             "volo", "tnt_", "twins", "pvt_",
                             "mvit", "convit", "nest_", "samvit",
                             "hiera", "sam2_hiera", "aimv2", "eva02",
                             "gcvit", "vitamin"]):
        return "transformer"
    if any(x in n for x in ["mixer", "gmixer", "gmlp", "resmlp"]):
        return "mlp_mixer"
    if any(x in n for x in ["coatnet", "coatnext", "maxvit", "maxxvit",
                             "efficientformer", "davit", "coat_lite",
                             "edgenext", "nextvit", "tiny_vit", "fastvit",
                             "repvit", "efficientvit", "levit", "visformer"]):
        return "hybrid_vit"
    if any(x in n for x in ["mambaout"]):
        return "mamba"
    if any(x in n for x in ["sequencer"]):
        return "lstm"
    if any(x in n for x in ["rexnet"]):
        return "rexnet"
    if any(x in n for x in ["convmixer"]):
        return "convmixer"
    return "other"


def classify_training_data(name: str) -> str:
    n = name.lower()
    if "clip" in n or "mclip" in n or "datacomp" in n:
        return "clip"
    if "sa1b" in n or "sam2" in n:
        return "sam"
    if "dino" in n:
        return "dino"
    if "mae" in n:
        return "mae"
    if "mim" in n:
        return "mim"
    if "fcmae" in n:
        return "fcmae"
    if "in22k" in n or "in21k" in n:
        return "in22k"
    if "in12k" in n:
        return "in12k"
    if "yfcc100m" in n:
        return "yfcc100m"
    return "in1k"


def classify_training_procedure(name: str) -> str:
    n = name.lower()
    if "dist" in n and "fb" in n:
        return "fb_distill"
    if "dist" in n and "snap" in n:
        return "snap_distill"
    if "dist" in n and "apple" in n:
        return "apple_distill"
    if "ssld" in n:
        return "ssld_distill"
    if "dist" in n:
        return "distill"
    if ".ra4_" in n:
        return "ra4"
    if ".ra3_" in n:
        return "ra3"
    if ".ra2_" in n:
        return "ra2"
    if ".ra_" in n:
        return "ra"
    if ".a1h_" in n:
        return "a1h"
    if ".a1_" in n:
        return "a1"
    if ".c1_" in n:
        return "c1"
    if ".c2ns_" in n:
        return "c2ns"
    if ".ch_" in n:
        return "ch"
    if ".sw_" in n:
        return "sw"
    if ".sail_" in n:
        return "sail"
    if ".rmsp_" in n:
        return "rmsp"
    if ".miil_" in n:
        return "miil"
    if ".gluon_" in n:
        return "gluon"
    if ".tv_" in n or ".tv2_" in n:
        return "torchvision"
    if ".tf_" in n:
        return "tf_port"
    if ".pycls_" in n:
        return "pycls"
    if ".agc_" in n:
        return "agc"
    return "vanilla"


def compute_analytical_similarity(f1: dict, f2: dict) -> float:
    """Estimate CKA similarity between two models from metadata alone.

    This is a heuristic proxy for actual CKA, based on research findings:
    1. Architecture family is the #1 predictor (Sirikova & Chan 2026)
    2. Training data/procedure is #2 (Wu et al. 2025)
    3. Preprocessing alignment is #3
    """
    score = 0.0

    # Architecture family match (strongest signal)
    if f1["arch_family"] == f2["arch_family"]:
        score += 0.40
    else:
        # Partial credit for related families
        related = {
            frozenset({"resnet_family", "darknet_family"}): 0.15,
            frozenset({"resnet_family", "attention_conv_hybrid"}): 0.20,
            frozenset({"resnet_family", "densenet"}): 0.12,
            frozenset({"resnet_family", "senet"}): 0.25,
            frozenset({"resnet_family", "rexnet"}): 0.10,
            frozenset({"resnet_family", "nfnet"}): 0.12,
            frozenset({"resnet_family", "regnet"}): 0.12,
            frozenset({"darknet_family", "attention_conv_hybrid"}): 0.12,
            frozenset({"efficientnet", "nas_mobile"}): 0.15,
            frozenset({"efficientnet", "mobilenet"}): 0.12,
            frozenset({"nas_mobile", "mobilenet"}): 0.12,
            frozenset({"ghostnet", "mobilenet"}): 0.10,
            frozenset({"metaformer", "convnext"}): 0.10,
            frozenset({"transformer", "swin"}): 0.10,
            frozenset({"transformer", "hybrid_vit"}): 0.08,
        }
        pair = frozenset({f1["arch_family"], f2["arch_family"]})
        score += related.get(pair, 0.0)

    # Training data match
    if f1["training_data"] == f2["training_data"]:
        score += 0.15
    elif {f1["training_data"], f2["training_data"]} <= {"in1k", "in12k", "in22k"}:
        score += 0.08  # All ImageNet variants

    # Training procedure match
    if f1["training_procedure"] == f2["training_procedure"]:
        score += 0.10
    else:
        # Partial credit for related procedures
        ra_variants = {"ra", "ra2", "ra3", "ra4"}
        if f1["training_procedure"] in ra_variants and f2["training_procedure"] in ra_variants:
            score += 0.06
        timm_recipes = {"a1", "a1h", "c1", "c2ns", "ch", "sw"}
        if f1["training_procedure"] in timm_recipes and f2["training_procedure"] in timm_recipes:
            score += 0.04

    # Preprocessing match
    if f1["mean"] == f2["mean"] and f1["std"] == f2["std"]:
        score += 0.08
    if f1["crop"] == f2["crop"]:
        score += 0.03
    if f1["resize"] == f2["resize"]:
        score += 0.02

    # Layer type match (indicates similar architecture output structure)
    if f1["layer"] == f2["layer"]:
        score += 0.05

    # Normalize to [0, 1] range (max possible: 0.40+0.15+0.10+0.08+0.03+0.02+0.05 = 0.83)
    return min(score / 0.83, 1.0)


def build_analytical_cka_matrix(features: dict) -> tuple[np.ndarray, list[str]]:
    """Build a proxy CKA matrix from analytical similarity scores."""
    names = sorted(features.keys())
    n = len(names)
    S = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            sim = compute_analytical_similarity(features[names[i]], features[names[j]])
            S[i, j] = sim
            S[j, i] = sim
    return S, names


# ---------------------------------------------------------------------------
# Phase 2: Combinatorial Optimization (works with any CKA matrix)
# ---------------------------------------------------------------------------

def greedy_densest_k(S: np.ndarray, k: int) -> tuple[list[int], float]:
    """Greedy algorithm: iteratively add model maximizing sum of CKA."""
    n = S.shape[0]
    # Initialize with the best pair
    off_diag = S.copy()
    np.fill_diagonal(off_diag, -np.inf)
    best_pair = np.unravel_index(np.argmax(off_diag), S.shape)
    selected = list(best_pair)
    remaining = set(range(n)) - set(selected)

    while len(selected) < k:
        best_score = -np.inf
        best_idx = -1
        for v in remaining:
            score = S[v, selected].sum()
            if score > best_score:
                best_score = score
                best_idx = v
        selected.append(best_idx)
        remaining.remove(best_idx)

    return selected, _obj(S, selected, k)


def spectral_densest_k(S: np.ndarray, k: int, n_restarts: int = 200) -> tuple[list[int], float]:
    """Spectral method: use top eigenvectors + randomized rounding."""
    from scipy.linalg import eigh
    eigenvalues, eigenvectors = eigh(S)

    best_obj = -np.inf
    best_set = None

    # Top eigenvector method
    v1 = eigenvectors[:, -1]
    top_k = np.argsort(np.abs(v1))[-k:]
    obj = _obj(S, top_k.tolist(), k)
    if obj > best_obj:
        best_obj = obj
        best_set = top_k.tolist()

    # Randomized rounding from top-r eigenvectors
    for r in [1, 2, 3, 5, 10, 20]:
        if r > S.shape[0]:
            continue
        V = eigenvectors[:, -r:]
        for _ in range(n_restarts):
            w = np.random.randn(r)
            scores = V @ w
            top_k = np.argsort(scores)[-k:]
            obj = _obj(S, top_k.tolist(), k)
            if obj > best_obj:
                best_obj = obj
                best_set = top_k.tolist()

    return best_set, best_obj


def truncated_power_method(S: np.ndarray, k: int, max_iter: int = 1000) -> tuple[list[int], float]:
    """Truncated power method for sparse leading eigenvector."""
    n = S.shape[0]
    x = np.random.randn(n)
    x = x / np.linalg.norm(x)

    for _ in range(max_iter):
        x_new = S @ x
        if k < n:
            threshold = np.sort(np.abs(x_new))[-k]
            x_new[np.abs(x_new) < threshold] = 0
        norm = np.linalg.norm(x_new)
        if norm < 1e-12:
            x = np.random.randn(n)
            x = x / np.linalg.norm(x)
            continue
        x_new = x_new / norm
        if np.linalg.norm(x_new - x) < 1e-8:
            break
        x = x_new

    selected = np.argsort(np.abs(x))[-k:]
    return selected.tolist(), _obj(S, selected.tolist(), k)


def frank_wolfe_dks(S: np.ndarray, k: int, lam: float = 1.0,
                    max_iter: int = 500) -> tuple[list[int], float]:
    """Frank-Wolfe with diagonal loading (Lu et al., AAAI 2025)."""
    n = S.shape[0]
    S_diag = S + lam * np.eye(n)

    x = np.ones(n) * (k / n)

    for t in range(max_iter):
        grad = S_diag @ x
        s = np.zeros(n)
        top_k = np.argpartition(grad, -k)[-k:]
        s[top_k] = 1.0
        d = s - x
        fw_gap = grad @ d
        if fw_gap <= 1e-10:
            break
        L = np.linalg.norm(S_diag, ord=2)
        gamma = min(1.0, fw_gap / (L * np.linalg.norm(d) ** 2 + 1e-12))
        x = x + gamma * d

    selected = np.argpartition(x, -k)[-k:]
    return selected.tolist(), _obj(S, selected.tolist(), k)


def local_search(S: np.ndarray, k: int, initial_set: list[int],
                 max_no_improve: int = 500) -> tuple[list[int], float]:
    """Vectorized steepest-ascent local search with 1-swap neighborhood."""
    n = S.shape[0]
    sel = np.array(sorted(initial_set[:k]))
    best_obj = _obj(S, sel.tolist(), k)
    best_set = sel.tolist()
    no_improve = 0

    while no_improve < max_no_improve:
        unsel = np.setdiff1d(np.arange(n), sel)
        # For each selected model, compute its total CKA to other selected
        S_sel = S[np.ix_(sel, sel)]
        out_scores = S_sel.sum(axis=1) - 1.0  # subtract self

        # For each unselected model, compute its total CKA to all selected
        in_scores_full = S[np.ix_(unsel, sel)]  # (n-k, k)

        found_improvement = False
        best_delta = 0
        best_out_idx = -1
        best_in_idx = -1

        for oi in range(k):
            loss = out_scores[oi]
            # If we remove sel[oi], gains for each unselected model
            gains = in_scores_full.sum(axis=1) - in_scores_full[:, oi]
            deltas = gains - loss
            max_idx = np.argmax(deltas)
            if deltas[max_idx] > best_delta:
                best_delta = deltas[max_idx]
                best_out_idx = oi
                best_in_idx = max_idx
                found_improvement = True

        if found_improvement:
            sel[best_out_idx] = unsel[best_in_idx]
            sel.sort()
            obj = _obj(S, sel.tolist(), k)
            if obj > best_obj:
                best_obj = obj
                best_set = sel.tolist()
            no_improve = 0
        else:
            no_improve += 1
            # Random perturbation
            oi = np.random.randint(k)
            unsel = np.setdiff1d(np.arange(n), sel)
            ii = np.random.randint(len(unsel))
            sel[oi] = unsel[ii]
            sel.sort()

    return best_set, best_obj


def simulated_annealing(S: np.ndarray, k: int, initial_set: list[int],
                        T_start: float = 0.05, T_end: float = 1e-6,
                        n_iter: int = 500000) -> tuple[list[int], float]:
    """Simulated annealing with 1-swap moves."""
    n = S.shape[0]
    in_set = np.zeros(n, dtype=bool)
    for i in initial_set[:k]:
        in_set[i] = True

    current_obj = _obj(S, list(np.where(in_set)[0]), k)
    best_obj = current_obj
    best_set = np.where(in_set)[0].tolist()

    alpha = (T_end / T_start) ** (1.0 / n_iter)
    T = T_start

    for _ in range(n_iter):
        sel = np.where(in_set)[0]
        unsel = np.where(~in_set)[0]
        v_out = np.random.choice(sel)
        v_in = np.random.choice(unsel)

        T_minus = sel[sel != v_out]
        gain = S[v_in, T_minus].sum() - S[v_out, T_minus].sum()
        delta_obj = gain * 2 / (k * (k - 1))

        if delta_obj > 0 or np.random.rand() < np.exp(delta_obj / max(T, 1e-15)):
            in_set[v_out] = False
            in_set[v_in] = True
            current_obj += delta_obj
            if current_obj > best_obj:
                best_obj = current_obj
                best_set = np.where(in_set)[0].tolist()

        T *= alpha

    return best_set, best_obj


def _obj(S: np.ndarray, selected: list[int], k: int) -> float:
    """Compute the mean pairwise similarity for a selection."""
    sub = S[np.ix_(selected, selected)]
    return (sub.sum() - k) / (k * (k - 1))


def comprehensive_optimize(S: np.ndarray, k: int = 20,
                           model_names: list[str] | None = None,
                           verbose: bool = True) -> tuple[list[int], float]:
    """Run all optimization algorithms and return the best result."""
    results = {}

    # 1. Row-sum heuristic
    row_means = (S.sum(axis=1) - 1) / (S.shape[0] - 1)
    top_k = np.argsort(row_means)[-k:]
    results["rowsum"] = (top_k.tolist(), _obj(S, top_k.tolist(), k))

    # 2. Greedy
    results["greedy"] = greedy_densest_k(S, k)

    # 3. Spectral
    results["spectral"] = spectral_densest_k(S, k, n_restarts=300)

    # 4. Truncated power
    results["truncated_power"] = truncated_power_method(S, k)

    # 5. Frank-Wolfe with multiple lambda values
    best_fw = (-np.inf, None)
    for lam in [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]:
        sel, obj = frank_wolfe_dks(S, k, lam=lam)
        if obj > best_fw[0]:
            best_fw = (obj, sel)
    results["frank_wolfe"] = (best_fw[1], best_fw[0])

    if verbose:
        print("\n=== Initial Algorithm Results ===")
        for method, (sel, obj) in sorted(results.items(), key=lambda x: -x[1][1]):
            print(f"  {method:20s}: {obj:.6f}")

    # 6. Local search refinement on all candidates
    refined = {}
    for method, (sel, obj) in results.items():
        sel_ls, obj_ls = local_search(S, k, sel, max_no_improve=5000)
        refined[f"{method}+LS"] = (sel_ls, obj_ls)

    if verbose:
        print("\n=== After Local Search ===")
        for method, (sel, obj) in sorted(refined.items(), key=lambda x: -x[1][1]):
            print(f"  {method:20s}: {obj:.6f}")

    # 7. Simulated annealing from best solutions + random restarts
    all_results = {**results, **refined}
    best_method = max(all_results, key=lambda m: all_results[m][1])
    best_init = all_results[best_method][0]

    sa_results = {}
    for i in range(5):
        init = best_init if i == 0 else np.random.choice(
            S.shape[0], k, replace=False).tolist()
        sel_sa, obj_sa = simulated_annealing(S, k, init, n_iter=200000)
        sel_sa, obj_sa = local_search(S, k, sel_sa, max_no_improve=200)
        sa_results[f"SA_restart_{i}"] = (sel_sa, obj_sa)

    if verbose:
        print("\n=== Simulated Annealing Results ===")
        for method, (sel, obj) in sorted(sa_results.items(), key=lambda x: -x[1][1]):
            print(f"  {method:20s}: {obj:.6f}")

    # Find overall best
    all_final = {**all_results, **sa_results}
    best_method = max(all_final, key=lambda m: all_final[m][1])
    best_sel, best_obj = all_final[best_method]

    if verbose:
        print(f"\n{'='*60}")
        print(f"BEST METHOD: {best_method}")
        print(f"BEST SCORE:  {best_obj:.6f}")
        if model_names:
            print(f"\nSelected models:")
            for idx in sorted(best_sel):
                print(f"  {model_names[idx]}")
        print(f"{'='*60}")

    return best_sel, best_obj


# ---------------------------------------------------------------------------
# Phase 2: Empirical CKA Computation (requires GPU + timm)
# ---------------------------------------------------------------------------

def compute_empirical_cka_matrix(registry_path: str,
                                 image_dir: str,
                                 n_images: int = 500,
                                 batch_size: int = 32,
                                 device: str = "cuda") -> tuple[np.ndarray, list[str]]:
    """Compute the actual CKA matrix by running all models on proxy images.

    Requires: torch, timm, PIL, torchvision
    """
    import torch
    import timm
    from PIL import Image
    from torchvision import transforms
    import glob

    registry = load_model_registry(registry_path)
    model_names = [m["model_name"] for m in registry]

    # Collect image paths
    image_paths = sorted(glob.glob(f"{image_dir}/**/*.JPEG", recursive=True) +
                         glob.glob(f"{image_dir}/**/*.jpg", recursive=True) +
                         glob.glob(f"{image_dir}/**/*.png", recursive=True))[:n_images]
    print(f"Using {len(image_paths)} images for CKA computation")

    # Extract embeddings for each model
    embeddings = {}
    for model_info in registry:
        name = model_info["model_name"]
        layer = model_info["layer"]
        pp = model_info["preprocess"]

        print(f"  Extracting embeddings from {name}...")
        try:
            model = timm.create_model(name, pretrained=True)
            model.eval()
            model.to(device)

            transform = transforms.Compose([
                transforms.Resize(pp["resize"]),
                transforms.CenterCrop(pp["crop"]),
                transforms.ToTensor(),
                transforms.Normalize(mean=pp["mean"], std=pp["std"]),
            ])

            all_features = []
            with torch.no_grad():
                for i in range(0, len(image_paths), batch_size):
                    batch_paths = image_paths[i:i + batch_size]
                    batch = torch.stack([
                        transform(Image.open(p).convert("RGB"))
                        for p in batch_paths
                    ]).to(device)

                    # Hook to extract intermediate layer output
                    features = {}
                    def hook_fn(module, input, output):
                        features["out"] = output

                    # Register hook on target layer
                    target_module = dict(model.named_modules())[layer]
                    handle = target_module.register_forward_hook(hook_fn)
                    model(batch)
                    handle.remove()

                    feat = features["out"]
                    if feat.dim() > 2:
                        feat = feat.flatten(1)
                    all_features.append(feat.cpu().numpy())

            embeddings[name] = np.concatenate(all_features, axis=0).astype(np.float64)
            del model
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"    ERROR: {e}")
            continue

    # Compute CKA matrix
    n = len(model_names)
    available = [name for name in model_names if name in embeddings]
    n_avail = len(available)
    S = np.eye(n_avail)

    for i in range(n_avail):
        for j in range(i + 1, n_avail):
            cka = linear_cka(embeddings[available[i]], embeddings[available[j]])
            S[i, j] = cka
            S[j, i] = cka
            if (i * n_avail + j) % 100 == 0:
                print(f"    CKA({available[i][:20]}, {available[j][:20]}) = {cka:.4f}")

    return S, available


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute linear CKA between two embedding matrices.

    Matches the evaluation server's biased CKA implementation.
    X: (n, p1), Y: (n, p2) — both float64, column-centered.
    """
    X = X.astype(np.float64)
    Y = Y.astype(np.float64)

    # Center columns
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)

    # Gram matrices
    K = X @ X.T
    L = Y @ Y.T

    # Center Gram matrices (biased HSIC)
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n

    KH = K @ H
    LH = L @ H

    hsic_kl = np.trace(KH @ LH)
    hsic_kk = np.trace(KH @ KH)
    hsic_ll = np.trace(LH @ LH)

    eps = 1e-6
    return float(hsic_kl / (np.sqrt(hsic_kk * hsic_ll) + eps))


# ---------------------------------------------------------------------------
# Main: Run analytical optimization
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    registry_path = "configs_blue_team_model_registry.txt"

    print("Loading model registry...")
    registry = load_model_registry(registry_path)
    print(f"  {len(registry)} models loaded")

    print("\nExtracting model features...")
    features = extract_model_features(registry)

    print("\nBuilding analytical CKA proxy matrix...")
    S, names = build_analytical_cka_matrix(features)

    print(f"\nMatrix shape: {S.shape}")
    print(f"Mean off-diagonal: {(S.sum() - len(names)) / (len(names) * (len(names) - 1)):.4f}")
    print(f"Max off-diagonal: {np.max(S - np.eye(len(names))):.4f}")

    print("\nRunning comprehensive optimization (k=20)...")
    best_sel, best_obj = comprehensive_optimize(S, k=20, model_names=names, verbose=True)

    # Output submission format
    print("\n\n=== SUBMISSION JSON ===")
    submission = []
    for idx in sorted(best_sel):
        model_name = names[idx]
        model_info = next(m for m in registry if m["model_name"] == model_name)
        submission.append({
            "model_name": model_name,
            "layer_name": model_info["layer"]
        })
    print(json.dumps(submission, indent=2))
