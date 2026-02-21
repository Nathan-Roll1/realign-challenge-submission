#!/usr/bin/env python3
"""
Cross-validate the CKA-based model selection.

Runs the full pipeline with multiple different subsets of proxy images
to verify that the optimal 20-model set is stable. If the same models
appear consistently across folds, we can be confident in the selection.

Usage:
  python cross_validate.py --gram-dir gram_matrices --n-folds 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np


def load_gram_matrices(gram_dir: str, registry: list[dict]
                       ) -> tuple[dict[str, np.ndarray], list[str]]:
    """Load all available centered Gram matrices."""
    grams = {}
    names = []
    for m in registry:
        name = m["model_name"]
        path = os.path.join(gram_dir, f"{name}.npy")
        if os.path.exists(path):
            grams[name] = np.load(path)
            names.append(name)
    return grams, names


def compute_cka_from_grams(grams: dict[str, np.ndarray],
                           names: list[str],
                           image_indices: np.ndarray | None = None
                           ) -> np.ndarray:
    """Compute CKA matrix, optionally using only a subset of images."""
    flat = []
    for name in names:
        K_c = grams[name]
        if image_indices is not None:
            K_full = K_c[np.ix_(image_indices, image_indices)]
            n = K_full.shape[0]
            rm = K_full.mean(axis=1, keepdims=True)
            cm = K_full.mean(axis=0, keepdims=True)
            gm = K_full.mean()
            K_c_sub = K_full - rm - cm + gm
            flat.append(K_c_sub.ravel())
        else:
            flat.append(K_c.ravel())

    G = np.stack(flat, axis=0)
    norms = np.linalg.norm(G, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    G_n = G / norms
    S = G_n @ G_n.T
    np.clip(S, 0.0, 1.0, out=S)
    np.fill_diagonal(S, 1.0)
    return S


def optimize_fast(S: np.ndarray, k: int = 20) -> tuple[list[int], float]:
    """Quick optimization: spectral + local search."""
    from scipy.linalg import eigh

    def obj(sel):
        sub = S[np.ix_(sel, sel)]
        return (sub.sum() - k) / (k * (k - 1))

    # Spectral
    _, evecs = eigh(S)
    best_obj = -np.inf
    best_set = None
    rng = np.random.RandomState(42)

    for r in [1, 2, 3, 5, 10, 20]:
        if r > S.shape[0]:
            continue
        V = evecs[:, -r:]
        for _ in range(200):
            w = rng.randn(r)
            scores = V @ w
            top_k = np.argsort(scores)[-k:]
            o = obj(top_k.tolist())
            if o > best_obj:
                best_obj = o
                best_set = top_k.tolist()

    # Local search
    n = S.shape[0]
    sel = np.array(sorted(best_set))
    best_obj = obj(sel.tolist())
    best_set = sel.tolist()
    min_delta = 1e-10
    for _ in range(3000):
        unsel = np.setdiff1d(np.arange(n), sel)
        S_sel = S[np.ix_(sel, sel)]
        out_sc = S_sel.sum(axis=1) - 1.0
        in_sc = S[np.ix_(unsel, sel)]
        found = False
        bd = min_delta
        boi = bii = -1
        for oi in range(k):
            gains = in_sc.sum(axis=1) - in_sc[:, oi]
            deltas = gains - out_sc[oi]
            mi = np.argmax(deltas)
            if deltas[mi] > bd:
                bd = deltas[mi]
                boi = oi
                bii = mi
                found = True
        if found:
            sel[boi] = unsel[bii]
            sel.sort()
            o = obj(sel.tolist())
            if o > best_obj:
                best_obj = o
                best_set = sel.tolist()
        else:
            break

    return best_set, best_obj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gram-dir", type=str, default="gram_matrices")
    parser.add_argument("--registry", type=str,
                        default="configs_blue_team_model_registry.txt")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--k", type=int, default=20)
    args = parser.parse_args()

    registry = json.loads(Path(args.registry).read_text())
    grams, names = load_gram_matrices(args.gram_dir, registry)
    print(f"Loaded {len(names)} models with Gram matrices")

    n_images = list(grams.values())[0].shape[0]
    print(f"Each Gram matrix: {n_images} x {n_images}")

    # Full optimization
    print("\n=== Full Dataset Optimization ===")
    S_full = compute_cka_from_grams(grams, names)
    best_full, score_full = optimize_fast(S_full, args.k)
    print(f"Score: {score_full:.6f}")
    print(f"Models: {[names[i] for i in sorted(best_full)]}")

    # Cross-validation folds
    print(f"\n=== {args.n_folds}-Fold Cross-Validation ===")
    all_indices = np.arange(n_images)
    rng = np.random.RandomState(42)
    rng.shuffle(all_indices)

    fold_size = n_images // args.n_folds
    fold_selections = []
    fold_scores = []

    for fold in range(args.n_folds):
        start = fold * fold_size
        end = start + fold_size
        test_idx = all_indices[start:end]
        train_idx = np.concatenate([all_indices[:start], all_indices[end:]])

        S_fold = compute_cka_from_grams(grams, names, train_idx)
        sel, score = optimize_fast(S_fold, args.k)
        fold_selections.append(set(names[i] for i in sel))
        fold_scores.append(score)
        print(f"  Fold {fold}: score={score:.6f}")

    # Bootstrap (random subsets)
    print(f"\n=== Bootstrap Validation (10 random subsets) ===")
    for boot in range(10):
        subset = rng.choice(n_images, n_images // 2, replace=False)
        S_boot = compute_cka_from_grams(grams, names, subset)
        sel, score = optimize_fast(S_boot, args.k)
        fold_selections.append(set(names[i] for i in sel))
        fold_scores.append(score)
        print(f"  Bootstrap {boot}: score={score:.6f}")

    # Stability analysis
    print("\n=== Stability Analysis ===")
    full_set = set(names[i] for i in best_full)
    model_freq = Counter()
    for sel_set in fold_selections:
        for name in sel_set:
            model_freq[name] += 1

    n_runs = len(fold_selections)
    print(f"\nModel frequency across {n_runs} runs:")
    for name, count in model_freq.most_common(30):
        pct = count / n_runs * 100
        in_full = "*" if name in full_set else " "
        print(f"  {in_full} {name:50s}: {count:2d}/{n_runs} ({pct:5.1f}%)")

    # Core models (appear in >80% of runs)
    core = {name for name, count in model_freq.items()
            if count >= n_runs * 0.8}
    print(f"\nCore models (>80% frequency): {len(core)}")
    for name in sorted(core):
        print(f"  {name}")

    # Jaccard similarity between full and folds
    overlaps = []
    for sel_set in fold_selections:
        overlap = len(full_set & sel_set) / len(full_set | sel_set)
        overlaps.append(overlap)
    print(f"\nJaccard similarity (full vs folds): "
          f"{np.mean(overlaps):.3f} +/- {np.std(overlaps):.3f}")

    print(f"\nScore stats: mean={np.mean(fold_scores):.6f}, "
          f"std={np.std(fold_scores):.6f}, "
          f"min={np.min(fold_scores):.6f}, max={np.max(fold_scores):.6f}")
    print(f"Full dataset score: {score_full:.6f}")


if __name__ == "__main__":
    main()
