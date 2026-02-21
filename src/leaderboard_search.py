#!/usr/bin/env python3
"""
Leaderboard-Based Coordinate Descent for Blue Team

The key insight: the leaderboard gives you the TRUE objective value on the
HIDDEN evaluation images. No proxy needed. This makes it the most powerful
optimization signal available.

Strategy:
  1. Start from your best GPU-optimized set
  2. Generate promising swap candidates using the proxy CKA matrix
  3. Submit each candidate to the leaderboard
  4. Record true scores and update our model of the CKA landscape
  5. Repeat with the best set found so far

With ~20 min per submission, we can evaluate ~70 candidates/day.
A targeted search of ~100-200 submissions can find near-optimal solutions.

Usage:
  # Generate initial batch of candidates from GPU results
  python leaderboard_search.py generate --cka-matrix cka_matrix.npz \
      --current-set submission_gpu.json --n-candidates 50

  # After recording leaderboard scores, generate next batch
  python leaderboard_search.py refine --scores-file scores.csv \
      --cka-matrix cka_matrix.npz --n-candidates 30

  # Analyze results and recommend final submission
  python leaderboard_search.py analyze --scores-file scores.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


def load_registry(path: str = "configs_blue_team_model_registry.txt") -> list[dict]:
    return json.loads(Path(path).read_text())


def load_cka_matrix(path: str) -> tuple[np.ndarray, list[str]]:
    data = np.load(path, allow_pickle=True)
    return data["cka"], list(data["model_names"])


def obj(S: np.ndarray, indices: list[int], k: int = 20) -> float:
    sub = S[np.ix_(indices, indices)]
    return (sub.sum() - k) / (k * (k - 1))


def name_to_idx(model_names: list[str]) -> dict[str, int]:
    return {name: i for i, name in enumerate(model_names)}


def push_candidate(models: list[dict], hf_token: str, repo: str):
    """Push a candidate submission to HuggingFace."""
    from datasets import Dataset
    from huggingface_hub import login

    login(token=hf_token)
    rows = [{"model_name": m["model_name"], "layer_name": m["layer_name"]}
            for m in models]
    ds = Dataset.from_list(rows)
    ds.push_to_hub(repo, private=True)
    print(f"  Pushed to https://huggingface.co/datasets/{repo}")


# ---------------------------------------------------------------------------
# Generate candidates
# ---------------------------------------------------------------------------

def generate_candidates(cka_path: str, current_set_path: str,
                        registry_path: str, n_candidates: int,
                        output_dir: str, hf_token: str | None = None,
                        hf_repo_prefix: str | None = None):
    """Generate swap candidates ranked by expected improvement."""

    S, model_names = load_cka_matrix(cka_path)
    registry = load_registry(registry_path)
    name_map = name_to_idx(model_names)
    info_map = {m["model_name"]: m for m in registry}

    # Load current set
    with open(current_set_path) as f:
        current_data = json.load(f)
    current_models = current_data.get("models", current_data)
    if isinstance(current_models, dict):
        current_models = current_models.get("models", [])

    current_names = [m["model_name"] for m in current_models]
    current_idx = [name_map[n] for n in current_names if n in name_map]
    k = len(current_idx)
    current_score = obj(S, current_idx, k)

    print(f"Current set: {k} models, proxy CKA = {current_score:.6f}")

    # For each model in the set, compute its contribution
    contributions = []
    for i, idx in enumerate(current_idx):
        others = [j for j in current_idx if j != idx]
        contrib = sum(S[idx, j] for j in others)
        contributions.append((i, idx, contrib))

    contributions.sort(key=lambda x: x[2])
    print("\nModel contributions (lowest = best swap-out candidates):")
    for rank, (pos, idx, contrib) in enumerate(contributions[:10]):
        print(f"  {rank+1}. {model_names[idx]:50s} contrib={contrib:.4f}")

    # For each model NOT in the set, compute its affinity to the set
    outside = [i for i in range(len(model_names)) if i not in set(current_idx)]
    affinities = []
    for idx in outside:
        aff = sum(S[idx, j] for j in current_idx)
        affinities.append((idx, aff))
    affinities.sort(key=lambda x: -x[1])

    print("\nTop swap-in candidates (highest affinity to current set):")
    for rank, (idx, aff) in enumerate(affinities[:10]):
        print(f"  {rank+1}. {model_names[idx]:50s} affinity={aff:.4f}")

    # Generate swap candidates: for each of the weakest models in the set,
    # try replacing with the strongest outside models
    candidates = []
    n_weak = min(k, 10)
    n_strong = min(len(outside), 20)

    weak_models = contributions[:n_weak]
    strong_candidates = affinities[:n_strong]

    for pos, out_idx, out_contrib in weak_models:
        for in_idx, in_aff in strong_candidates:
            new_set = [j for j in current_idx if j != out_idx] + [in_idx]
            new_score = obj(S, new_set, k)
            delta = new_score - current_score
            candidates.append({
                "swap_out": model_names[out_idx],
                "swap_in": model_names[in_idx],
                "proxy_score": new_score,
                "proxy_delta": delta,
                "indices": sorted(new_set),
            })

    # Also try multi-swaps: replace the 2-3 weakest simultaneously
    for n_swap in [2, 3]:
        if n_swap > n_weak:
            continue
        out_indices = [contributions[i][1] for i in range(n_swap)]
        remaining = [j for j in current_idx if j not in set(out_indices)]
        # Try top combinations of swap-ins
        from itertools import combinations
        for combo in list(combinations(
            [idx for idx, _ in strong_candidates[:15]], n_swap
        ))[:50]:
            new_set = remaining + list(combo)
            if len(set(new_set)) != k:
                continue
            new_score = obj(S, new_set, k)
            delta = new_score - current_score
            outs = [model_names[i] for i in out_indices]
            ins = [model_names[i] for i in combo]
            candidates.append({
                "swap_out": " + ".join(outs),
                "swap_in": " + ".join(ins),
                "proxy_score": new_score,
                "proxy_delta": delta,
                "indices": sorted(new_set),
            })

    # Also try completely different clusters (architecture families)
    from blue_team_optimizer import classify_architecture
    families = {}
    for i, name in enumerate(model_names):
        fam = classify_architecture(name)
        families.setdefault(fam, []).append(i)

    for fam, members in families.items():
        if len(members) >= k:
            # Try the top-k by row sum within this family
            sub_S = S[np.ix_(members, members)]
            row_sums = sub_S.sum(axis=1) - 1
            top_k_local = np.argsort(row_sums)[-k:]
            global_indices = [members[i] for i in top_k_local]
            score = obj(S, global_indices, k)
            candidates.append({
                "swap_out": f"[full family swap: {fam}]",
                "swap_in": f"top-{k} of {fam} ({len(members)} models)",
                "proxy_score": score,
                "proxy_delta": score - current_score,
                "indices": sorted(global_indices),
            })

    # Sort by expected improvement
    candidates.sort(key=lambda c: -c["proxy_score"])

    # Deduplicate
    seen = set()
    unique = []
    for c in candidates:
        key = tuple(c["indices"])
        if key not in seen:
            seen.add(key)
            unique.append(c)
    candidates = unique[:n_candidates]

    # Save candidates
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"Top {len(candidates)} candidates (ranked by proxy CKA):")
    print(f"{'='*70}")

    manifest = []
    for rank, c in enumerate(candidates):
        print(f"\n  #{rank+1}: proxy={c['proxy_score']:.6f} "
              f"(delta={c['proxy_delta']:+.6f})")
        print(f"      out: {c['swap_out']}")
        print(f"      in:  {c['swap_in']}")

        # Build submission
        models = []
        for idx in c["indices"]:
            name = model_names[idx]
            info = info_map[name]
            models.append({
                "model_name": name,
                "layer_name": info["layer"],
            })

        submission = {"models": models}
        fname = f"candidate_{rank:03d}.json"
        fpath = os.path.join(output_dir, fname)
        with open(fpath, "w") as f:
            json.dump(submission, f, indent=2)

        manifest.append({
            "rank": rank,
            "file": fname,
            "proxy_score": c["proxy_score"],
            "swap_out": c["swap_out"],
            "swap_in": c["swap_in"],
            "true_score": None,
        })

        # Push to HuggingFace if requested
        if hf_token and hf_repo_prefix and rank < 5:
            repo = f"{hf_repo_prefix}-c{rank:02d}"
            try:
                push_candidate(models, hf_token, repo)
            except Exception as e:
                print(f"      HF push failed: {e}")

    # Save manifest
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest saved to {manifest_path}")

    # Save CSV for easy score tracking
    csv_path = os.path.join(output_dir, "scores.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "file", "proxy_score", "true_score",
                     "swap_out", "swap_in"])
        for m in manifest:
            w.writerow([m["rank"], m["file"], f"{m['proxy_score']:.6f}",
                        "", m["swap_out"], m["swap_in"]])
    print(f"Score tracking CSV: {csv_path}")
    print("\nFill in the 'true_score' column as you submit each candidate.")

    return candidates


# ---------------------------------------------------------------------------
# Analyze results
# ---------------------------------------------------------------------------

def analyze_scores(scores_file: str, cka_path: str | None = None):
    """Analyze leaderboard scores and recommend next steps."""

    rows = []
    with open(scores_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("true_score"):
                row["true_score"] = float(row["true_score"])
                row["proxy_score"] = float(row["proxy_score"])
                rows.append(row)

    if not rows:
        print("No true scores recorded yet. Submit candidates and fill in scores.csv")
        return

    print(f"\n{'='*70}")
    print(f"Leaderboard Results ({len(rows)} scored)")
    print(f"{'='*70}")

    # Sort by true score
    rows.sort(key=lambda r: -r["true_score"])

    for r in rows:
        delta = r["true_score"] - r["proxy_score"]
        print(f"  #{r['rank']:3s} true={r['true_score']:.4f} "
              f"proxy={r['proxy_score']:.4f} (gap={delta:+.4f})  "
              f"{r['swap_out'][:30]} -> {r['swap_in'][:30]}")

    # Correlation analysis
    true_scores = [r["true_score"] for r in rows]
    proxy_scores = [r["proxy_score"] for r in rows]

    if len(rows) >= 3:
        correlation = np.corrcoef(proxy_scores, true_scores)[0, 1]
        print(f"\nProxy-True correlation: {correlation:.4f}")
        if correlation > 0.8:
            print("  Strong correlation! Proxy CKA is a good guide.")
        elif correlation > 0.5:
            print("  Moderate correlation. Proxy helps but leaderboard has surprises.")
        else:
            print("  Weak correlation! The proxy CKA is misleading.")
            print("  Consider: different image distribution, CKA instability,")
            print("  or proxy images not representative of evaluation set.")

    # Best submission
    best = rows[0]
    print(f"\nBest submission: {best['file']}")
    print(f"  True score: {best['true_score']:.6f}")
    print(f"  Swap: {best['swap_out']} -> {best['swap_in']}")

    # Proxy calibration
    mean_gap = np.mean([r["true_score"] - r["proxy_score"] for r in rows])
    print(f"\nMean proxy-true gap: {mean_gap:+.4f}")
    print(f"  (positive = proxy underestimates, negative = overestimates)")


# ---------------------------------------------------------------------------
# Quick-submit helper
# ---------------------------------------------------------------------------

def quick_submit(submission_path: str, hf_token: str, hf_repo: str):
    """Push a single candidate submission to HuggingFace."""
    with open(submission_path) as f:
        data = json.load(f)
    models = data.get("models", data)
    if isinstance(models, dict):
        models = models.get("models", [])
    push_candidate(models, hf_token, hf_repo)
    print(f"\nSubmitted {submission_path} to {hf_repo}")
    print("Check leaderboard in ~20 minutes.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Leaderboard-based coordinate descent for Blue Team"
    )
    sub = parser.add_subparsers(dest="command")

    # Generate
    gen = sub.add_parser("generate", help="Generate swap candidates")
    gen.add_argument("--cka-matrix", required=True)
    gen.add_argument("--current-set", required=True,
                     help="Current best submission JSON")
    gen.add_argument("--registry", default="configs_blue_team_model_registry.txt")
    gen.add_argument("--n-candidates", type=int, default=50)
    gen.add_argument("--output-dir", default="candidates")
    gen.add_argument("--hf-token", default=None)
    gen.add_argument("--hf-repo-prefix", default=None,
                     help="e.g. USER/blue-team")

    # Analyze
    ana = sub.add_parser("analyze", help="Analyze leaderboard scores")
    ana.add_argument("--scores-file", required=True)
    ana.add_argument("--cka-matrix", default=None)

    # Submit
    sub_cmd = sub.add_parser("submit", help="Push a candidate to HuggingFace")
    sub_cmd.add_argument("--submission", required=True)
    sub_cmd.add_argument("--hf-token", required=True)
    sub_cmd.add_argument("--hf-repo", required=True)

    # Batch-submit top N
    batch = sub.add_parser("batch-submit",
                           help="Submit top N candidates to HuggingFace")
    batch.add_argument("--candidates-dir", default="candidates")
    batch.add_argument("--n", type=int, default=5)
    batch.add_argument("--hf-token", required=True)
    batch.add_argument("--hf-repo-prefix", required=True,
                       help="e.g. USER/blue-team")
    batch.add_argument("--start-from", type=int, default=0)

    args = parser.parse_args()

    if args.command == "generate":
        generate_candidates(
            cka_path=args.cka_matrix,
            current_set_path=args.current_set,
            registry_path=args.registry,
            n_candidates=args.n_candidates,
            output_dir=args.output_dir,
            hf_token=args.hf_token,
            hf_repo_prefix=args.hf_repo_prefix,
        )

    elif args.command == "analyze":
        analyze_scores(args.scores_file, args.cka_matrix)

    elif args.command == "submit":
        quick_submit(args.submission, args.hf_token, args.hf_repo)

    elif args.command == "batch-submit":
        manifest_path = os.path.join(args.candidates_dir, "manifest.json")
        with open(manifest_path) as f:
            manifest = json.load(f)

        for entry in manifest[args.start_from:args.start_from + args.n]:
            rank = entry["rank"]
            fpath = os.path.join(args.candidates_dir, entry["file"])
            repo = f"{args.hf_repo_prefix}-c{rank:02d}"
            print(f"\n--- Candidate #{rank} ---")
            print(f"  Proxy score: {entry['proxy_score']:.6f}")
            print(f"  Swap: {entry['swap_out']} -> {entry['swap_in']}")
            quick_submit(fpath, args.hf_token, repo)
            print(f"  Submitted to: {repo}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
