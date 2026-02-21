#!/usr/bin/env python3
"""
Run the Blue Team GPU pipeline on Modal.com with an A100 GPU.

This deploys the full pipeline to a cloud A100, runs extraction + CKA + optimization,
and downloads the results locally. Total time: ~30-60 minutes, cost: ~$2-5.

Prerequisites:
  pip install modal
  modal token set  # one-time auth

Usage:
  python run_on_modal.py                          # auto-download proxy images
  python run_on_modal.py --n-images 1000          # use 1000 proxy images
  python run_on_modal.py --gpu a100               # specify GPU type
  python run_on_modal.py --hf-token hf_xxx --hf-repo user/blue-team-gpu
"""

import argparse
import json
import os
import sys
from pathlib import Path

import modal

GPU_MAP = {
    "a100": modal.gpu.A100(size="80GB"),
    "a100-40": modal.gpu.A100(size="40GB"),
    "a10g": modal.gpu.A10G(),
    "t4": modal.gpu.T4(),
    "h100": modal.gpu.H100(),
    "l4": modal.gpu.L4(),
}

app = modal.App("blue-team-cka-pipeline")

pip_packages = [
    "torch>=2.1.0",
    "torchvision>=0.16.0",
    "timm>=1.0.0",
    "numpy>=1.24.0",
    "scipy>=1.11.0",
    "Pillow>=10.0.0",
    "tqdm>=4.65.0",
    "huggingface_hub>=0.20.0",
    "datasets>=2.16.0",
]

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(*pip_packages)
)

vol = modal.Volume.from_name("blue-team-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu="A100",
    timeout=7200,
    volumes={"/cache": vol},
    _allow_background_volume_commits=True,
)
def run_pipeline(registry_json: str, n_images: int = 500,
                 hf_token: str | None = None,
                 hf_repo: str | None = None) -> dict:
    """Run the full CKA pipeline on a cloud GPU."""
    import gc
    import time
    import warnings
    warnings.filterwarnings("ignore")

    import numpy as np
    import torch

    gpu_name = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    gpu_mem = (getattr(props, 'total_memory', None) or getattr(props, 'total_mem', 0)) / 1e9
    print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")

    registry = json.loads(registry_json)
    print(f"Registry: {len(registry)} models")

    gram_dir = "/cache/gram_matrices"
    os.makedirs(gram_dir, exist_ok=True)

    # --- Phase 1: Get proxy images ---
    print("\n=== Phase 1a: Getting proxy images ===")
    proxy_dir = "/cache/proxy_images"
    image_paths = _get_or_download_images(proxy_dir, n_images)
    print(f"Using {len(image_paths)} proxy images")

    # --- Phase 1b: Extract Gram matrices ---
    print("\n=== Phase 1b: Extracting Gram matrices ===")
    t0 = time.time()
    results = {}
    failed = []

    for i, model_info in enumerate(registry):
        name = model_info["model_name"]
        out_path = os.path.join(gram_dir, f"{name}.npy")

        if os.path.exists(out_path):
            results[name] = out_path
            continue

        print(f"[{i+1:3d}/{len(registry)}] {name}...", end="", flush=True)
        t1 = time.time()
        path = _extract_gram(model_info, image_paths, "cuda", gram_dir)
        dt = time.time() - t1

        if path:
            results[name] = path
            K_c = np.load(path)
            print(f" done ({dt:.1f}s, shape={K_c.shape})")
        else:
            failed.append(name)
            print(f" FAILED ({dt:.1f}s)")

        if i % 10 == 0:
            vol.commit()

    vol.commit()
    dt_total = time.time() - t0
    print(f"\nExtraction: {len(results)}/{len(registry)} success, "
          f"{len(failed)} failed in {dt_total/60:.1f} min")
    if failed:
        print(f"Failed models: {failed}")

    # --- Phase 2: Compute CKA matrix ---
    print("\n=== Phase 2: Computing CKA matrix ===")
    model_names = []
    gram_flat = []

    for model_info in registry:
        name = model_info["model_name"]
        gram_path = os.path.join(gram_dir, f"{name}.npy")
        if not os.path.exists(gram_path):
            continue
        K_c = np.load(gram_path)
        gram_flat.append(K_c.ravel())
        model_names.append(name)

    G = np.stack(gram_flat, axis=0)
    norms = np.linalg.norm(G, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    G_normed = G / norms
    S = G_normed @ G_normed.T
    np.clip(S, 0.0, 1.0, out=S)
    np.fill_diagonal(S, 1.0)

    n_avail = len(model_names)
    mean_cka = (S.sum() - n_avail) / (n_avail * (n_avail - 1))
    max_cka = np.max(S - np.eye(n_avail))
    print(f"CKA matrix: {S.shape}, mean={mean_cka:.4f}, max={max_cka:.4f}")

    # --- Phase 3: Optimization ---
    print("\n=== Phase 3: Optimization ===")
    from scipy.linalg import eigh

    k = 20
    all_results = {}

    # Row-sum
    row_means = (S.sum(axis=1) - 1) / (n_avail - 1)
    top_k = np.argsort(row_means)[-k:]
    all_results["rowsum"] = (top_k.tolist(), _obj_fn(S, top_k.tolist(), k))

    # Greedy
    all_results["greedy"] = _greedy(S, k)

    # Spectral
    all_results["spectral"] = _spectral(S, k, eigh, 1000)

    # Frank-Wolfe
    best_fw_obj = -np.inf
    best_fw_sel = None
    for lam in [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]:
        sel, obj = _frank_wolfe(S, k, lam)
        if obj > best_fw_obj:
            best_fw_obj = obj
            best_fw_sel = sel
    all_results["frank_wolfe"] = (best_fw_sel, best_fw_obj)

    print("\nInitial results:")
    for m, (s, o) in sorted(all_results.items(), key=lambda x: -x[1][1]):
        print(f"  {m:20s}: {o:.6f}")

    # Local search from each
    refined = {}
    for method, (sel, obj) in all_results.items():
        sel_ls, obj_ls = _local_search(S, k, sel, 5000)
        refined[f"{method}+LS"] = (sel_ls, obj_ls)

    print("\nAfter local search:")
    for m, (s, o) in sorted(refined.items(), key=lambda x: -x[1][1]):
        print(f"  {m:20s}: {o:.6f}")

    # Simulated annealing
    all_combined = {**all_results, **refined}
    best_method = max(all_combined, key=lambda m: all_combined[m][1])
    best_init = all_combined[best_method][0]

    sa_results = {}
    for i in range(10):
        if i == 0:
            init = best_init
        elif i <= 4:
            init = list(best_init)
            rng = np.random.RandomState(i)
            for _ in range(max(1, k // 4)):
                oi = rng.randint(k)
                avail = list(set(range(n_avail)) - set(init))
                init[oi] = rng.choice(avail)
        else:
            init = np.random.RandomState(i + 200).choice(
                n_avail, k, replace=False).tolist()

        print(f"  SA restart {i}...", end="", flush=True)
        sel_sa, obj_sa = _simulated_annealing(S, k, init, 3_000_000, i)
        sel_sa, obj_sa = _local_search(S, k, sel_sa, 5000)
        sa_results[f"SA_{i}"] = (sel_sa, obj_sa)
        print(f" {obj_sa:.6f}")

    all_final = {**all_combined, **refined, **sa_results}
    best_method = max(all_final, key=lambda m: all_final[m][1])
    best_sel, best_obj = all_final[best_method]

    print(f"\nBEST: {best_method} = {best_obj:.6f}")
    print("Selected models:")
    for idx in sorted(best_sel):
        print(f"  {model_names[idx]}")

    # Build submission
    name_to_info = {m["model_name"]: m for m in registry}
    submission = []
    for idx in sorted(best_sel):
        name = model_names[idx]
        info = name_to_info[name]
        submission.append({
            "model_name": name,
            "layer_name": info["layer"],
        })

    # Push to HuggingFace if requested
    hf_url = None
    if hf_token and hf_repo:
        print(f"\nPushing to HuggingFace: {hf_repo}")
        try:
            from datasets import Dataset
            from huggingface_hub import login
            login(token=hf_token)
            rows = [{"model_name": m["model_name"], "layer_name": m["layer_name"]}
                    for m in submission]
            ds = Dataset.from_list(rows)
            ds.push_to_hub(hf_repo, private=True)
            hf_url = f"https://huggingface.co/datasets/{hf_repo}"
            print(f"Dataset pushed: {hf_url}")
        except Exception as e:
            print(f"HuggingFace push failed: {e}")

    # Save to volume
    np.savez("/cache/cka_results.npz",
             cka=S, model_names=np.array(model_names),
             best_sel=np.array(best_sel), best_obj=best_obj)
    vol.commit()

    return {
        "submission": submission,
        "predicted_score": float(best_obj),
        "best_method": best_method,
        "n_models_extracted": len(model_names),
        "n_models_failed": len(failed),
        "failed_models": failed,
        "cka_matrix_mean": float(mean_cka),
        "cka_matrix_max": float(max_cka),
        "hf_url": hf_url,
    }


# ---------------------------------------------------------------------------
# Helper functions (defined inside the module for Modal serialization)
# ---------------------------------------------------------------------------

def _get_or_download_images(proxy_dir: str, n_images: int) -> list[str]:
    """Download or load cached proxy images."""
    os.makedirs(proxy_dir, exist_ok=True)
    from pathlib import Path as P
    existing = sorted(list(P(proxy_dir).glob("*.jpg")) +
                      list(P(proxy_dir).glob("*.png")) +
                      list(P(proxy_dir).glob("*.JPEG")))
    if len(existing) >= n_images:
        print(f"Using {n_images} cached proxy images")
        return [str(p) for p in existing[:n_images]]

    print("Downloading Food-101 test images...")
    import torchvision.datasets as dsets
    import numpy as np

    food = dsets.Food101(root="/tmp/food101", split="test", download=True)
    indices = np.random.RandomState(42).choice(
        len(food), min(n_images, len(food)), replace=False
    )
    paths = []
    for i, idx in enumerate(indices):
        img, _ = food[idx]
        out = os.path.join(proxy_dir, f"food_{i:05d}.jpg")
        img.save(out)
        paths.append(out)
        if (i + 1) % 200 == 0:
            print(f"  Saved {i+1}/{len(indices)}")
    print(f"Downloaded {len(paths)} proxy images")
    return paths


def _extract_gram(model_info: dict, image_paths: list[str],
                  device: str, gram_dir: str) -> str | None:
    """Extract centered Gram matrix for one model."""
    import gc
    import torch
    import timm
    import numpy as np
    from PIL import Image
    from torchvision import transforms

    name = model_info["model_name"]
    layer = model_info["layer"]
    pp = model_info["preprocess"]
    out_path = os.path.join(gram_dir, f"{name}.npy")

    crop = pp["crop"]
    bs = 64 if crop <= 224 else (32 if crop <= 288 else (16 if crop <= 384 else (8 if crop <= 512 else (4 if crop <= 768 else 2))))

    try:
        model = timm.create_model(name, pretrained=True)
        model.eval().to(device).half()

        transform = transforms.Compose([
            transforms.Resize(pp["resize"],
                             interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(crop),
            transforms.ToTensor(),
            transforms.Normalize(mean=pp["mean"], std=pp["std"]),
        ])

        modules = dict(model.named_modules())
        if layer not in modules:
            del model; torch.cuda.empty_cache()
            return None

        features_buf = {}
        def hook_fn(m, inp, out):
            features_buf["out"] = out

        handle = modules[layer].register_forward_hook(hook_fn)
        all_feats = []

        with torch.no_grad(), torch.cuda.amp.autocast():
            for i in range(0, len(image_paths), bs):
                batch_paths = image_paths[i:i+bs]
                imgs = []
                for p in batch_paths:
                    try:
                        imgs.append(transform(Image.open(p).convert("RGB")))
                    except:
                        imgs.append(torch.zeros(3, crop, crop))
                if not imgs:
                    continue
                batch = torch.stack(imgs).to(device).half()
                try:
                    model(batch)
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        torch.cuda.empty_cache()
                        half = max(1, len(imgs) // 2)
                        for si in range(0, len(imgs), half):
                            sb = torch.stack(imgs[si:si+half]).to(device).half()
                            model(sb)
                            f = features_buf["out"]
                            if f.dim() > 2: f = f.flatten(1)
                            all_feats.append(f.float().cpu().numpy())
                            del sb; torch.cuda.empty_cache()
                        continue
                    raise
                f = features_buf["out"]
                if f.dim() > 2: f = f.flatten(1)
                all_feats.append(f.float().cpu().numpy())
                del batch

        handle.remove()
        if not all_feats:
            del model; torch.cuda.empty_cache()
            return None

        X = np.concatenate(all_feats, axis=0).astype(np.float64)
        X -= X.mean(axis=0, keepdims=True)
        K = X @ X.T
        n = K.shape[0]
        rm = K.mean(axis=1, keepdims=True)
        cm = K.mean(axis=0, keepdims=True)
        gm = K.mean()
        K_c = K - rm - cm + gm

        np.save(out_path, K_c)
        del model, X, K, K_c, all_feats
        torch.cuda.empty_cache()
        gc.collect()
        return out_path

    except Exception as e:
        print(f" [{e}]", end="")
        try:
            import torch; torch.cuda.empty_cache()
        except:
            pass
        gc.collect()
        return None


def _obj_fn(S, sel, k):
    import numpy as np
    sub = S[np.ix_(sel, sel)]
    return (sub.sum() - k) / (k * (k - 1))


def _greedy(S, k):
    import numpy as np
    n = S.shape[0]
    off = S.copy(); np.fill_diagonal(off, -np.inf)
    bp = np.unravel_index(np.argmax(off), S.shape)
    sel = list(bp); rem = set(range(n)) - set(sel)
    while len(sel) < k:
        rl = list(rem)
        scores = S[rl][:, sel].sum(axis=1)
        best = rl[np.argmax(scores)]
        sel.append(best); rem.remove(best)
    return sel, _obj_fn(S, sel, k)


def _spectral(S, k, eigh, n_restarts):
    import numpy as np
    evals, evecs = eigh(S)
    best_obj = -np.inf; best_set = None
    v1 = evecs[:, -1]
    top_k = np.argsort(np.abs(v1))[-k:]
    obj = _obj_fn(S, top_k.tolist(), k)
    if obj > best_obj: best_obj = obj; best_set = top_k.tolist()
    rng = np.random.RandomState(42)
    for r in [1, 2, 3, 5, 10, 20, 30, 50]:
        if r > S.shape[0]: continue
        V = evecs[:, -r:]
        for _ in range(n_restarts):
            w = rng.randn(r)
            scores = V @ w
            top_k = np.argsort(scores)[-k:]
            obj = _obj_fn(S, top_k.tolist(), k)
            if obj > best_obj: best_obj = obj; best_set = top_k.tolist()
    return best_set, best_obj


def _frank_wolfe(S, k, lam, max_iter=1000):
    import numpy as np
    n = S.shape[0]
    Sd = S + lam * np.eye(n)
    Ln = np.linalg.norm(Sd, ord=2)
    x = np.ones(n) * (k / n)
    for _ in range(max_iter):
        g = Sd @ x
        tk = np.argpartition(g, -k)[-k:]
        s = np.zeros(n); s[tk] = 1.0
        d = s - x; gap = g @ d
        if gap <= 1e-10: break
        gamma = min(1.0, gap / (Ln * np.linalg.norm(d)**2 + 1e-12))
        x += gamma * d
    sel = np.argpartition(x, -k)[-k:]
    return sel.tolist(), _obj_fn(S, sel.tolist(), k)


def _local_search(S, k, initial, max_no_improve=2000, max_iter=50000):
    import numpy as np
    n = S.shape[0]
    sel = np.array(sorted(initial[:k]))
    best_obj = _obj_fn(S, sel.tolist(), k); best_set = sel.tolist()
    no_imp = 0; total_iter = 0; min_delta = 1e-10
    rng = np.random.RandomState(42)
    while no_imp < max_no_improve and total_iter < max_iter:
        total_iter += 1
        unsel = np.setdiff1d(np.arange(n), sel)
        Ss = S[np.ix_(sel, sel)]
        out_sc = Ss.sum(axis=1) - 1.0
        in_sc = S[np.ix_(unsel, sel)]
        found = False; bd = min_delta; boi = -1; bii = -1
        for oi in range(k):
            loss = out_sc[oi]
            gains = in_sc.sum(axis=1) - in_sc[:, oi]
            deltas = gains - loss
            mi = np.argmax(deltas)
            if deltas[mi] > bd: bd = deltas[mi]; boi = oi; bii = mi; found = True
        if found:
            sel[boi] = unsel[bii]; sel.sort()
            obj = _obj_fn(S, sel.tolist(), k)
            if obj > best_obj: best_obj = obj; best_set = sel.tolist()
            no_imp = 0
        else:
            no_imp += 1
            oi = rng.randint(k)
            un = np.setdiff1d(np.arange(n), sel)
            sel[oi] = un[rng.randint(len(un))]; sel.sort()
    return best_set, best_obj


def _simulated_annealing(S, k, initial, n_iter, seed):
    import numpy as np
    n = S.shape[0]; rng = np.random.RandomState(seed)
    in_set = np.zeros(n, dtype=bool)
    for i in initial[:k]: in_set[i] = True
    cur = _obj_fn(S, list(np.where(in_set)[0]), k)
    best = cur; best_set = np.where(in_set)[0].tolist()
    T = 0.1; alpha = (1e-7 / 0.1) ** (1.0 / n_iter)
    for _ in range(n_iter):
        sel = np.where(in_set)[0]; uns = np.where(~in_set)[0]
        vo = rng.choice(sel); vi = rng.choice(uns)
        tm = sel[sel != vo]
        g = S[vi, tm].sum() - S[vo, tm].sum()
        d = g * 2 / (k * (k-1))
        if d > 0 or rng.rand() < np.exp(d / max(T, 1e-15)):
            in_set[vo] = False; in_set[vi] = True
            cur += d
            if cur > best: best = cur; best_set = np.where(in_set)[0].tolist()
        T *= alpha
    return best_set, best


# ---------------------------------------------------------------------------
# Local entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run Blue Team CKA pipeline on Modal.com"
    )
    parser.add_argument("--gpu", type=str, default="a100",
                        choices=list(GPU_MAP.keys()),
                        help="GPU type (default: a100)")
    parser.add_argument("--n-images", type=int, default=500,
                        help="Number of proxy images")
    parser.add_argument("--registry", type=str,
                        default="configs_blue_team_model_registry.txt",
                        help="Path to model registry")
    parser.add_argument("--hf-token", type=str, default=None)
    parser.add_argument("--hf-repo", type=str, default=None)
    parser.add_argument("--output", type=str, default="submission_modal.json")
    args = parser.parse_args()

    registry_json = Path(args.registry).read_text()

    print("Deploying to Modal...")
    print(f"GPU: {args.gpu}, Images: {args.n_images}")

    with modal.enable_output():
        with app.run():
            result = run_pipeline.remote(
                registry_json=registry_json,
                n_images=args.n_images,
                hf_token=args.hf_token,
                hf_repo=args.hf_repo,
            )

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Predicted score: {result['predicted_score']:.6f}")
    print(f"Best method: {result['best_method']}")
    print(f"Models extracted: {result['n_models_extracted']}")
    print(f"Models failed: {result['n_models_failed']}")
    if result['failed_models']:
        print(f"Failed: {result['failed_models']}")
    if result['hf_url']:
        print(f"HuggingFace: {result['hf_url']}")

    # Save submission
    output = {
        "models": result["submission"],
        "metadata": {
            "predicted_score": result["predicted_score"],
            "method": result["best_method"],
            "n_proxy_images": args.n_images,
        }
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSubmission saved to {args.output}")

    print("\n=== SUBMISSION JSON ===")
    print(json.dumps({"models": result["submission"]}, indent=2))


if __name__ == "__main__":
    main()
