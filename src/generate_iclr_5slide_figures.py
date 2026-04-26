#!/usr/bin/env python3
"""Generate all five figures for the ICLR Re-Align five-slide talk.

Slide 1 (title):  Conceptual diagram — two stimulus conditions, one formula.
Slide 2 (claim):  Cleaned real Red Gram figure (from report PNG, no text).
Slide 3 (blue):   Cleaned real Blue CKA heatmap (from report PNG, no text).
Slide 4 (red):    Red optimization funnel + proxy model strip.
Slide 5 (results): Score bars + Blue submission composition.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BLUE_SUBMISSION = ROOT / "submissions" / "blue_team_submission.json"
RED_SUBMISSION = ROOT / "submissions" / "red_team_submission.json"
BLUE_REPORT_FIG = ROOT / "submissions" / "blue_team_real_cka_matrix.png"
RED_REPORT_FIG = ROOT / "submissions" / "red_team_real_gram_insight.png"
OUT = ROOT / "figures" / "iclr_5slide"

BG = "#fbfaf7"
INK = "#080808"
MUTED = "#77736a"
BLUE = "#0072b2"
RED = "#b33c00"
GOLD = "#e69f00"
GREY = "#999999"
TEAL = "#009e73"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def family_for(name: str) -> str:
    n = name.lower()
    if any(k in n for k in ["vit", "deit", "beit", "cait", "xcit", "pit",
                             "pvt", "tnt", "volo", "twins", "convit",
                             "crossvit", "davit", "flexivit", "gcvit",
                             "maxvit", "mvit", "nest", "nextvit", "aim"]):
        return "ViT / attention"
    return "hybrid / other"


def setup():
    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": BG,
        "savefig.facecolor": BG, "savefig.dpi": 300,
        "font.family": "sans-serif", "font.size": 13,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "text.color": INK,
    })


def make_slide1():
    """Conceptual diagram: two stimulus conditions, one CKA formula."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    for ax in axes:
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect("equal")
        ax.axis("off")

    rng = np.random.RandomState(7)

    # Left: broad categories (dogs cluster + cars cluster + birds cluster)
    clusters = [
        ("dogs", RED, -0.5, 0.5, 0.15),
        ("cars", BLUE, 0.5, -0.4, 0.15),
        ("birds", TEAL, -0.3, -0.6, 0.12),
        ("buildings", GOLD, 0.6, 0.5, 0.13),
    ]
    ax = axes[0]
    for label, color, cx, cy, spread in clusters:
        pts = rng.randn(18, 2) * spread + [cx, cy]
        ax.scatter(pts[:, 0], pts[:, 1], s=28, color=color, alpha=0.7, zorder=2)
        ax.text(cx, cy + spread + 0.15, label, ha="center", fontsize=11,
                color=color, fontweight="bold")
    ax.set_title("broad stimulus set", fontsize=15, pad=14, color=INK)

    # Right: single superclass (all dogs, spread out)
    ax = axes[1]
    pts = rng.randn(60, 2) * 0.38
    ax.scatter(pts[:, 0], pts[:, 1], s=28, color=RED, alpha=0.55, zorder=2)
    axes_labels = ["texture", "pose", "scene", "silhouette", "crop"]
    angles = np.linspace(0, 2 * np.pi, len(axes_labels), endpoint=False) - np.pi / 2
    for angle, label in zip(angles, axes_labels):
        x, y = 0.95 * np.cos(angle), 0.95 * np.sin(angle)
        ax.annotate("", xy=(x, y), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
        ax.text(x * 1.12, y * 1.12, label, ha="center", va="center",
                fontsize=10, color=MUTED)
    ax.set_title("dog superclass only", fontsize=15, pad=14, color=INK)

    fig.text(0.5, 0.02, "same CKA formula, different stimulus question",
             ha="center", fontsize=13, color=MUTED, style="italic")
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(OUT / "slide1_concept.png", bbox_inches="tight")
    plt.close(fig)


def make_slide2():
    """Clean the real Red Gram figure: crop out titles, keep four heatmaps."""
    from PIL import Image

    red = Image.open(RED_REPORT_FIG).convert("RGB")
    crops = [
        red.crop((28, 207, 1735, 1420)),
        red.crop((1800, 207, 3502, 1420)),
        red.crop((28, 1680, 1735, 2893)),
        red.crop((1800, 1680, 3502, 2893)),
    ]
    gap = 22
    w = crops[0].width * 2 + gap
    h = crops[0].height * 2 + gap
    canvas = Image.new("RGB", (w, h), "white")
    canvas.paste(crops[0], (0, 0))
    canvas.paste(crops[1], (crops[0].width + gap, 0))
    canvas.paste(crops[2], (0, crops[0].height + gap))
    canvas.paste(crops[3], (crops[0].width + gap, crops[0].height + gap))
    canvas.save(OUT / "slide2_gram.png")


def make_slide3():
    """Clean the real Blue CKA heatmap: rebuild cells, keep white dividers."""
    from PIL import Image, ImageDraw

    blue = Image.open(BLUE_REPORT_FIG).convert("RGB")
    heatmap = blue.crop((27, 164, 2326, 2271))

    n = 18
    arr = np.asarray(heatmap)
    rebuilt = Image.new("RGB", heatmap.size, "white")
    draw = ImageDraw.Draw(rebuilt)

    for row in range(n):
        y0 = round(row * heatmap.height / n)
        y1 = round((row + 1) * heatmap.height / n)
        for col in range(n):
            x0 = round(col * heatmap.width / n)
            x1 = round((col + 1) * heatmap.width / n)
            patch = arr[y0:y1, x0:x1, :]
            rgb = tuple(int(v) for v in np.median(patch.reshape(-1, 3), axis=0))
            draw.rectangle((x0, y0, x1, y1), fill=rgb)

    for boundary in (6, 12):
        x = round(boundary * heatmap.width / n)
        y = round(boundary * heatmap.height / n)
        draw.line((x, 0, x, heatmap.height), fill="white", width=8)
        draw.line((0, y, heatmap.width, y), fill="white", width=8)

    rebuilt.save(OUT / "slide3_cka.png")


def make_slide4():
    """Red optimization funnel + proxy model diversity strip."""
    red = load_json(RED_SUBMISSION)
    metadata = red.get("metadata", {})
    proxies = metadata.get("proxy_models", [])

    fig, (ax_funnel, ax_models) = plt.subplots(
        1, 2, figsize=(10, 4.5), gridspec_kw={"width_ratios": [1.1, 1]})

    # Funnel
    labels = ["synsets", "candidates", "scored pool", "submitted"]
    values = [125, 6250, metadata.get("n_candidates", 5000), 1000]
    colors = [MUTED, BLUE, GOLD, RED]
    bars = ax_funnel.barh(labels[::-1], values[::-1], color=colors[::-1],
                          height=0.6, edgecolor="white", linewidth=1.5)
    ax_funnel.set_xscale("log")
    ax_funnel.set_xlabel("count (log scale)")
    ax_funnel.set_title("selection pipeline", fontsize=14)
    for bar, val in zip(bars, values[::-1]):
        ax_funnel.text(val * 1.15, bar.get_y() + bar.get_height() / 2,
                       f"{val:,}", va="center", fontsize=11)
    ax_funnel.spines["left"].set_visible(False)
    ax_funnel.tick_params(left=False)

    # Proxy model diversity
    families = {
        "CNN": ["vgg11", "resnet101", "densenet121"],
        "ViT": ["deit3", "beit", "xcit"],
        "CLIP": ["apple mclip"],
        "self-sup": ["dino resmlp", "mae hiera"],
        "exotic": ["mambaout", "mlp-mixer"],
    }
    y_pos = 0
    fam_colors = {"CNN": RED, "ViT": BLUE, "CLIP": TEAL,
                  "self-sup": GOLD, "exotic": MUTED}
    for fam, models in families.items():
        for model in models:
            ax_models.barh(y_pos, 1, color=fam_colors[fam], height=0.7,
                           edgecolor="white", linewidth=1)
            ax_models.text(0.5, y_pos, model, ha="center", va="center",
                           fontsize=9, color="white", fontweight="bold")
            y_pos += 1
        y_pos += 0.3

    ax_models.set_xlim(0, 1)
    ax_models.set_ylim(-0.5, y_pos)
    ax_models.invert_yaxis()
    ax_models.axis("off")
    ax_models.set_title("11 proxy models", fontsize=14)

    patches = [mpatches.Patch(color=c, label=f) for f, c in fam_colors.items()]
    ax_models.legend(handles=patches, loc="lower right", fontsize=9,
                     frameon=False, ncol=2)

    fig.suptitle("Red Team: restrict to dog superclass, then optimize",
                 fontsize=15, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT / "slide4_red_pipeline.png", bbox_inches="tight")
    plt.close(fig)


def make_slide5():
    """Results: Red scores + Blue submission composition."""
    red = load_json(RED_SUBMISSION)
    metadata = red.get("metadata", {})
    proxy_score = metadata.get("predicted_score", 0.584)
    hidden_score = 0.547
    baseline = 0.479

    blue_models = [m["model_name"] for m in load_json(BLUE_SUBMISSION)["models"]]
    family_counts = Counter(family_for(name) for name in blue_models)

    fig = plt.figure(figsize=(10, 4.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.1], wspace=0.35)

    # Red scores
    ax0 = fig.add_subplot(gs[0])
    labels = ["baseline\n0.479", "hidden eval\n0.547", "proxy\n0.584"]
    vals = [baseline, hidden_score, proxy_score]
    colors = [GREY, RED, GOLD]
    bars = ax0.bar(labels, vals, color=colors, width=0.55, edgecolor="white",
                   linewidth=1.5)
    ax0.set_ylim(0.40, 0.65)
    ax0.set_ylabel("score = 1 − mean CKA", fontsize=12)
    ax0.set_title("Red Team result", fontsize=14)
    for bar in bars:
        y = bar.get_height()
        ax0.text(bar.get_x() + bar.get_width() / 2, y + 0.006,
                 f"{y:.3f}", ha="center", va="bottom", fontsize=12,
                 fontweight="bold")
    ax0.axhline(baseline, color=GREY, ls="--", lw=0.8, zorder=0)

    # Blue composition
    ax1 = fig.add_subplot(gs[1])
    fams = list(family_counts.keys())
    vals_b = [family_counts[f] for f in fams]
    colors_b = [BLUE if "ViT" in f else MUTED for f in fams]
    bars_b = ax1.barh(fams, vals_b, color=colors_b, height=0.5,
                      edgecolor="white", linewidth=1.5)
    ax1.set_xlim(0, max(vals_b) + 3)
    ax1.set_xlabel("models in submitted set", fontsize=12)
    ax1.set_title("Blue Team: 20-model submission", fontsize=14)
    for bar, val in zip(bars_b, vals_b):
        ax1.text(val + 0.4, bar.get_y() + bar.get_height() / 2,
                 str(val), va="center", fontsize=12, fontweight="bold")
    ax1.spines["left"].set_visible(False)
    ax1.tick_params(left=False)

    fig.suptitle("Both tracks exploit the same objective weakness",
                 fontsize=15, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT / "slide5_results.png", bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    setup()
    for name, fn in [("slide1", make_slide1), ("slide2", make_slide2),
                     ("slide3", make_slide3), ("slide4", make_slide4),
                     ("slide5", make_slide5)]:
        print(f"  {name}...")
        fn()
    print(f"Done → {OUT}")


if __name__ == "__main__":
    main()
