#!/usr/bin/env python3
"""Generate a professional TAB diagram for the ICLR slide deck.

Message of the figure:
Human productions and LLM outputs can be measured in the same behavioral
symptom space, which lets us ask alignment questions through failure behavior.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches


OUT = Path(__file__).resolve().parents[1] / "figures" / "iclr_5slide"

BG = "#fbfaf7"
INK = "#111111"
MUTED = "#6f6b63"
LINE = "#c9c5ba"
HUMAN = "#5f6368"
MODEL = "#0072b2"
SCORER = "#e69f00"
SPACE = "#111111"
ACCENT = "#009e73"


def add_box(ax, x, y, w, h, label, *, edge, face="white", color=INK,
            fs=10.5, weight="bold"):
    box = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        linewidth=1.8,
        edgecolor=edge,
        facecolor=face,
        transform=ax.transAxes,
        clip_on=False,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2, y + h / 2, label,
        ha="center", va="center",
        fontsize=fs, fontweight=weight,
        color=color,
        transform=ax.transAxes,
        linespacing=1.15,
    )


def arrow(ax, x0, y0, x1, y1, *, color=MUTED, lw=1.7):
    ax.annotate(
        "",
        xy=(x1, y1),
        xytext=(x0, y0),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", lw=lw, color=color, shrinkA=4, shrinkB=4),
    )


def symptom_vector(ax, x, y, w, h):
    add_box(
        ax, x, y, w, h,
        "shared 19-symptom space\nTAB deficit coordinates",
        edge=SPACE,
        face="white",
        fs=10.5,
    )
    colors = ["#0072b2", "#009e73", "#e69f00", "#b33c00", "#7b2d8e"]
    bar_x0 = x + 0.10
    bar_y = y + 0.030
    bar_w = (w - 0.20) / len(colors)
    for i, c in enumerate(colors):
        bx = bar_x0 + i * bar_w
        ax.add_patch(
            patches.Rectangle(
                (bx, bar_y), bar_w * 0.72, 0.025,
                transform=ax.transAxes,
                facecolor=c,
                edgecolor="none",
                alpha=0.9,
            )
        )


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.2, 7.8))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Layout coordinates.
    left_x, right_x = 0.07, 0.56
    box_w, box_h = 0.37, 0.10
    y_source, y_measure, y_score = 0.78, 0.59, 0.40

    # Column headers.
    ax.text(left_x + box_w / 2, 0.92, "human behavior", ha="center", fontsize=11,
            color=HUMAN, fontweight="bold", transform=ax.transAxes)
    ax.text(right_x + box_w / 2, 0.92, "model behavior", ha="center", fontsize=11,
            color=MODEL, fontweight="bold", transform=ax.transAxes)

    # Human lane.
    add_box(
        ax, left_x, y_source, box_w, box_h,
        "AphasiaBank\npatients + controls",
        edge=HUMAN,
        face="#f4f3f0",
        fs=10.5,
    )
    add_box(
        ax, left_x, y_measure, box_w, box_h,
        "expert SLP\nannotations",
        edge=HUMAN,
        face="#f4f3f0",
        fs=10.5,
    )
    add_box(
        ax, left_x, y_score, box_w, box_h,
        "human symptom\nprofiles",
        edge=ACCENT,
        face="#f7fff9",
        fs=10.5,
    )

    # Model lane.
    add_box(
        ax, right_x, y_source, box_w, box_h,
        "LLM outputs\nintact + lesioned",
        edge=MODEL,
        face="#f2f8fc",
        fs=10.5,
    )
    add_box(
        ax, right_x, y_measure, box_w, box_h,
        "TAB + scorer\nvalidated vs SLPs",
        edge=SCORER,
        face="#fff8e8",
        fs=10.5,
    )
    add_box(
        ax, right_x, y_score, box_w, box_h,
        "model symptom\nprofiles",
        edge=ACCENT,
        face="#f7fff9",
        fs=10.5,
    )

    # Vertical arrows.
    for x in (left_x + box_w / 2, right_x + box_w / 2):
        arrow(ax, x, y_source, x, y_measure + box_h)
        arrow(ax, x, y_measure, x, y_score + box_h)

    # Shared symptom space.
    sx, sy, sw, sh = 0.16, 0.10, 0.68, 0.18
    symptom_vector(ax, sx, sy, sw, sh)

    # Converging arrows.
    arrow(ax, left_x + box_w / 2, y_score, sx + sw * 0.30, sy + sh)
    arrow(ax, right_x + box_w / 2, y_score, sx + sw * 0.70, sy + sh)

    # Alignment comparison label.
    ax.text(
        0.5,
        0.035,
        "behavioral alignment: same failure coordinates?",
        ha="center",
        va="center",
        fontsize=9.5,
        color=MUTED,
        transform=ax.transAxes,
    )

    path = OUT / "tab_diagram.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Done -> {path}")


if __name__ == "__main__":
    main()
