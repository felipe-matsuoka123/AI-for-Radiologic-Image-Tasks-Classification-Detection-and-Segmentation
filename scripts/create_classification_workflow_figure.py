#!/usr/bin/env python3
"""Create a publication-ready classification workflow figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


DEFAULT_OUTPUT = Path("images/classification_workflow.png")


def add_box(
    ax,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    *,
    facecolor: str = "#dbeafe",
    edgecolor: str = "#1f2937",
    fontsize: int = 10,
) -> None:
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.045",
        linewidth=1.2,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color="#111827",
        linespacing=1.2,
    )


def add_arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    connectionstyle: str = "arc3,rad=0",
) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=1.1,
        color="#111827",
        connectionstyle=connectionstyle,
        shrinkA=4,
        shrinkB=4,
    )
    ax.add_patch(arrow)


def make_figure(output_path: Path, dpi: int) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "figure.dpi": dpi,
            "savefig.dpi": dpi,
        }
    )

    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    blue = "#bfdbfe"
    blue_dark = "#93c5fd"
    green = "#dcfce7"
    amber = "#fef3c7"
    gray = "#f3f4f6"

    add_box(ax, (0.04, 0.53), 0.16, 0.16, "Shenzhen\nradiographs", facecolor=blue_dark)
    add_box(ax, (0.04, 0.28), 0.16, 0.13, "Image-level labels\nNo TB / TB", facecolor=gray, fontsize=9)

    add_box(ax, (0.27, 0.53), 0.17, 0.16, "Train / validation /\ntest split", facecolor=blue)
    add_box(ax, (0.50, 0.53), 0.16, 0.16, "Image\npreprocessing", facecolor=blue)
    add_box(ax, (0.71, 0.53), 0.16, 0.16, "ResNet34\nclassifier", facecolor=blue_dark)
    add_box(ax, (0.71, 0.28), 0.16, 0.13, "Predicted TB\nprobability", facecolor=amber)

    add_box(ax, (0.50, 0.79), 0.16, 0.12, "Validation set\ncheckpoint selection", facecolor=green, fontsize=9)
    add_box(ax, (0.50, 0.12), 0.16, 0.12, "Held-out Shenzhen\ntest set", facecolor=green, fontsize=9)
    add_box(ax, (0.78, 0.12), 0.17, 0.12, "External Montgomery\nevaluation", facecolor=green, fontsize=9)

    add_arrow(ax, (0.20, 0.61), (0.27, 0.61))
    add_arrow(ax, (0.20, 0.35), (0.27, 0.56), connectionstyle="arc3,rad=-0.18")
    add_arrow(ax, (0.44, 0.61), (0.50, 0.61))
    add_arrow(ax, (0.66, 0.61), (0.71, 0.61))
    add_arrow(ax, (0.79, 0.53), (0.79, 0.41))

    add_arrow(ax, (0.355, 0.69), (0.50, 0.84), connectionstyle="arc3,rad=0.15")
    add_arrow(ax, (0.79, 0.28), (0.66, 0.18), connectionstyle="arc3,rad=-0.20")
    add_arrow(ax, (0.87, 0.34), (0.865, 0.24))

    ax.text(
        0.12,
        0.75,
        "Development cohort",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        color="#111827",
    )
    ax.text(
        0.585,
        0.05,
        "Internal evaluation",
        ha="center",
        va="center",
        fontsize=9,
        color="#374151",
    )
    ax.text(
        0.865,
        0.05,
        "External evaluation",
        ha="center",
        va="center",
        fontsize=9,
        color="#374151",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    make_figure(args.output, args.dpi)
    print(f"Saved figure to {args.output}")
    print(f"Saved PDF to {args.output.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
