#!/usr/bin/env python3
"""Create a publication-ready qualitative YOLO detection figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch, Rectangle
from PIL import Image


DEFAULT_YOLO_DATASET_DIR = Path("notebooks/results/montgomery_lung_yolo")
DEFAULT_WEIGHTS = Path("notebooks/results/yolo_detection_runs/yolo26n_montgomery_lung_boxes/weights/best.pt")
DEFAULT_OUTPUT = Path("images/yolo_qualitative_detection.png")

GROUND_TRUTH_COLOR = "#0072B2"
PREDICTION_COLOR = "#D55E00"


def read_grayscale_png(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Could not read image: {path}")
    return np.asarray(Image.open(path).convert("L"))


def normalize_percentile(image: np.ndarray, lower: float = 1, upper: float = 99, eps: float = 1e-7) -> np.ndarray:
    image = image.astype(np.float32)
    lo, hi = np.percentile(image, [lower, upper])
    image = np.clip(image, lo, hi)
    return (image - lo) / (hi - lo + eps)


def yolo_label_to_xyxy(label_path: Path, image_width: int, image_height: int) -> np.ndarray:
    boxes = []
    for line in label_path.read_text().replace("\\n", "\n").splitlines():
        if not line.strip():
            continue
        _, x_center, y_center, width, height = map(float, line.split())
        x_center *= image_width
        y_center *= image_height
        width *= image_width
        height *= image_height
        boxes.append(
            [
                x_center - width / 2,
                y_center - height / 2,
                x_center + width / 2,
                y_center + height / 2,
            ]
        )
    return np.asarray(boxes, dtype=np.float32)


def draw_boxes(
    ax,
    boxes: np.ndarray,
    color: str,
    *,
    linewidth: float = 2.2,
    scores: np.ndarray | None = None,
) -> None:
    for idx, box in enumerate(boxes):
        x_min, y_min, x_max, y_max = box
        ax.add_patch(
            Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                linewidth=linewidth,
                edgecolor=color,
                facecolor="none",
            )
        )
        if scores is not None:
            ax.text(
                x_min,
                y_min,
                f"{scores[idx]:.2f}",
                color=color,
                fontsize=8,
                ha="left",
                va="bottom",
                bbox={"facecolor": "black", "edgecolor": "none", "alpha": 0.55, "pad": 1.5},
            )


def style_axis(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def test_image_paths(dataset_dir: Path) -> list[Path]:
    image_dir = dataset_dir / "images" / "test"
    if not image_dir.exists():
        raise FileNotFoundError(f"YOLO test image directory not found: {image_dir}")
    images = sorted(image_dir.glob("*.png"))
    if not images:
        raise FileNotFoundError(f"No PNG images found under {image_dir}")
    return images


def result_boxes_and_scores(result, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    if result.boxes is None or len(result.boxes) == 0:
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)
    boxes = result.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
    scores = result.boxes.conf.detach().cpu().numpy().astype(np.float32)
    order = np.argsort(scores)[::-1][:top_k]
    return boxes[order], scores[order]


def choose_prediction_example(model, candidates: list[Path], image_size: int, confidence: float, top_k: int, max_candidates: int):
    subset = candidates[:max_candidates]
    results = model.predict(
        source=[str(path) for path in subset],
        imgsz=image_size,
        conf=confidence,
        save=False,
        verbose=False,
    )
    selected = 0
    selected_boxes = np.empty((0, 4), dtype=np.float32)
    selected_scores = np.empty((0,), dtype=np.float32)
    for idx, result in enumerate(results):
        boxes, scores = result_boxes_and_scores(result, top_k=top_k)
        if len(boxes) >= min(top_k, 2):
            selected = idx
            selected_boxes = boxes
            selected_scores = scores
            break
    else:
        selected_boxes, selected_scores = result_boxes_and_scores(results[0], top_k=top_k)
    return subset[selected], selected_boxes, selected_scores


def make_figure(
    dataset_dir: Path,
    weights_path: Path,
    output_path: Path,
    image_size: int,
    confidence: float,
    top_k: int,
    max_candidates: int,
    dpi: int,
) -> None:
    from ultralytics import YOLO

    if not weights_path.exists():
        raise FileNotFoundError(f"YOLO weights not found: {weights_path}")

    model = YOLO(str(weights_path))
    image_path, pred_boxes, pred_scores = choose_prediction_example(
        model,
        test_image_paths(dataset_dir),
        image_size=image_size,
        confidence=confidence,
        top_k=top_k,
        max_candidates=max_candidates,
    )
    label_path = dataset_dir / "labels" / "test" / f"{image_path.stem}.txt"
    image = read_grayscale_png(image_path)
    image_height, image_width = image.shape[:2]
    true_boxes = yolo_label_to_xyxy(label_path, image_width=image_width, image_height=image_height)
    display_image = normalize_percentile(image)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "figure.dpi": dpi,
            "savefig.dpi": dpi,
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(10.8, 4.4))
    fig.patch.set_facecolor("white")

    panels = [
        ("Ground Truth", True, False),
        ("YOLO Prediction", False, True),
        ("Ground Truth + Prediction", True, True),
    ]
    for ax, (title, show_gt, show_pred) in zip(axes, panels):
        ax.imshow(display_image, cmap="gray")
        if show_gt:
            draw_boxes(ax, true_boxes, GROUND_TRUTH_COLOR)
        if show_pred:
            draw_boxes(ax, pred_boxes, PREDICTION_COLOR, scores=pred_scores)
        ax.set_title(title, pad=7)
        style_axis(ax)

    legend_handles = [
        Patch(facecolor="none", edgecolor=GROUND_TRUTH_COLOR, linewidth=2.2, label="Ground truth"),
        Patch(facecolor="none", edgecolor=PREDICTION_COLOR, linewidth=2.2, label="YOLO prediction"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.02),
        fontsize=10,
    )

    fig.subplots_adjust(left=0.015, right=0.995, top=0.88, bottom=0.18, wspace=0.04)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_YOLO_DATASET_DIR)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--confidence", type=float, default=0.10)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--max-candidates", type=int, default=12)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    make_figure(
        dataset_dir=args.dataset_dir,
        weights_path=args.weights,
        output_path=args.output,
        image_size=args.image_size,
        confidence=args.confidence,
        top_k=args.top_k,
        max_candidates=args.max_candidates,
        dpi=args.dpi,
    )
    print(f"Saved figure to {args.output}")
    print(f"Saved PDF to {args.output.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
