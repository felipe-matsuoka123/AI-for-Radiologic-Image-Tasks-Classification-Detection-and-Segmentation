#!/usr/bin/env python3
"""Create a publication-ready segmentation mask comparison figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from PIL import Image


DEFAULT_MONTGOMERY_DIR = Path("/media/felipe/KINGSTON/datasets/NLM-MontgomeryCXRSet/MontgomerySet")
DEFAULT_SEGMENTATION_CHECKPOINT = Path("results/resnet34_unet_montgomery_lung.pt")
DEFAULT_OUTPUT = Path("images/publication_task_grid.png")

GROUND_TRUTH_COLOR = "#0072B2"
PREDICTION_COLOR = "#D55E00"
OVERLAP_COLOR = "#F0E442"
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def read_grayscale_png(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Could not read image: {path}")
    return np.asarray(Image.open(path).convert("L"))


def normalize_percentile(image: np.ndarray, lower: float = 1, upper: float = 99, eps: float = 1e-7) -> np.ndarray:
    image = image.astype(np.float32)
    lo, hi = np.percentile(image, [lower, upper])
    image = np.clip(image, lo, hi)
    return (image - lo) / (hi - lo + eps)


def resize_array(image: np.ndarray, size: int, resample: int) -> np.ndarray:
    return np.asarray(Image.fromarray(image).resize((size, size), resample=resample))


def hex_to_rgb01(color: str) -> np.ndarray:
    color = color.lstrip("#")
    return np.array([int(color[i : i + 2], 16) / 255 for i in (0, 2, 4)], dtype=np.float32)


def colorize_binary_mask(mask: np.ndarray, color: str) -> np.ndarray:
    rgb = np.zeros((*mask.shape, 3), dtype=np.float32)
    rgb[mask] = hex_to_rgb01(color)
    return rgb


def comparison_overlay(ground_truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    rgb = np.zeros((*ground_truth.shape, 3), dtype=np.float32)
    rgb[ground_truth] = hex_to_rgb01(GROUND_TRUTH_COLOR)
    rgb[prediction] = hex_to_rgb01(PREDICTION_COLOR)
    rgb[ground_truth & prediction] = hex_to_rgb01(OVERLAP_COLOR)
    return rgb


def lung_mask(montgomery_dir: Path, image_path: Path) -> np.ndarray:
    left = read_grayscale_png(montgomery_dir / "ManualMask" / "leftMask" / image_path.name) > 0
    right = read_grayscale_png(montgomery_dir / "ManualMask" / "rightMask" / image_path.name) > 0
    return left | right


def select_sample(image_dir: Path) -> Path:
    images = sorted(image_dir.glob("*.png"))
    if len(images) < 4:
        raise ValueError(f"Expected at least 4 Montgomery images in {image_dir}")
    return images[len(images) // 2]


def load_segmentation_model(checkpoint_path: Path):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class BasicBlock(nn.Module):
        expansion = 1

        def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: nn.Module | None = None):
            super().__init__()
            self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)
            self.downsample = downsample

        def forward(self, x):
            identity = x
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            if self.downsample is not None:
                identity = self.downsample(x)
            out += identity
            return self.relu(out)

    class ResNet34Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.inplanes = 64
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.layer1 = self._make_layer(64, 3)
            self.layer2 = self._make_layer(128, 4, stride=2)
            self.layer3 = self._make_layer(256, 6, stride=2)
            self.layer4 = self._make_layer(512, 3, stride=2)

        def _make_layer(self, planes: int, blocks: int, stride: int = 1):
            downsample = None
            if stride != 1 or self.inplanes != planes:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes),
                )
            layers = [BasicBlock(self.inplanes, planes, stride, downsample)]
            self.inplanes = planes
            layers.extend(BasicBlock(self.inplanes, planes) for _ in range(1, blocks))
            return nn.Sequential(*layers)

        def forward(self, x):
            x0 = x
            x1 = self.relu(self.bn1(self.conv1(x)))
            x2 = self.maxpool(x1)
            x2 = self.layer1(x2)
            x3 = self.layer2(x2)
            x4 = self.layer3(x3)
            x5 = self.layer4(x4)
            return x0, x1, x2, x3, x4, x5

    class DecoderBlock(nn.Module):
        def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        def forward(self, x, skip):
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            return self.block(x)

    class ResNet34UNet(nn.Module):
        def __init__(self, out_channels: int = 1):
            super().__init__()
            self.encoder = ResNet34Encoder()
            self.decoder4 = DecoderBlock(512, 256, 256)
            self.decoder3 = DecoderBlock(256, 128, 128)
            self.decoder2 = DecoderBlock(128, 64, 64)
            self.decoder1 = DecoderBlock(64, 64, 32)
            self.decoder0 = DecoderBlock(32, 3, 32)
            self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)

        def forward(self, x):
            x0, x1, x2, x3, x4, x5 = self.encoder(x)
            x = self.decoder4(x5, x4)
            x = self.decoder3(x, x3)
            x = self.decoder2(x, x2)
            x = self.decoder1(x, x1)
            x = self.decoder0(x, x0)
            return self.final_conv(x)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = ResNet34UNet()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, int(checkpoint.get("image_size", 512)), torch


def predict_mask(model, torch_module, image: np.ndarray, size: int, threshold: float = 0.5) -> np.ndarray:
    resized = resize_array(image, size, Image.Resampling.BILINEAR)
    normalized = normalize_percentile(resized).astype(np.float32)
    model_input = np.repeat(normalized[..., None], 3, axis=-1)
    model_input = (model_input - IMAGENET_MEAN) / IMAGENET_STD
    model_input = np.ascontiguousarray(model_input.transpose(2, 0, 1)[None, ...])
    tensor = torch_module.from_numpy(model_input).float()
    with torch_module.no_grad():
        probability = torch_module.sigmoid(model(tensor))[0, 0].cpu().numpy()
    return probability > threshold


def style_axis(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def make_figure(montgomery_dir: Path, checkpoint_path: Path, output_path: Path, dpi: int) -> None:
    image_dir = montgomery_dir / "CXR_png"
    if not image_dir.exists():
        raise FileNotFoundError(f"Montgomery image directory not found: {image_dir}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Segmentation checkpoint not found: {checkpoint_path}")

    image_path = select_sample(image_dir)
    raw_image = read_grayscale_png(image_path)
    model, image_size, torch_module = load_segmentation_model(checkpoint_path)
    ground_truth = resize_array(
        lung_mask(montgomery_dir, image_path).astype(np.uint8),
        image_size,
        Image.Resampling.NEAREST,
    ) > 0
    prediction = predict_mask(model, torch_module, raw_image, image_size)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 13,
            "figure.dpi": dpi,
            "savefig.dpi": dpi,
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(10.6, 4.7))
    fig.patch.set_facecolor("white")

    panels = [
        ("Ground Truth", colorize_binary_mask(ground_truth, GROUND_TRUTH_COLOR)),
        ("Predicted Mask", colorize_binary_mask(prediction, PREDICTION_COLOR)),
        ("Ground Truth + Prediction", comparison_overlay(ground_truth, prediction)),
    ]
    for ax, (title, panel) in zip(axes, panels):
        ax.imshow(panel, vmin=0, vmax=1)
        ax.set_title(title, pad=8)
        style_axis(ax)

    legend_handles = [
        Patch(facecolor=GROUND_TRUTH_COLOR, edgecolor="none", label="Ground truth"),
        Patch(facecolor=PREDICTION_COLOR, edgecolor="none", label="Prediction"),
        Patch(facecolor=OVERLAP_COLOR, edgecolor="none", label="Overlap"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.02),
        fontsize=10,
    )

    fig.subplots_adjust(left=0.015, right=0.995, top=0.87, bottom=0.18, wspace=0.04)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--montgomery-dir", type=Path, default=DEFAULT_MONTGOMERY_DIR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_SEGMENTATION_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    make_figure(args.montgomery_dir, args.checkpoint, args.output, args.dpi)
    print(f"Saved figure to {args.output}")
    print(f"Saved PDF to {args.output.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
