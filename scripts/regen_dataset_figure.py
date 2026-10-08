#!/usr/bin/env python3
"""P14 — Dataset & corruption-processing qualitative figure.

Loads a real FLIR ADAS v2 thermal image and applies the paper's own corruption
pipeline (src/corruption/corruption_pipeline.py) so the figure shows exactly the
degradations used in the robustness experiments. Two rows:
  (top)    clean + all six corruption types at severity 3, SNR-labelled
  (bottom) the same scene under Gaussian noise at severities 1--4 (severity ramp)

Output: paper/figs/fig_dataset_corruptions.pdf  (pdf.fonttype=42 -> no Type-3)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from src.corruption.corruption_pipeline import CORRUPTION_REGISTRY, image_snr_db  # noqa: E402

# thermal val images: $FLIR_ROOT if set, else the repo's datasets/ folder
FLIR_ROOT = Path(os.environ.get("FLIR_ROOT", REPO / "datasets/flir_adas_v2/FLIR_ADAS_v2"))
IMG_DIRS = [FLIR_ROOT / "images_thermal_val/data"]
OUT = REPO / "paper" / "figs" / "fig_dataset_corruptions.pdf"
SEED = 42  # matches experiment corruption seed
# hand-picked frame with a clear street scene + pedestrians/vehicles
IMG_NAME = "video-2SI21mausmR2SM8sm-frame-001642-A8t6nMWtDrgtHKZbc.jpg"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "cm", "font.size": 8,
    "pdf.fonttype": 42, "ps.fonttype": 42,   # embed TrueType -> no Type-3
    "savefig.dpi": 300, "savefig.bbox": "tight",
})

CORRS = ["gaussian_noise", "poisson_noise", "impulse_noise",
         "motion_blur", "fog", "resolution"]
LABEL = {"gaussian_noise": "Gaussian", "poisson_noise": "Poisson",
         "impulse_noise": "Impulse", "motion_blur": "Motion blur",
         "fog": "Fog", "resolution": "Resolution"}
CORR_IDX = {n: i for i, n in enumerate(CORRUPTION_REGISTRY)}


def _rng(cname: str, sev: int, i: int = 0) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([SEED, CORR_IDX[cname], sev, i]))


def _apply(cname: str, img: np.ndarray, sev: int) -> np.ndarray:
    return CORRUPTION_REGISTRY[cname](img, sev, _rng(cname, sev))


def _load() -> np.ndarray:
    for d in IMG_DIRS:
        p = d / IMG_NAME
        if p.exists():
            return np.array(Image.open(p).convert("L"))
        # fallback: first image in the dir
        if d.exists():
            imgs = sorted(d.glob("*.jpg"))
            if imgs:
                print(f"[warn] {IMG_NAME} not found; using {imgs[0].name}")
                return np.array(Image.open(imgs[0]).convert("L"))
    sys.exit("No thermal val images found.")


def _show(ax, img, title):
    ax.imshow(img, cmap="gray", vmin=0, vmax=255)
    ax.set_title(title, fontsize=8, pad=2)
    ax.set_xticks([]); ax.set_yticks([])


def main() -> None:
    clean = _load()

    fig, axes = plt.subplots(2, 7, figsize=(12.5, 4.0))

    # each row = clean + 6 corruptions at a fixed severity (moderate then severe)
    for row, sev in enumerate([2, 4]):
        _show(axes[row, 0], clean, "Clean")
        for j, c in enumerate(CORRS, start=1):
            corr = _apply(c, clean, sev)
            snr = image_snr_db(clean, corr)
            _show(axes[row, j], corr, f"{LABEL[c]}\nSNR {snr:.1f} dB")

    fig.text(0.5, 0.965, "Corruption types at severity 2 (moderate)", ha="center", fontsize=9)
    fig.text(0.5, 0.475, "Corruption types at severity 4 (severe)", ha="center", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    fig.subplots_adjust(hspace=0.35)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    fig.savefig(OUT.with_suffix(".png"), dpi=150)
    plt.close(fig)
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
