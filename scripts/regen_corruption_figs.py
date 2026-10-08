#!/usr/bin/env python3
"""Regenerate the corruption-robustness figures from the FINAL leakage-free CSVs.

Replaces the stale (8-Jun) figures that were built from a defunct
corruption_results.json path and the biased self-relative mCE.

Inputs (produced by scripts/22_corruption_robustness.py):
  results/corruption_robustness.csv          per (detector, corruption): ce_shared, ...
  results/corruption_robustness_detail.csv   per (detector, corruption, severity): map50

Outputs (match the manuscript captions in paper/main.tex):
  paper/figs/fig_corruption_heatmap.pdf   detector x corruption CE_shared heatmap
                                          (darker = higher corruption error;
                                           RT-DETR uniformly lightest)
  paper/figs/fig_corruption_curves.pdf    mAP@0.5 vs severity, one panel per
                                          corruption, four detector lines
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
OUT = REPO / "paper" / "figs"
OUT.mkdir(parents=True, exist_ok=True)

DETECTORS = ["yolov8m", "rtdetr", "faster_rcnn", "retinanet"]
DET_LABEL = {"yolov8m": "YOLOv8m", "rtdetr": "RT-DETR",
             "faster_rcnn": "Faster R-CNN", "retinanet": "RetinaNet"}
DET_COLOR = {"yolov8m": "#004EA6", "rtdetr": "#C00000",
             "faster_rcnn": "#007038", "retinanet": "#E07020"}
CORRUPTIONS = ["gaussian_noise", "poisson_noise", "impulse_noise",
               "motion_blur", "fog", "resolution"]
CORR_SHORT = {"gaussian_noise": "G-Noise", "poisson_noise": "P-Noise",
              "impulse_noise": "I-Noise", "motion_blur": "Blur",
              "fog": "Fog", "resolution": "Res."}
CORR_LABEL = {"gaussian_noise": "Gaussian Noise", "poisson_noise": "Poisson Noise",
              "impulse_noise": "Impulse Noise", "motion_blur": "Motion Blur",
              "fog": "Fog", "resolution": "Resolution"}
SEVERITIES = [1, 2, 3, 4]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "lines.linewidth": 1.5, "axes.linewidth": 0.8,
    "xtick.direction": "in", "ytick.direction": "in",
    "figure.dpi": 200, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05, "savefig.dpi": 300,
})


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        sys.exit(f"Missing input: {path}")
    with open(path) as f:
        return list(csv.DictReader(f))


def heatmap() -> None:
    rows = _read_csv(RESULTS / "corruption_robustness.csv")
    ce = defaultdict(dict)
    for r in rows:
        if r["corruption"] in CORRUPTIONS:
            ce[r["detector"]][r["corruption"]] = float(r["ce_shared"])
    mat = np.array([[ce[d][c] for c in CORRUPTIONS] for d in DETECTORS])

    fig, ax = plt.subplots(figsize=(6.2, 2.4))
    # light = low error (robust), dark = high error (fragile)
    im = ax.imshow(mat, cmap="YlOrRd", aspect="auto",
                   vmin=float(mat.min()), vmax=float(mat.max()))
    ax.set_xticks(range(len(CORRUPTIONS)))
    ax.set_xticklabels([CORR_SHORT[c] for c in CORRUPTIONS])
    ax.set_yticks(range(len(DETECTORS)))
    ax.set_yticklabels([DET_LABEL[d] for d in DETECTORS])
    thr = 0.5 * (mat.min() + mat.max())
    for i in range(len(DETECTORS)):
        for j in range(len(CORRUPTIONS)):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                    fontsize=7, color="black" if mat[i, j] < thr else "white")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label(r"$\mathrm{CE_{shared}}$ (lower = more robust)", fontsize=8)
    ax.set_title("Corruption error by detector and corruption type", fontsize=9)
    fig.tight_layout()
    out = OUT / "fig_corruption_heatmap.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=200)
    plt.close(fig)
    print(f"[heatmap] Saved: {out}")


def curves() -> None:
    rows = _read_csv(RESULTS / "corruption_robustness_detail.csv")
    m = defaultdict(dict)  # (det, corr) -> {sev: map50}
    for r in rows:
        m[(r["detector"], r["corruption"])][int(r["severity"])] = float(r["map50"])

    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.4), sharex=True, sharey=True)
    for ax, corr in zip(axes.ravel(), CORRUPTIONS):
        for d in DETECTORS:
            ys = [m[(d, corr)].get(s, np.nan) for s in SEVERITIES]
            ax.plot(SEVERITIES, ys, "o-", color=DET_COLOR[d],
                    label=DET_LABEL[d], markersize=3)
        ax.set_title(CORR_LABEL[corr], fontsize=9)
        ax.set_xticks(SEVERITIES)
        ax.grid(alpha=0.3)
        ax.set_ylim(-0.02, 0.85)
    for ax in axes[-1, :]:
        ax.set_xlabel("Severity")
    for ax in axes[:, 0]:
        ax.set_ylabel("mAP@0.5")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4,
               bbox_to_anchor=(0.5, 1.06), frameon=False)
    fig.tight_layout()
    out = OUT / "fig_corruption_curves.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[curves]  Saved: {out}")


if __name__ == "__main__":
    heatmap()
    curves()
    print("Done.")
