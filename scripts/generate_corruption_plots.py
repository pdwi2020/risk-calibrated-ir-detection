"""Generate corruption robustness figures for the MV Paper (Phase 2).

Reads per-detector corruption_results.json files produced by
scripts/10_run_corruption_eval.py and generates publication-quality figures:

    fig_corruption_heatmap_{detector}.pdf   — 6×4 cell heatmap (mAP@0.5),
                                               one per detector
    fig_corruption_curves.pdf               — mAP vs severity line plots,
                                               one panel per corruption type,
                                               4 detector lines each
    fig_corruption_mce_bar.pdf              — mCE grouped bar chart
                                               (6 corruption types × 4 detectors)

Run from project root after Phase 2 GPU run:
    python scripts/generate_corruption_plots.py

Outputs go to paper/figs/ (same location as generate_figures.py).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

RESULTS = REPO / "results"
OUT = REPO / "paper" / "figs"
OUT.mkdir(parents=True, exist_ok=True)

# ── Detector configs ───────────────────────────────────────────────────────────
DETECTORS = {
    "yolov8m":     {"label": "YOLOv8m",    "color": "#004EA6"},
    "rtdetr":      {"label": "RT-DETR",    "color": "#C00000"},
    "faster_rcnn": {"label": "F-RCNN",     "color": "#007038"},
    "retinanet":   {"label": "RetinaNet",  "color": "#E07020"},
}

CORRUPTION_LABELS = {
    "gaussian_noise": "Gauss. Noise",
    "poisson_noise":  "Poisson Noise",
    "impulse_noise":  "Impulse Noise",
    "motion_blur":    "Motion Blur",
    "fog":            "Fog",
    "resolution":     "Resolution",
}
SEVERITIES = [1, 2, 3, 4]

# ── IEEE-style rc params ──────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size":         9,
    "axes.labelsize":    9,
    "axes.titlesize":    9,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   8,
    "lines.linewidth":   1.5,
    "axes.linewidth":    0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.direction":  "in",
    "ytick.direction":  "in",
    "figure.dpi":       200,
    "savefig.bbox":     "tight",
    "savefig.pad_inches": 0.05,
    "savefig.dpi":      300,
})


def clean_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_results(det_name: str) -> dict | None:
    """Load per-detector corruption_results.json; return None if not yet produced."""
    path = RESULTS / f"{det_name}_flir_seed0" / "eval" / "corruption" / "corruption_results.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _extract_map_matrix(data: dict) -> tuple[list[str], np.ndarray]:
    """Return (corruption_names, matrix[6 corruptions × 4 severities]) of mAP@0.5."""
    names = list(CORRUPTION_LABELS.keys())
    mat = np.zeros((len(names), len(SEVERITIES)), dtype=np.float32)
    for i, cname in enumerate(names):
        for j, sev in enumerate(SEVERITIES):
            mat[i, j] = data["corruptions"][cname][str(sev)]["mAP50"]
    return names, mat


# ── Figure 1: per-detector heatmaps ──────────────────────────────────────────

def plot_heatmaps(all_data: dict[str, dict]) -> None:
    """4 side-by-side heatmaps (one per detector), 6 corruptions × 4 severities."""
    n_det = len(all_data)
    if n_det == 0:
        print("[heatmap] No data available — skipping.")
        return

    fig, axes = plt.subplots(1, n_det, figsize=(2.3 * n_det, 2.8), sharey=True)
    if n_det == 1:
        axes = [axes]

    corr_names = list(CORRUPTION_LABELS.keys())
    corr_labels = [CORRUPTION_LABELS[c] for c in corr_names]

    vmin, vmax = 0.0, 1.0
    # Compute global vmin from data for better contrast
    all_maps = [all_data[d]["corruptions"][c][str(s)]["mAP50"]
                for d in all_data
                for c in corr_names
                for s in SEVERITIES]
    if all_maps:
        vmin = max(0.0, min(all_maps) - 0.02)

    for ax, (det_name, data) in zip(axes, all_data.items()):
        _, mat = _extract_map_matrix(data)
        clean_map = data["clean"]["mAP50"]
        im = ax.imshow(mat, vmin=vmin, vmax=vmax, cmap="RdYlGn", aspect="auto")

        ax.set_xticks(range(4))
        ax.set_xticklabels([f"s{s}" for s in SEVERITIES], fontsize=7)
        if ax is axes[0]:
            ax.set_yticks(range(len(corr_labels)))
            ax.set_yticklabels(corr_labels, fontsize=7)
        else:
            ax.set_yticks([])

        ax.set_title(f"{DETECTORS[det_name]['label']}\n"
                     f"(clean={clean_map:.3f})", fontsize=8)

        # Annotate cells
        for i in range(len(corr_names)):
            for j in range(4):
                val = mat[i, j]
                color = "black" if val > (vmin + vmax) / 2 else "white"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=6, color=color)

    fig.colorbar(im, ax=axes[-1], fraction=0.046, pad=0.04,
                 label="mAP@0.5")
    fig.suptitle("Corruption Robustness — mAP@0.5 per Severity",
                 fontsize=9, y=1.02)
    fig.tight_layout()
    out = OUT / "fig_corruption_heatmap.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"[heatmap] Saved: {out}")


def plot_heatmap_per_detector(all_data: dict[str, dict]) -> None:
    """Individual heatmap PDF per detector (for supplemental)."""
    corr_names = list(CORRUPTION_LABELS.keys())
    corr_labels = [CORRUPTION_LABELS[c] for c in corr_names]

    for det_name, data in all_data.items():
        _, mat = _extract_map_matrix(data)
        clean_map = data["clean"]["mAP50"]
        fig, ax = plt.subplots(figsize=(3.2, 2.6))
        vmin = max(0.0, mat.min() - 0.02)
        im = ax.imshow(mat, vmin=vmin, vmax=1.0, cmap="RdYlGn", aspect="auto")

        ax.set_xticks(range(4))
        ax.set_xticklabels([f"Sev {s}" for s in SEVERITIES])
        ax.set_yticks(range(len(corr_labels)))
        ax.set_yticklabels(corr_labels)
        ax.set_title(
            f"{DETECTORS[det_name]['label']} — Corruption Robustness\n"
            f"Clean mAP@0.5 = {clean_map:.3f}", fontsize=9
        )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="mAP@0.5")

        for i in range(len(corr_names)):
            for j in range(4):
                val = mat[i, j]
                mid = (max(0.0, mat.min() - 0.02) + 1.0) / 2
                color = "black" if val > mid else "white"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=7, color=color)

        fig.tight_layout()
        out = OUT / f"fig_corruption_heatmap_{det_name}.pdf"
        fig.savefig(out)
        plt.close(fig)
        print(f"[heatmap] Saved: {out}")


# ── Figure 2: mAP degradation curves ─────────────────────────────────────────

def plot_degradation_curves(all_data: dict[str, dict]) -> None:
    """6-panel figure: one panel per corruption type, 4 detector lines per panel."""
    corr_names = list(CORRUPTION_LABELS.keys())
    n = len(corr_names)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5, nrows * 1.9))
    axes_flat = list(axes.flat) if hasattr(axes, "flat") else [axes]

    for idx, cname in enumerate(corr_names):
        ax = axes_flat[idx]
        clean_lines = []

        for det_name, data in all_data.items():
            cfg = DETECTORS[det_name]
            clean_val = data["clean"]["mAP50"]
            sev_vals = [data["corruptions"][cname][str(s)]["mAP50"] for s in SEVERITIES]
            x = SEVERITIES
            ax.plot(x, sev_vals, color=cfg["color"], marker="o", markersize=3,
                    label=cfg["label"])
            ax.axhline(clean_val, color=cfg["color"], linestyle="--",
                       linewidth=0.7, alpha=0.5)
            clean_lines.append(clean_val)

        ax.set_title(CORRUPTION_LABELS[cname], fontsize=8)
        ax.set_xlabel("Severity", fontsize=7)
        ax.set_ylabel("mAP@0.5", fontsize=7)
        ax.set_xticks(SEVERITIES)
        ax.set_ylim(0, 1)
        clean_ax(ax)
        if idx == 0:
            ax.legend(fontsize=6, loc="lower left")

    # Hide empty panels
    for idx in range(n, nrows * ncols):
        axes_flat[idx].set_visible(False)

    fig.suptitle("mAP@0.5 Degradation vs. Corruption Severity\n"
                 "(dashed = clean baseline per detector)", fontsize=9)
    fig.tight_layout()
    out = OUT / "fig_corruption_curves.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"[curves]  Saved: {out}")


# ── Figure 3: mCE grouped bar chart ──────────────────────────────────────────

def plot_mce_bars(all_data: dict[str, dict]) -> None:
    """Grouped bar chart: mCE per corruption type, one group per corruption."""
    corr_names = list(CORRUPTION_LABELS.keys())
    det_names = list(all_data.keys())
    n_det = len(det_names)
    n_corr = len(corr_names)

    x = np.arange(n_corr)
    width = 0.18
    offsets = np.linspace(-(n_det - 1) / 2.0, (n_det - 1) / 2.0, n_det) * width

    fig, ax = plt.subplots(figsize=(6.5, 2.6))

    for i, det_name in enumerate(det_names):
        cfg = DETECTORS[det_name]
        mce_vals = [all_data[det_name]["mce"][c] for c in corr_names]
        bars = ax.bar(x + offsets[i], mce_vals, width,
                      color=cfg["color"], label=cfg["label"], alpha=0.85)

    ax.axhline(1.0, color="gray", linewidth=0.8, linestyle="--", label="mCE=1 (no robustness)")
    ax.set_xticks(x)
    ax.set_xticklabels([CORRUPTION_LABELS[c] for c in corr_names],
                       rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("mCE (lower is better)", fontsize=9)
    ax.set_ylim(0, None)
    ax.legend(fontsize=8, ncol=2)
    ax.set_title("Mean Corruption Error (mCE) by Corruption Type", fontsize=9)
    clean_ax(ax)

    # Annotate mean mCE per detector in legend area
    mean_text = "  ".join(
        f"{DETECTORS[d]['label']}: {all_data[d]['mean_mce']:.3f}"
        for d in det_names
    )
    ax.text(0.02, 0.98, f"Mean mCE — {mean_text}",
            transform=ax.transAxes, fontsize=7, va="top",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

    fig.tight_layout()
    out = OUT / "fig_corruption_mce_bar.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"[mce bar] Saved: {out}")


# ── Figure 4: per-class degradation (optional, YOLOv8m only) ─────────────────

def plot_per_class_degradation(data: dict) -> None:
    """If per-class mAP data is available, plot person/bicycle/car degradation."""
    # Per-class data is not produced by the current run_corruption_eval —
    # this is a stub for future extension.
    pass


# ── Summary table printer ─────────────────────────────────────────────────────

def print_summary_table(all_data: dict[str, dict]) -> None:
    """Print a LaTeX-ready summary to stdout."""
    corr_names = list(CORRUPTION_LABELS.keys())
    det_names = list(all_data.keys())

    print("\n% ── Corruption mCE summary (LaTeX table body) ──────────────────")
    print(r"\midrule")
    for det_name in det_names:
        data = all_data[det_name]
        label = DETECTORS[det_name]["label"]
        clean = data["clean"]["mAP50"]
        mces = [data["mce"][c] for c in corr_names]
        mean_mce = data["mean_mce"]
        row = f"{label} & {clean:.3f}"
        row += "".join(f" & {m:.3f}" for m in mces)
        row += f" & {mean_mce:.3f} \\\\"
        print(row)
    print(r"\bottomrule")
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading corruption results ...")
    all_data: dict[str, dict] = {}
    for det_name in DETECTORS:
        data = _load_results(det_name)
        if data is None:
            print(f"  [{det_name}] No results yet — skipping "
                  f"(run 10_run_corruption_eval.py first).")
        else:
            all_data[det_name] = data
            print(f"  [{det_name}] loaded — clean mAP={data['clean']['mAP50']:.4f}  "
                  f"mean_mCE={data['mean_mce']:.4f}")

    if not all_data:
        print("\nNo corruption results available yet. Run:")
        print("  python scripts/10_run_corruption_eval.py")
        return

    print(f"\nGenerating figures → {OUT}")
    plot_heatmap_per_detector(all_data)
    plot_heatmaps(all_data)
    plot_degradation_curves(all_data)
    plot_mce_bars(all_data)
    print_summary_table(all_data)
    print(f"\nDone. PDFs in {OUT}/")


if __name__ == "__main__":
    main()
