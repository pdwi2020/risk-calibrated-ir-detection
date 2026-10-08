"""Publication-quality figures for the MV Paper (risk-calibrated IR detection).

Generates 5 figures from cached prediction JSONs on X9:
  fig1_map_comparison.pdf      -- mAP@.5 / mAP@.5:.95 bar chart (3 detectors)
  fig2_cost_curves.pdf         -- E[C(theta)] vs theta (YOLOv8m + FRCNN, 2 panels)
  fig3_reliability.pdf         -- reliability diagram: raw vs isotonic (YOLOv8m)
  fig4_ece_ablation.pdf        -- ECE grouped bar (4 methods x 2 detectors)
  fig5_conformal.pdf           -- conformal risk R_hat(lambda) vs lambda (YOLOv8m)

Run from repo root:
  python scripts/generate_figures.py
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

from src.risk.cost_sensitive import CostSensitiveThreshold
from src.risk.conformal import ConformalRiskController
from src.calibration.isotonic import IsotonicCalibrator
from src.calibration.temperature_scaling import DetectionTemperatureScaling

# ── Paths ─────────────────────────────────────────────────────────────────────
RESULTS = REPO / "results"
YOLO_CALIB  = RESULTS / "yolov8m_flir_seed0/eval/yolov8m_calib_predictions.json"
YOLO_TEST   = RESULTS / "yolov8m_flir_seed0/eval/yolov8m_test_predictions.json"
FRCNN_CALIB = RESULTS / "faster_rcnn_flir_seed0/eval/faster_rcnn_calibration_predictions.json"
FRCNN_TEST  = RESULTS / "faster_rcnn_flir_seed0/eval/faster_rcnn_test_predictions.json"
OUT = REPO / "paper/figs"
OUT.mkdir(parents=True, exist_ok=True)

# ── IEEE-style matplotlib settings ────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset":   "cm",
    "font.size":           9,
    "axes.labelsize":      9,
    "axes.titlesize":      9,
    "xtick.labelsize":     8,
    "ytick.labelsize":     8,
    "legend.fontsize":     8,
    "lines.linewidth":     1.5,
    "axes.linewidth":      0.8,
    "xtick.major.width":   0.8,
    "ytick.major.width":   0.8,
    "xtick.direction":    "in",
    "ytick.direction":    "in",
    "figure.dpi":         200,
    "savefig.bbox":       "tight",
    "savefig.pad_inches":  0.05,
    "savefig.dpi":        300,
})
IEEE_BLUE   = "#004EA6"
IEEE_RED    = "#C00000"
IEEE_GREEN  = "#007038"
IEEE_ORANGE = "#E07020"
GRAY        = "#888888"


def clean_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ── Data loading helpers ───────────────────────────────────────────────────────
def load_cached(path: Path) -> list:
    return json.loads(path.read_text())


def to_preds_gts(cached: list):
    preds = [{"boxes": r["pred_boxes"], "scores": r["pred_scores"],
               "labels": r["pred_labels"]} for r in cached]
    gts   = [{"boxes": r["gt_boxes"],  "labels": r["gt_labels"]}  for r in cached]
    return preds, gts


# ── IoU helper ────────────────────────────────────────────────────────────────
def _iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2]-a[0]) * (a[3]-a[1])
    area_b = (b[2]-b[0]) * (b[3]-b[1])
    return inter / (area_a + area_b - inter)


def matched_tp_pairs(cached: list, iou_thr: float = 0.5, min_conf: float = 0.0):
    """Return (confidences, is_tp_list) for all detections across all images."""
    confs, tps = [], []
    for rec in cached:
        pb, ps, pl = rec["pred_boxes"], rec["pred_scores"], rec["pred_labels"]
        gb, gl = rec["gt_boxes"], rec["gt_labels"]
        # keep only predictions above min_conf
        keep = [(b, s, l) for b, s, l in zip(pb, ps, pl) if s >= min_conf]
        keep.sort(key=lambda x: -x[1])  # descending confidence
        matched_gt = set()
        for b, s, l in keep:
            best_iou, best_j = 0.0, -1
            for j, (gb_j, gl_j) in enumerate(zip(gb, gl)):
                if j in matched_gt:
                    continue
                iou = _iou(b, gb_j)
                if iou > best_iou:
                    best_iou, best_j = iou, j
            is_tp = 1 if (best_iou >= iou_thr and best_j >= 0) else 0
            if is_tp:
                matched_gt.add(best_j)
            confs.append(float(s))
            tps.append(is_tp)
    return confs, tps


# ── Cost sweep helper ─────────────────────────────────────────────────────────
THETA_GRID = sorted({0.001, 0.005, 0.01, 0.02, 0.03, 0.04}
                    | {round(0.05 * k, 2) for k in range(1, 20)})


def cost_sweep(preds, gts, grid=THETA_GRID):
    cst = CostSensitiveThreshold()
    costs = []
    for th in grid:
        c = cst.compute_cost(preds, gts, conf_threshold=th)
        costs.append(c / len(preds))  # per-image
    return np.array(grid), np.array(costs)


# ── Conformal sweep helper ────────────────────────────────────────────────────
LAMBDA_GRID = sorted({0.001, 0.005} | {round(0.01 * k, 2) for k in range(1, 100)})


def conformal_sweep(preds, gts, lam_grid=LAMBDA_GRID, iou_thr=0.5):
    """Empirical miss-rate R_hat(lambda) for each lambda on the calibration set."""
    crc = ConformalRiskController(alpha=0.10, iou_threshold=iou_thr)
    miss_rates = [crc.empirical_risk(preds, gts, lam) for lam in lam_grid]
    return np.array(lam_grid), np.array(miss_rates)


def crc_lambda(lam_grid, miss_rates, n_cal: int, alpha: float = 0.1):
    """Eq.(4): largest lambda where (n*R_hat + 1)/(n+1) <= alpha."""
    lam_hat = None
    for lam, r in zip(lam_grid, miss_rates):
        if (n_cal * r + 1) / (n_cal + 1) <= alpha:
            lam_hat = lam
    return lam_hat


# ── Reliability diagram helper ────────────────────────────────────────────────
def reliability_diagram(confs, tps, n_bins=10):
    """Returns (bin_centers, mean_conf_per_bin, precision_per_bin, counts)."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    centers, mean_conf, prec, cnts = [], [], [], []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        idx = [j for j, c in enumerate(confs) if lo <= c < hi]
        if not idx:
            continue
        cs = [confs[j] for j in idx]
        ts = [tps[j] for j in idx]
        centers.append((lo + hi) / 2)
        mean_conf.append(float(np.mean(cs)))
        prec.append(float(np.mean(ts)))
        cnts.append(len(idx))
    return mean_conf, prec, cnts


# ══════════════════════════════════════════════════════════════════════════════
#  Figure 1 — mAP comparison bar chart
# ══════════════════════════════════════════════════════════════════════════════
def fig1_map_comparison():
    print("[fig1] mAP comparison bar chart …")
    detectors = ["YOLOv8m", "RT-DETR", "Faster\nR-CNN", "RetinaNet"]
    map50     = [0.7686, 0.7721, 0.7153, 0.6176]
    map5095   = [0.5138, 0.4942, 0.4138, 0.3510]

    x = np.arange(len(detectors))
    w = 0.35

    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    b1 = ax.bar(x - w/2, map50,   w, label="mAP@.5",      color=IEEE_BLUE,  zorder=3)
    b2 = ax.bar(x + w/2, map5095, w, label="mAP@.5:.95",  color=IEEE_ORANGE, zorder=3)

    # value labels
    for rect in list(b1) + list(b2):
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2, h + 0.007,
                f"{h:.3f}", ha="center", va="bottom", fontsize=6.5)

    ax.set_xticks(x)
    ax.set_xticklabels(detectors)
    ax.set_ylabel("AP score")
    ax.set_ylim(0, 0.92)
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.05))
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5, zorder=0)
    ax.legend(loc="upper right")
    clean_ax(ax)
    ax.text(0.5, -0.18, "(RetinaNet results pending)",
            transform=ax.transAxes, ha="center", fontsize=6.5, color=GRAY,
            style="italic")

    fig.savefig(OUT / "fig1_map_comparison.pdf")
    plt.close(fig)
    print("  → fig1_map_comparison.pdf")


# ══════════════════════════════════════════════════════════════════════════════
#  Figure 2 — Cost curves E[C(theta)] vs theta  (two panels)
# ══════════════════════════════════════════════════════════════════════════════
def fig2_cost_curves():
    print("[fig2] cost curves (loading prediction JSONs) …")
    yc  = load_cached(YOLO_CALIB);  yp, yg   = to_preds_gts(yc)
    yt  = load_cached(YOLO_TEST);   ytp, ytg  = to_preds_gts(yt)
    fc  = load_cached(FRCNN_CALIB); fp, fg    = to_preds_gts(fc)
    ft  = load_cached(FRCNN_TEST);  ftp, ftg  = to_preds_gts(ft)

    print("  sweeping YOLOv8m …")
    yl_c, yc_costs = cost_sweep(yp,  yg)
    yl_t, yt_costs = cost_sweep(ytp, ytg)
    print("  sweeping Faster R-CNN …")
    fl_c, fc_costs = cost_sweep(fp,  fg)
    fl_t, ft_costs = cost_sweep(ftp, ftg)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=False)

    for ax, (lc, cc, lt, ct, name, theta_star, col) in zip(axes, [
        (yl_c, yc_costs, yl_t, yt_costs, "YOLOv8m",     0.05, IEEE_BLUE),
        (fl_c, fc_costs, fl_t, ft_costs, "Faster R-CNN", 0.50, IEEE_RED),
    ]):
        ax.plot(lc, cc, color=col, linestyle="--", linewidth=1.2,
                label="calib split", alpha=0.7)
        ax.plot(lt, ct, color=col, linewidth=1.8, label="test split")

        # theta* vertical line
        ax.axvline(theta_star, color="k", linestyle=":", linewidth=1.0)
        ymax = max(ct)
        ax.text(theta_star + 0.01, ymax * 0.97,
                r"$\theta^*$" + f"={theta_star}", fontsize=7.5, va="top")

        # cost-reduction annotation for YOLO only
        if name == "YOLOv8m":
            c_half = ct[list(lt).index(0.5)] if 0.5 in lt else ct[-1]
            c_star = ct[np.argmin(ct)]
            ax.annotate(
                f"−31.9%",
                xy=(theta_star, c_star), xytext=(0.25, c_star * 1.15),
                arrowprops=dict(arrowstyle="->", lw=0.8, color=col),
                fontsize=7.5, color=col,
            )

        ax.set_xlabel(r"Confidence threshold $\theta$")
        ax.set_ylabel(r"$\mathbb{E}[\mathcal{C}(\theta)]$ / image")
        ax.set_title(name)
        ax.legend(loc="upper right")
        ax.set_xlim(-0.01, 0.95)
        ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.4, zorder=0)
        clean_ax(ax)

    fig.tight_layout(w_pad=1.2)
    fig.savefig(OUT / "fig2_cost_curves.pdf")
    plt.close(fig)
    print("  → fig2_cost_curves.pdf")


# ══════════════════════════════════════════════════════════════════════════════
#  Figure 3 — Reliability diagram: raw vs isotonic (YOLOv8m)
# ══════════════════════════════════════════════════════════════════════════════
def fig3_reliability():
    print("[fig3] reliability diagram …")
    MIN_CONF = 0.05

    calib = load_cached(YOLO_CALIB)
    test  = load_cached(YOLO_TEST)

    cal_confs, cal_tps = matched_tp_pairs(calib, min_conf=MIN_CONF)
    te_confs,  te_tps  = matched_tp_pairs(test,  min_conf=MIN_CONF)

    # Fit isotonic on calib
    iso = IsotonicCalibrator().fit(cal_confs, cal_tps)
    te_iso = iso.transform(te_confs)

    # Fit temperature on calib
    ts = DetectionTemperatureScaling()
    ts.fit(cal_confs, cal_tps)
    te_temp = ts.transform(te_confs)

    # Reliability bins
    mc_raw,  prec_raw,  cnt_raw  = reliability_diagram(te_confs, te_tps, n_bins=15)
    mc_iso,  prec_iso,  _        = reliability_diagram(te_iso,   te_tps, n_bins=15)
    mc_temp, prec_temp, _        = reliability_diagram(te_temp,  te_tps, n_bins=15)

    fig, ax = plt.subplots(figsize=(3.5, 3.0))

    # Perfect calibration diagonal
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.0, label="Perfect calibration",
            zorder=1)

    ax.plot(mc_raw,  prec_raw,  "o-", color=IEEE_RED,    markersize=4,
            label="Raw", linewidth=1.4, zorder=3)
    ax.plot(mc_temp, prec_temp, "s-", color=IEEE_ORANGE,  markersize=4,
            label="Temperature", linewidth=1.4, zorder=3)
    ax.plot(mc_iso,  prec_iso,  "^-", color=IEEE_GREEN,  markersize=4,
            label="Isotonic", linewidth=1.4, zorder=4)

    ax.set_xlabel("Mean confidence (calibrated)")
    ax.set_ylabel("Fraction of true positives")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(loc="upper left")
    ax.set_title("Reliability Diagram — YOLOv8m (test split)")
    ax.set_aspect("equal")
    clean_ax(ax)

    fig.savefig(OUT / "fig3_reliability.pdf")
    plt.close(fig)
    print("  → fig3_reliability.pdf")


# ══════════════════════════════════════════════════════════════════════════════
#  Figure 4 — ECE ablation grouped bar chart
# ══════════════════════════════════════════════════════════════════════════════
def fig4_ece_ablation():
    print("[fig4] ECE ablation bar chart …")
    methods  = ["Raw", "Temp.\nScaling", "Per-class\nTemp.", "Isotonic\n(ours)"]
    yolo_ece = [0.1015, 0.0979, 0.0978, 0.0109]
    frcnn_ece= [0.1892, 0.1917, 0.1907, 0.0072]

    x = np.arange(len(methods))
    w = 0.35

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    b1 = ax.bar(x - w/2, yolo_ece,  w, label="YOLOv8m",     color=IEEE_BLUE,  zorder=3)
    b2 = ax.bar(x + w/2, frcnn_ece, w, label="Faster R-CNN", color=IEEE_RED,   zorder=3)

    # ECE gate line
    ax.axhline(0.05, color="k", linestyle=":", linewidth=1.0, zorder=5)
    ax.text(3.6, 0.053, "ECE gate\n(<0.05)", fontsize=6.5, va="bottom", ha="right")

    for rect in list(b1) + list(b2):
        h = rect.get_height()
        if h > 0.004:
            ax.text(rect.get_x() + rect.get_width()/2, h + 0.003,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=6.0,
                    rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=7.5)
    ax.set_ylabel("Detection ECE")
    ax.set_ylim(0, 0.26)
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.4, zorder=0)
    clean_ax(ax)

    fig.savefig(OUT / "fig4_ece_ablation.pdf")
    plt.close(fig)
    print("  → fig4_ece_ablation.pdf")


# ══════════════════════════════════════════════════════════════════════════════
#  Figure 5 — Conformal risk curve R_hat(lambda) vs lambda (YOLOv8m)
# ══════════════════════════════════════════════════════════════════════════════
def fig5_conformal():
    print("[fig5] conformal risk curve (loading calib JSONs) …")
    calib = load_cached(YOLO_CALIB)
    cp, cg = to_preds_gts(calib)
    # Scene-level conformal: n = number of GT-bearing images (exchangeable unit),
    # NOT the box count. crc._n_images encapsulates this.
    _crc = ConformalRiskController(alpha=0.10, iou_threshold=0.5)
    n_cal = _crc._n_images(cg)

    print("  sweeping lambda …")
    lam_grid, miss_rates = conformal_sweep(cp, cg)

    alpha = 0.10
    lam_hat = crc_lambda(lam_grid, miss_rates, n_cal, alpha)  # None at alpha=0.10
    # finite-sample-corrected bound: (n*R+1)/(n+1)
    adjusted = (n_cal * miss_rates + 1) / (n_cal + 1)
    floor = float(np.min(adjusted))  # smallest certifiable alpha (miss floor)

    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    ax.plot(lam_grid, miss_rates, color=IEEE_BLUE, linewidth=2.0,
            marker="o", markersize=2.2, markevery=10,
            label=r"Empirical $\hat{R}(\lambda)$", zorder=4)
    ax.plot(lam_grid, adjusted,   color=IEEE_ORANGE, linewidth=1.5,
            linestyle="--",
            label=r"$(n\hat{R}+1)/(n+1)$", zorder=5)

    # alpha=0.10 target line (never reached -> infeasible at scene level)
    ax.axhline(alpha, color="k", linestyle=":", linewidth=1.0, zorder=5)
    ax.text(0.97, alpha + 0.01, r"$\alpha=0.10$ (infeasible)",
            fontsize=7.0, ha="right")

    # miss floor: smallest alpha that could ever be certified
    ax.axhline(floor, color=IEEE_GREEN, linestyle="-.", linewidth=1.2, zorder=4)
    ax.text(0.97, floor + 0.012, rf"miss floor $={floor:.3f}$",
            fontsize=7.0, ha="right", color=IEEE_GREEN)

    ax.set_xlabel(r"Threshold $\lambda$")
    ax.set_ylabel("Per-image miss-rate")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper left")
    ax.set_title("Scene-level Conformal Risk — YOLOv8m (calib)")
    ax.grid(linestyle="--", linewidth=0.4, alpha=0.4, zorder=0)
    clean_ax(ax)

    fig.savefig(OUT / "fig5_conformal.pdf")
    plt.close(fig)
    print(f"  → fig5_conformal.pdf (n_cal_img={n_cal}, miss_floor={floor:.3f}, "
          f"lam_hat@0.10={lam_hat})")


# ══════════════════════════════════════════════════════════════════════════════
#  Figure 6 — RA-AP vs mAP@0.5 scatter (ranking divergence)
# ══════════════════════════════════════════════════════════════════════════════
def fig6_raap_vs_map():
    print("[fig6] RA-AP vs mAP scatter …")
    import csv as _csv
    detectors  = ["YOLOv8m", "RT-DETR", "Faster R-CNN", "RetinaNet"]
    model_keys = ["yolov8m", "rtdetr", "faster_rcnn", "retinanet"]
    # Load computed values from results/raap_variants.csv (global / clean variant)
    # so the figure traces to a regenerated results file (no hardcoded numbers).
    vals = {}
    with open(RESULTS / "raap_variants.csv", newline="") as _f:
        for row in _csv.DictReader(_f):
            if row["variant"] == "global":
                vals[row["model"]] = (float(row["map50"]), float(row["raap"]))
    map50  = [vals[k][0] for k in model_keys]
    raap   = [vals[k][1] for k in model_keys]
    colors = [IEEE_BLUE, IEEE_GREEN, IEEE_RED, IEEE_ORANGE]

    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    for x, y, name, c in zip(map50, raap, detectors, colors):
        ax.scatter(x, y, color=c, zorder=4, s=55)
        offx = 0.003 if name != "RT-DETR" else -0.003
        offy = 0.010 if name not in ("Faster R-CNN",) else -0.018
        ha   = "left" if name != "RT-DETR" else "right"
        ax.text(x + offx, y + offy, name, fontsize=7.5, color=c, ha=ha)

    # Diagonal reference (RA-AP = mAP would mean zero cost penalty)
    xs = np.linspace(0.58, 0.80, 100)
    ax.plot(xs, xs, "k--", linewidth=0.8, alpha=0.4, label="RA-AP = mAP@.5")

    # Rank arrows: show mAP rank vs RA-AP rank swap (FRCNN vs RetinaNet)
    ax.annotate("", xy=(map50[2], raap[2]), xytext=(map50[3], raap[3]),
                arrowprops=dict(arrowstyle="<->", color=GRAY, lw=0.8))
    ax.text((map50[2]+map50[3])/2 + 0.003, (raap[2]+raap[3])/2,
            "rank\npreserved", fontsize=6, color=GRAY, ha="left", va="center")

    ax.set_xlabel("mAP@0.5")
    ax.set_ylabel("RA-AP")
    ax.set_xlim(0.58, 0.82)
    ax.set_ylim(0.15, 0.65)
    ax.legend(loc="upper left", fontsize=7)
    clean_ax(ax)

    fig.savefig(OUT / "fig6_raap_vs_map.pdf")
    plt.close(fig)
    print("  → fig6_raap_vs_map.pdf")


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    fig1_map_comparison()
    fig2_cost_curves()
    fig3_reliability()
    fig4_ece_ablation()
    fig5_conformal()
    fig6_raap_vs_map()
    print(f"\nAll figures saved to {OUT}")
