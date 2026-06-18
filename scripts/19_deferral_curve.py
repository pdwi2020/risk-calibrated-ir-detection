"""7.6 Uncertainty-Gated Deferral Risk Curve.

Defers the most-uncertain X% of detections (by entropy of calibrated
confidence) to human review.  Plots deferral-rate vs missed-detection cost
vs E[C] under clean AND corrupted conditions, showing:
  - A small deferral budget sharply cuts high-cost misses.
  - The budget needed grows with corruption severity.

Extends Sbeyti UAI 2024 (RGB, no corruption) to thermal IR + corruption
interaction — explicitly cited as the RGB precedent.

Inputs:
  results/{det}_flir_seed0/eval/{prefix}_test_predictions.json   (clean)
  results/corruption_preds/{model}_{type}_{sev}.json             (corrupted)
  results/{det}_flir_seed0/eval/{prefix}_calib_predictions.json  (for θ* + IR)

Outputs:
  results/deferral_curve.csv      — cost metrics at each deferral rate × condition
  results/fig_deferral.pdf        — E[C] vs deferral rate, clean + 3 corruption types

Usage:
    python scripts/19_deferral_curve.py [--root <project_root>]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np
from sklearn.isotonic import IsotonicRegression
from src.risk.cost_sensitive import CostSensitiveThreshold
from src.corruption.corruption_pipeline import CORRUPTION_REGISTRY

DETECTORS = {
    "yolov8m":     "yolov8m",
    "yolov11m":    "yolov11m",
    "rtdetr":      "rtdetr",
    "faster_rcnn": "frcnn",
    "retinanet":   "retinanet",
}
THETA_SWEEP     = [round(0.05 * i, 2) for i in range(1, 20)]
DEFERRAL_RATES  = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
PLOT_CORRUPTS   = ["gaussian_noise", "motion_blur", "fog"]   # 3 for figure
PLOT_SEV        = 4


def load_preds(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    raw = json.loads(path.read_text())
    return [{"boxes":  r.get("pred_boxes",  r.get("boxes",  [])),
             "scores": r.get("pred_scores", r.get("scores", [])),
             "labels": r.get("pred_labels", r.get("labels", []))} for r in raw]


def load_gt(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    return [{"boxes": r["gt_boxes"], "labels": r["gt_labels"]}
            for r in json.loads(path.read_text())]


def add_uncertainty(preds: List[Dict], ir: IsotonicRegression = None) -> List[Dict]:
    """Add binary-entropy uncertainty to each prediction.
    Uses calibrated confidence if IR provided, else raw confidence.
    """
    out = []
    for p in preds:
        scores = np.array(p["scores"], dtype=np.float32)
        if ir is not None and len(scores):
            scores = np.clip(ir.predict(scores).astype(np.float32), 1e-6, 1 - 1e-6)
        unc = -(scores * np.log(scores + 1e-9) +
                (1 - scores) * np.log(1 - scores + 1e-9))   # binary entropy
        out.append({"boxes": p["boxes"], "scores": p["scores"],
                    "labels": p["labels"],
                    "uncertainty": unc.tolist()})
    return out


def run_deferral_sweep(
    preds: List[Dict],
    gt: List[Dict],
    theta_star: float,
    deferral_rates: List[float],
) -> List[Dict]:
    """For each deferral rate, compute E[C] with uncertainty-gated deferral."""
    cs = CostSensitiveThreshold(c_fn=10, c_fp=1)
    rows = []
    for dr in deferral_rates:
        if dr == 0.0:
            cost = cs.compute_cost(preds, gt, conf_threshold=theta_star)
            rows.append({"deferral_rate": dr, "cost": round(cost, 2),
                         "unc_threshold": None})
            continue
        # Collect all uncertainty scores to find the threshold at deferral_rate
        all_unc = []
        for p in preds:
            conf_arr = np.array(p["scores"], dtype=np.float32)
            keep = conf_arr >= theta_star
            if p.get("uncertainty"):
                unc_arr = np.array(p["uncertainty"], dtype=np.float32)
                all_unc.extend(unc_arr[keep].tolist())
        if not all_unc:
            rows.append({"deferral_rate": dr, "cost": float("nan"),
                         "unc_threshold": None})
            continue
        # Set unc_threshold so that dr fraction of kept detections are deferred
        unc_thresh = float(np.quantile(all_unc, 1.0 - dr))
        cost = cs.compute_cost(preds, gt, conf_threshold=theta_star,
                               uncertainty_threshold=unc_thresh)
        rows.append({"deferral_rate": dr, "cost": round(cost, 2),
                     "unc_threshold": round(unc_thresh, 4)})
    return rows


def run_analysis(repo: Path) -> List[Dict]:
    preds_dir = repo / "results/corruption_preds"
    rows = []

    for model_name, prefix in DETECTORS.items():
        eval_dir = repo / f"results/{model_name}_flir_seed0/eval"
        calib_path = eval_dir / f"{prefix}_calib_predictions.json"
        test_path  = eval_dir / f"{prefix}_test_predictions.json"

        calib_preds_raw = load_preds(calib_path)
        test_preds_raw  = load_preds(test_path)
        calib_gt  = load_gt(calib_path)
        test_gt   = load_gt(test_path)

        if not calib_preds_raw:
            print(f"  [{model_name}] missing eval files — skipping.")
            continue

        # Fit isotonic calibrator on calib split
        from src.calibration.cach import match_detections_to_gt
        all_s, all_tp = [], []
        for rec_path in [calib_path]:
            recs = json.loads(rec_path.read_text()) if rec_path.exists() else []
            for r in recs:
                s, tp = match_detections_to_gt(
                    r["pred_boxes"], r["pred_scores"], r["pred_labels"],
                    r["gt_boxes"], r["gt_labels"])
                all_s.append(s); all_tp.append(tp)
        if all_s:
            ir = IsotonicRegression(out_of_bounds="clip").fit(
                np.concatenate(all_s), np.concatenate(all_tp))
        else:
            ir = None

        # Optimal θ*
        cs = CostSensitiveThreshold()
        theta_star, _ = cs.optimize_threshold(calib_preds_raw, calib_gt, THETA_SWEEP)

        # ── Clean deferral curve ───────────────────────────────────────────
        test_preds_unc = add_uncertainty(test_preds_raw, ir)
        sweep = run_deferral_sweep(test_preds_unc, test_gt, theta_star, DEFERRAL_RATES)
        for s in sweep:
            rows.append({"model": model_name, "condition": "clean", "severity": 0,
                         **s})

        # ── Corrupted deferral curves ──────────────────────────────────────
        if not preds_dir.exists():
            continue

        for cname in PLOT_CORRUPTS:
            rec_path = preds_dir / f"{model_name}_{cname}_{PLOT_SEV}.json"
            if not rec_path.exists():
                continue
            recs = json.loads(rec_path.read_text())
            c_preds = [{"boxes": r["pred_boxes"], "scores": r["pred_scores"],
                        "labels": r["pred_labels"]} for r in recs]
            c_gt    = [{"boxes": r["gt_boxes"], "labels": r["gt_labels"]}
                       for r in recs]
            c_preds_unc = add_uncertainty(c_preds, ir)
            sweep = run_deferral_sweep(c_preds_unc, c_gt, theta_star, DEFERRAL_RATES)
            for s in sweep:
                rows.append({"model": model_name, "condition": cname,
                             "severity": PLOT_SEV, **s})

        print(f"  [{model_name}] done  θ*={theta_star:.2f}")

    return rows


def make_figure(rows: List[Dict], out_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping figure.")
        return

    models = list(dict.fromkeys(r["model"] for r in rows))
    conditions = ["clean"] + PLOT_CORRUPTS
    colors = {"clean": "black", "gaussian_noise": "red",
              "motion_blur": "blue", "fog": "green"}

    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4), squeeze=False)
    for mi, model in enumerate(models):
        ax = axes[0][mi]
        for cond in conditions:
            crows = [r for r in rows if r["model"] == model and r["condition"] == cond]
            if not crows:
                continue
            dr = [r["deferral_rate"] for r in crows]
            cost = [r["cost"] for r in crows]
            label = cond.replace("_", " ")
            ax.plot(dr, cost, "o-", color=colors.get(cond, "gray"),
                    label=label, lw=1.5, markersize=4)
        ax.set_xlabel("Deferral rate")
        ax.set_ylabel("E[C]")
        ax.set_title(model, fontsize=9)
        ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(str(out_path), bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(REPO))
    args = ap.parse_args()
    repo = Path(args.root)

    print("Running deferral risk curve analysis ...")
    rows = run_analysis(repo)
    if not rows:
        sys.exit("No data.")

    out_csv = repo / "results/deferral_curve.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"Saved: {out_csv}  ({len(rows)} rows)")
    make_figure(rows, repo / "results/fig_deferral.pdf")


if __name__ == "__main__":
    main()
