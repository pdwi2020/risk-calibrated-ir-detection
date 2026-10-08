#!/usr/bin/env python3
"""Recompute the conformal risk-control table at the IMAGE level (P2).

The first draft controlled a BOX-level miss rate with n = total ground-truth
boxes (~5.7k), which is not the exchangeable unit the per-scene safety budget is
about and silently inflated the effective sample size. This script recomputes
lambda_hat / test miss rate / coverage / feasibility for every detector using
the corrected ``ConformalRiskController`` (per-image binary miss event,
n = #GT-bearing images, infeasible budgets reported as a FAILED gate).

It reads only the cached calibration/test prediction JSONs (no GPU/torch) and
writes:
  * results/conformal_image_level.csv   -- one row per detector (Table V source)
  * results/conformal_image_level.json  -- full payload incl. grid + params

Usage:
  python scripts/26_conformal_image_level.py \
      --root /path/to/project \
      --alpha 0.10 --iou 0.5
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.risk.conformal import ConformalRiskController  # noqa: E402


# detector -> (calibration preds, test preds) relative to <root>/results
DETECTORS = {
    "YOLOv8m": (
        "yolov8m_flir_seed0/eval/yolov8m_calib_predictions.json",
        "yolov8m_flir_seed0/eval/yolov8m_test_predictions.json",
    ),
    "RT-DETR": (
        "rtdetr_flir_seed0/eval/rtdetr_calib_predictions.json",
        "rtdetr_flir_seed0/eval/rtdetr_test_predictions.json",
    ),
    "Faster R-CNN": (
        "faster_rcnn_flir_seed0/eval/faster_rcnn_calibration_predictions.json",
        "faster_rcnn_flir_seed0/eval/faster_rcnn_test_predictions.json",
    ),
    "RetinaNet": (
        "retinanet_flir_seed0/eval/retinanet_calib_predictions.json",
        "retinanet_flir_seed0/eval/retinanet_test_predictions.json",
    ),
}


def reshape(records: List[Dict]):
    """Cached record -> (preds, gts) in the controller's {boxes,scores,labels} form."""
    preds, gts = [], []
    for d in records:
        preds.append({
            "boxes": d.get("pred_boxes", []),
            "scores": d.get("pred_scores", []),
            "labels": d.get("pred_labels", []),
        })
        gts.append({
            "boxes": d.get("gt_boxes", []),
            "labels": d.get("gt_labels", []),
        })
    return preds, gts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="MV_Paper root (contains results/)")
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--iou", type=float, default=0.5)
    args = ap.parse_args()

    results_dir = Path(args.root) / "results"
    # Grid Lambda = {0.001, 0.005} U {0.01 k}_{k=1..99}  (matches the paper).
    grid = [0.001, 0.005] + [round(0.01 * k, 2) for k in range(1, 100)]

    rows = []
    for name, (cal_rel, test_rel) in DETECTORS.items():
        cal_path = results_dir / cal_rel
        test_path = results_dir / test_rel
        if not cal_path.exists() or not test_path.exists():
            print(f"[skip] {name}: missing {cal_path if not cal_path.exists() else test_path}")
            continue
        cal_recs = json.loads(cal_path.read_text())
        test_recs = json.loads(test_path.read_text())
        cal_pred, cal_gt = reshape(cal_recs)
        test_pred, test_gt = reshape(test_recs)

        crc = ConformalRiskController(alpha=args.alpha, iou_threshold=args.iou)
        lam = crc.calibrate(cal_pred, cal_gt, grid)
        cov = crc.validate_coverage(test_pred, test_gt)

        n_cal_imgs = crc._n_images(cal_gt)
        n_test_imgs = crc._n_images(test_gt)
        cal_risk = crc.empirical_risk(cal_pred, cal_gt, lam) if lam is not None else None

        # --- achievable per-image miss-rate FRONTIER ------------------------
        # risk is monotone non-decreasing in lambda, so the loosest threshold
        # minimises it; the finite-sample-corrected value is the smallest alpha
        # for which the gate can ever be feasible (the "miss floor").
        cal_risk_by_lam = {l: crc.empirical_risk(cal_pred, cal_gt, l) for l in grid}
        lam_floor = min(grid, key=lambda l: cal_risk_by_lam[l])
        floor_corrected = (n_cal_imgs * cal_risk_by_lam[lam_floor] + 1.0) / (n_cal_imgs + 1.0)
        floor_test = crc.empirical_risk(test_pred, test_gt, lam_floor)

        # feasibility / operating point across a sweep of alpha targets
        sweep = {}
        for a in (0.10, 0.20, 0.30, 0.40, 0.50):
            c2 = ConformalRiskController(alpha=a, iou_threshold=args.iou)
            la = c2.calibrate(cal_pred, cal_gt, grid)
            cv = c2.validate_coverage(test_pred, test_gt)
            sweep[f"{a:.2f}"] = {
                "feasible": cv["feasible"],
                "lambda_hat": cv["lambda_hat"],
                "test_miss_rate": round(float(cv["test_risk"]), 4),
                "test_coverage": round(float(cv["test_coverage"]), 4),
                "controlled": cv["controlled"],
            }

        row = {
            "detector": name,
            "alpha": args.alpha,
            "iou": args.iou,
            "n_cal_records": len(cal_recs),
            "n_cal_gt_images": n_cal_imgs,
            "n_test_records": len(test_recs),
            "n_test_gt_images": n_test_imgs,
            "feasible": cov["feasible"],
            "lambda_hat": cov["lambda_hat"],
            "cal_risk_at_lambda": round(cal_risk, 4) if cal_risk is not None else None,
            "test_miss_rate": round(float(cov["test_risk"]), 4),
            "test_coverage": round(float(cov["test_coverage"]), 4),
            "controlled": cov["controlled"],
            "best_achievable_lambda": cov.get("best_achievable_lambda"),
            "miss_floor_lambda": lam_floor,
            "miss_floor_cal_corrected": round(floor_corrected, 4),
            "miss_floor_test": round(floor_test, 4),
            "alpha_sweep": sweep,
        }
        rows.append(row)
        print(f"{name:13s} miss_floor(alpha>=){row['miss_floor_cal_corrected']:.3f} "
              f"(lam={lam_floor}, test_miss={floor_test:.3f})  "
              f"@alpha={args.alpha}: feasible={row['feasible']}")

    # Outputs
    out_csv = results_dir / "conformal_image_level.csv"
    out_sweep = results_dir / "conformal_alpha_sweep.csv"
    out_json = results_dir / "conformal_image_level.json"
    fields = ["detector", "alpha", "iou", "n_cal_records", "n_cal_gt_images",
              "n_test_records", "n_test_gt_images", "feasible", "lambda_hat",
              "cal_risk_at_lambda", "test_miss_rate", "test_coverage",
              "controlled", "best_achievable_lambda",
              "miss_floor_lambda", "miss_floor_cal_corrected", "miss_floor_test"]
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    # Long-form alpha sweep: one row per (detector, alpha target).
    with out_sweep.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["detector", "alpha", "feasible", "lambda_hat",
                    "test_miss_rate", "test_coverage", "controlled"])
        for r in rows:
            for a, s in r["alpha_sweep"].items():
                w.writerow([r["detector"], a, s["feasible"], s["lambda_hat"],
                            s["test_miss_rate"], s["test_coverage"], s["controlled"]])

    out_json.write_text(json.dumps(
        {"params": {"alpha": args.alpha, "iou": args.iou, "grid": grid,
                    "risk": "per-image binary missed-detection event",
                    "n_unit": "GT-bearing image"},
         "detectors": rows}, indent=2))
    print(f"\nwrote {out_csv}\nwrote {out_sweep}\nwrote {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
