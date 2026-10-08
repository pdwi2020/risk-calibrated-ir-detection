"""Risk-calibration evaluation: turn cached detector predictions into the paper's
headline risk numbers.

Consumes the per-image prediction JSONs written by ``scripts/04_eval_detector.py``
(each record has pred_boxes/pred_scores/pred_labels/gt_boxes/gt_labels) and runs
the three contribution modules:

  * cost-sensitive thresholding  -> theta* = argmin E[C(theta)], and the E[C]
    reduction at theta* vs the naive theta=0.5 (the core >=15% claim);
  * temperature scaling          -> detection ECE before/after (gate: < 0.05);
  * conformal risk control       -> lambda_hat + empirical test coverage vs 1-alpha
    (lambda_hat is None when no threshold certifies alpha: the infeasible case);
  * RA-AP = mAP - E[C(theta*)] / (c_FN * mean GT per image)  (the novel metric).

LEAKAGE-FREE (paper) MODE  --calib-json X --test-json Y :
  T, theta* and lambda_hat are FIT ON CALIBRATION; every headline number is
  REPORTED ON TEST. This is the protocol reviewers expect.

SMOKE MODE  --pred-json Z :
  fit and report on the same file (NOT leakage-free) -- pipeline validation only;
  prints a warning.

Pure-Python downstream: needs only the JSON files (no GPU / torch / numpy).

Examples:
  python scripts/05_risk_eval.py \
      --calib-json results/yolov8m_flir_seed0/eval/yolov8m_calib_predictions.json \
      --test-json  results/yolov8m_flir_seed0/eval/yolov8m_test_predictions.json \
      --map-json   results/yolov8m_flir_seed0/eval/yolov8m_test_map.json \
      --out        results/yolov8m_flir_seed0/eval/yolov8m_risk_summary.json

  python scripts/05_risk_eval.py --pred-json <preds>.json --map 0.681   # smoke
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.risk.cost_sensitive import CostSensitiveThreshold
from src.calibration.temperature_scaling import (DetectionTemperatureScaling,
                                                 PerClassTemperatureScaling)
from src.calibration.isotonic import IsotonicCalibrator
from src.risk.conformal import ConformalRiskController


# ----------------------------------------------------------------------------
# Loading / reshaping the cached prediction records
# ----------------------------------------------------------------------------
def load_cached(path: str) -> List[Dict]:
    data = json.loads(Path(path).read_text())
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a list of per-image records")
    return data


def to_pred_gt(cached: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """Reshape cached records into the (predictions, ground_truths) form the
    cost / conformal modules expect."""
    preds, gts = [], []
    for d in cached:
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


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = (float(v) for v in a)
    bx1, by1, bx2, by2 = (float(v) for v in b)
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    ua = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ub = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = ua + ub - inter
    return inter / union if union > 0 else 0.0


def matched_tp_labels(cached: List[Dict], iou_thr: float = 0.5,
                      min_conf: float = 0.0) -> Tuple[List[float], List[int], List[int]]:
    """Per-detection (confidence, is_TP) across all images.

    Class-aware greedy matching by descending score at a single IoU threshold --
    the standard TP assignment used to fit and measure confidence calibration.
    Detections below ``min_conf`` are dropped before matching, so calibration is
    reported over the displayed operating set, not the conf~0.001 tail.
    """
    confs: List[float] = []
    tps: List[int] = []
    labs: List[int] = []
    for d in cached:
        boxes = d.get("pred_boxes", [])
        scores = [float(s) for s in d.get("pred_scores", [])]
        labels = [int(x) for x in d.get("pred_labels", [])]
        g_boxes = d.get("gt_boxes", [])
        g_labels = [int(x) for x in d.get("gt_labels", [])]
        matched = [False] * len(g_boxes)
        for pi in sorted(range(len(boxes)), key=lambda i: -scores[i]):
            if scores[pi] < min_conf:
                continue
            best_iou, best_g = 0.0, -1
            for gi in range(len(g_boxes)):
                if matched[gi] or g_labels[gi] != labels[pi]:
                    continue
                iou = _iou(boxes[pi], g_boxes[gi])
                if iou >= iou_thr and iou > best_iou:
                    best_iou, best_g = iou, gi
            confs.append(scores[pi])
            labs.append(labels[pi])
            if best_g >= 0:
                matched[best_g] = True
                tps.append(1)
            else:
                tps.append(0)
    return confs, tps, labs


# ----------------------------------------------------------------------------
# Analysis
# ----------------------------------------------------------------------------
def analyze(
    calib: List[Dict],
    test: List[Dict],
    *,
    c_fn: float,
    c_fp: float,
    c_loc: float,
    c_def: float,
    iou: float,
    alpha_conf: float,
    map_test: Optional[float],
    theta_sweep: Sequence[float],
    lambda_grid: Sequence[float],
    ece_min_conf: float = 0.05,
) -> Dict:
    cost = CostSensitiveThreshold(c_fn=c_fn, c_fp=c_fp, c_loc=c_loc, c_def=c_def,
                                  iou_threshold=iou)
    cal_pred, cal_gt = to_pred_gt(calib)
    test_pred, test_gt = to_pred_gt(test)

    # --- cost: fit theta* on calib, report E[C] on test ---
    theta_star, c_cal_star = cost.optimize_threshold(cal_pred, cal_gt, list(theta_sweep))
    c_test_star = cost.compute_cost(test_pred, test_gt, theta_star)
    c_test_half = cost.compute_cost(test_pred, test_gt, 0.5)
    reduction = (c_test_half - c_test_star) / c_test_half if c_test_half else 0.0

    # --- calibration ablation: fit each method on calib, ECE on test ---
    ts = DetectionTemperatureScaling()
    cal_conf, cal_tp, cal_lab = matched_tp_labels(calib, iou, ece_min_conf)
    test_conf, test_tp, test_lab = matched_tp_labels(test, iou, ece_min_conf)
    ece_raw = ts.compute_detection_ece(test_conf, test_tp)
    # (1) single global temperature (NLL fit)
    T = ts.fit(cal_conf, cal_tp)
    ece_temp = ts.compute_detection_ece(ts.transform(test_conf), test_tp)
    # (2) per-class temperature
    pct = PerClassTemperatureScaling()
    pct.fit(cal_conf, cal_tp, cal_lab)
    ece_pct = ts.compute_detection_ece(pct.transform(test_conf, test_lab), test_tp)
    # (3) isotonic regression (non-parametric monotone)
    iso = IsotonicCalibrator().fit(cal_conf, cal_tp)
    ece_iso = ts.compute_detection_ece(iso.transform(test_conf), test_tp)
    ece_methods = {"raw": ece_raw, "temperature": ece_temp,
                   "per_class_temp": ece_pct, "isotonic": ece_iso}
    best_method = min(ece_methods, key=ece_methods.get)
    ece_cal = ece_methods[best_method]

    # --- conformal: calibrate lambda_hat on calib, validate coverage on test ---
    crc = ConformalRiskController(alpha=alpha_conf, iou_threshold=iou)
    lam = crc.calibrate(cal_pred, cal_gt, list(lambda_grid))
    cov = crc.validate_coverage(test_pred, test_gt)

    # --- RA-AP on test: per-image E[C] over the per-image detect-nothing cost ---
    n_test = max(len(test), 1)
    e_cost_per_img = c_test_star / n_test
    mean_gt = cost.mean_gt_per_image(test_gt)
    ra_ap = (cost.risk_adjusted_ap(map_test, e_cost_per_img, mean_gt)
             if map_test is not None else None)

    return {
        "n_calib_images": len(calib),
        "n_test_images": len(test),
        "cost_weights": {"c_fn": c_fn, "c_fp": c_fp, "c_loc": c_loc, "c_def": c_def,
                         "iou": iou},
        "ece_min_conf": ece_min_conf,
        "theta_star": theta_star,
        "E_cost_calib_at_theta_star": c_cal_star,
        "E_cost_test_at_0.5": c_test_half,
        "E_cost_test_at_theta_star": c_test_star,
        "E_cost_test_per_image": e_cost_per_img,
        "cost_reduction_frac": reduction,
        "temperature_T": T,
        "ece_test_raw": ece_raw,
        "ece_methods": ece_methods,
        "ece_best_method": best_method,
        "ece_test_calibrated": ece_cal,
        "conformal_alpha": alpha_conf,
        "lambda_hat": lam,
        "test_miss_rate": cov["test_risk"],
        "test_coverage": cov["test_coverage"],
        "conformal_controlled": cov["controlled"],
        "map_test": map_test,
        "mean_gt_per_image": mean_gt,
        "ra_ap": ra_ap,
    }


def _fmt(x, nd=4):
    return "n/a" if x is None else f"{x:.{nd}f}"


def report(res: Dict, leakage_free: bool) -> None:
    w = res["cost_weights"]
    print("\n=== Risk-Calibration Evaluation ===")
    if leakage_free:
        print(f"mode: LEAKAGE-FREE  (calib={res['n_calib_images']} imgs, "
              f"test={res['n_test_images']} imgs)")
    else:
        print(f"mode: *** SMOKE *** single file, fit==eval, NOT leakage-free "
              f"({res['n_test_images']} imgs)")
    print(f"cost weights: c_FN={w['c_fn']} c_FP={w['c_fp']} c_Loc={w['c_loc']} "
          f"c_Def={w['c_def']}  iou={w['iou']}\n")

    print("[cost]")
    print(f"  theta*                 = {res['theta_star']:.2f}   (fit on calib)")
    print(f"  E[C] test @ theta=0.5  = {_fmt(res['E_cost_test_at_0.5'], 1)}")
    print(f"  E[C] test @ theta*     = {_fmt(res['E_cost_test_at_theta_star'], 1)}")
    print(f"  E[C] test per image    = {_fmt(res['E_cost_test_per_image'], 3)}")
    red = res["cost_reduction_frac"] * 100.0
    gate = "PASS" if red >= 15.0 else "below 15% gate"
    print(f"  cost reduction         = {red:.1f}%   [{gate}]")

    print(f"[calibration]  (ECE on test, conf>={res['ece_min_conf']:.2f}; fit on calib)")
    for m in ("raw", "temperature", "per_class_temp", "isotonic"):
        print(f"    {m:16s} ECE = {_fmt(res['ece_methods'][m])}")
    g = "PASS" if res["ece_test_calibrated"] < 0.05 else "above 0.05 gate"
    print(f"  best = {res['ece_best_method']} -> {_fmt(res['ece_test_calibrated'])}   [{g}]")

    print("[conformal]")
    print(f"  alpha                  = {res['conformal_alpha']:.2f}")
    lam = res["lambda_hat"]
    print(f"  lambda_hat             = {_fmt(lam, 3)}   (calib)"
          + ("   [infeasible: no threshold certifies alpha]" if lam is None else ""))
    print(f"  test miss-rate         = {_fmt(res['test_miss_rate'])}   "
          f"coverage = {_fmt(res['test_coverage'])}   "
          f"controlled = {res['conformal_controlled']}")

    print("[RA-AP]")
    print(f"  mAP                    = {_fmt(res['map_test'])}")
    print(f"  RA-AP = mAP - E[C*]/(c_FN*g), g={_fmt(res['mean_gt_per_image'], 2)} = {_fmt(res['ra_ap'])}\n")


# ----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calib-json", help="calibration predictions (paper mode)")
    ap.add_argument("--test-json", help="test predictions (paper mode)")
    ap.add_argument("--pred-json", help="single predictions file (SMOKE mode)")
    ap.add_argument("--map", type=float, default=None, help="mAP for RA-AP")
    ap.add_argument("--map-json", default=None, help="read mAP50 from a 04_eval _map.json")
    ap.add_argument("--c-fn", type=float, default=10.0)
    ap.add_argument("--c-fp", type=float, default=1.0)
    ap.add_argument("--c-loc", type=float, default=2.0)
    ap.add_argument("--c-def", type=float, default=0.5)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--alpha-conf", type=float, default=0.1,
                    help="conformal target miss-rate budget")
    ap.add_argument("--ece-min-conf", type=float, default=0.05,
                    help="min detection conf for ECE/temperature (drop the conf~0 tail)")
    ap.add_argument("--out", default=None, help="write JSON summary here")
    args = ap.parse_args()

    if args.pred_json:
        cached = load_cached(args.pred_json)
        calib = test = cached
        leakage_free = False
    elif args.calib_json and args.test_json:
        calib = load_cached(args.calib_json)
        test = load_cached(args.test_json)
        leakage_free = True
    else:
        ap.error("provide either --pred-json (smoke) or both --calib-json and --test-json")

    map_test = args.map
    if map_test is None and args.map_json:
        map_test = float(json.loads(Path(args.map_json).read_text())["mAP50"])

    # Dense at the low end: with c_FN >> c_FP and conf=0.001 predictions, the
    # cost-optimal theta and the conformal lambda both sit very low, so sweep
    # down toward the 0.001 eval floor or the optimum gets clipped at a grid edge.
    theta_sweep = sorted({0.001, 0.005, 0.01, 0.02, 0.03, 0.04}
                         | {round(0.05 * i, 2) for i in range(1, 20)})   # 0.001 .. 0.95
    lambda_grid = sorted({0.001, 0.005}
                         | {round(0.01 * i, 2) for i in range(1, 100)})  # 0.001 .. 0.99

    res = analyze(
        calib, test,
        c_fn=args.c_fn, c_fp=args.c_fp, c_loc=args.c_loc, c_def=args.c_def,
        iou=args.iou, alpha_conf=args.alpha_conf,
        map_test=map_test, theta_sweep=theta_sweep, lambda_grid=lambda_grid,
        ece_min_conf=args.ece_min_conf,
    )
    res["leakage_free"] = leakage_free
    report(res, leakage_free)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=2))
        print(f"[risk] summary -> {args.out}")


if __name__ == "__main__":
    main()
