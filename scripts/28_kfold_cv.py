#!/usr/bin/env python3
"""P11 — K-fold cross-validation of the headline calibration / risk results.

The paper reports single-split (seed-42, 572/572) numbers. This script pools the
calibration + test prediction caches (1144 images per detector) and runs stratified
5-fold CV, refitting each estimator on the training folds and evaluating on the
held-out fold, to show the headline metrics are not artefacts of one split.

Metrics per fold (all leakage-free: fit on train folds, evaluate on held-out fold):
  * Isotonic ECE        — detection ECE after PAV isotonic calibration
  * Cost reduction (%)  — (C(0.5) - C(theta*)) / C(0.5), theta* fit on train
  * Conformal miss floor— smallest certifiable per-image miss rate (test)

Reuses the paper's own methodology modules (no re-implementation):
  src/calibration/cach.py:match_detections_to_gt, src/risk/cost_sensitive.py,
  src/risk/conformal.py.

Output: results/kfold_cv.csv  (per detector: mean +/- std over 5 folds)
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression

import sys
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from src.calibration.cach import match_detections_to_gt          # noqa: E402
from src.risk.cost_sensitive import CostSensitiveThreshold        # noqa: E402
from src.risk.conformal import ConformalRiskController            # noqa: E402

LOCAL = REPO / "results"
# Optional second location for the prediction caches (e.g. an external drive).
EXTRA = Path(os.environ["MV_RESULTS_DIR"]) if os.environ.get("MV_RESULTS_DIR") else None
DET_FILES = {
    "YOLOv8m":      ("yolov8m_flir_seed0/eval/yolov8m_calib_predictions.json",
                     "yolov8m_flir_seed0/eval/yolov8m_test_predictions.json"),
    "RT-DETR":      ("rtdetr_flir_seed0/eval/rtdetr_calib_predictions.json",
                     "rtdetr_flir_seed0/eval/rtdetr_test_predictions.json"),
    "Faster R-CNN": ("faster_rcnn_flir_seed0/eval/frcnn_calib_predictions.json",
                     "faster_rcnn_flir_seed0/eval/frcnn_test_predictions.json"),
    "RetinaNet":    ("retinanet_flir_seed0/eval/retinanet_calib_predictions.json",
                     "retinanet_flir_seed0/eval/retinanet_test_predictions.json"),
}
K = 5
IOU, N_BINS, SEED = 0.5, 10, 42
THETA_GRID = [0.001, 0.005, 0.01, 0.02, 0.03, 0.04] + [round(0.05 * k, 2) for k in range(1, 20)]
LAMBDA_GRID = [0.001, 0.005] + [round(0.01 * k, 2) for k in range(1, 100)]


def _resolve(rel: str) -> Path:
    for base in (LOCAL, EXTRA):
        if base is None:
            continue
        p = base / rel
        if p.exists():
            return p
    raise FileNotFoundError(rel)


def load_records(det: str) -> list[dict]:
    cal, test = DET_FILES[det]
    return json.loads(_resolve(cal).read_text()) + json.loads(_resolve(test).read_text())


def to_pred_gt(rec: dict):
    return ({"boxes": rec["pred_boxes"], "scores": rec["pred_scores"], "labels": rec["pred_labels"]},
            {"boxes": rec["gt_boxes"], "labels": rec["gt_labels"]})


def ece(scores: np.ndarray, is_tp: np.ndarray, n_bins: int = N_BINS) -> float:
    if len(scores) == 0:
        return float("nan")
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (scores > lo) & (scores <= hi)
        if m.any():
            e += abs(is_tp[m].mean() - scores[m].mean()) * m.mean()
    return float(e)


def dets_of(records: list[dict]):
    s, tp = [], []
    for r in records:
        ss, tt = match_detections_to_gt(r["pred_boxes"], r["pred_scores"], r["pred_labels"],
                                        r["gt_boxes"], r["gt_labels"], IOU)
        s.append(ss); tp.append(tt)
    return (np.concatenate(s) if s else np.array([])), (np.concatenate(tp) if tp else np.array([]))


def strat_folds(records: list[dict], k: int, seed: int):
    """Stratify by GT-presence so every fold has comparable miss-event support."""
    rng = np.random.default_rng(seed)
    has_gt = np.array([len(r["gt_boxes"]) > 0 for r in records])
    folds = [[] for _ in range(k)]
    for group in (np.where(has_gt)[0], np.where(~has_gt)[0]):
        idx = group.copy(); rng.shuffle(idx)
        for j, i in enumerate(idx):
            folds[j % k].append(int(i))
    return folds


def run_detector(det: str) -> dict:
    records = load_records(det)
    folds = strat_folds(records, K, SEED)
    cost = CostSensitiveThreshold(c_fn=10.0, c_fp=1.0, c_loc=2.0, c_def=0.5, iou_threshold=IOU)
    per = {"ece_iso": [], "cost_red": [], "miss_floor": []}

    for f in range(K):
        te = folds[f]
        tr = [i for j in range(K) if j != f for i in folds[j]]
        tr_recs = [records[i] for i in tr]
        te_recs = [records[i] for i in te]

        # --- isotonic ECE (fit on train dets, evaluate on test dets) ---
        s_tr, tp_tr = dets_of(tr_recs)
        s_te, tp_te = dets_of(te_recs)
        ir = IsotonicRegression(out_of_bounds="clip").fit(s_tr, tp_tr)
        per["ece_iso"].append(ece(ir.predict(s_te), tp_te))

        # --- cost reduction (theta* on train, evaluate on test) ---
        tr_pred, tr_gt = zip(*(to_pred_gt(r) for r in tr_recs))
        te_pred, te_gt = zip(*(to_pred_gt(r) for r in te_recs))
        theta, _ = cost.optimize_threshold(list(tr_pred), list(tr_gt), THETA_GRID)
        c_base = cost.compute_cost(list(te_pred), list(te_gt), 0.5)
        c_opt = cost.compute_cost(list(te_pred), list(te_gt), theta)
        per["cost_red"].append(100.0 * (c_base - c_opt) / c_base if c_base > 0 else 0.0)

        # --- conformal miss floor (fit on train, evaluate on test) ---
        crc = ConformalRiskController(alpha=0.10, iou_threshold=IOU)
        risk_by_lam = {l: crc.empirical_risk(list(tr_pred), list(tr_gt), l) for l in LAMBDA_GRID}
        lam_floor = min(LAMBDA_GRID, key=lambda l: risk_by_lam[l])
        per["miss_floor"].append(crc.empirical_risk(list(te_pred), list(te_gt), lam_floor))

    out = {"detector": det}
    for k_, v in per.items():
        a = np.array(v, dtype=float)
        out[f"{k_}_mean"] = float(a.mean())
        out[f"{k_}_std"] = float(a.std(ddof=1))
    return out


def main() -> None:
    rows = []
    for det in DET_FILES:
        print(f"[{det}] running {K}-fold CV ...", flush=True)
        r = run_detector(det)
        rows.append(r)
        print(f"    isotonic ECE {r['ece_iso_mean']:.4f}±{r['ece_iso_std']:.4f}  "
              f"cost-red {r['cost_red_mean']:.1f}±{r['cost_red_std']:.1f}%  "
              f"miss-floor {r['miss_floor_mean']:.3f}±{r['miss_floor_std']:.3f}")
    out = LOCAL / "kfold_cv.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
