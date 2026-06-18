"""Bootstrap 95% CIs — fast version using precomputed per-image costs.

Instead of re-running full cost evaluation 10,000 times, precompute the
per-image cost components once, then resample indices to get bootstrap
distribution. This is O(N_boot * N_images) instead of O(N_boot * N_images * N_preds).

~10-20s total runtime.
"""
from __future__ import annotations
import csv, json, random, sys
from pathlib import Path

REPO = Path("/Volumes/Crucial X9/Research Projects/MV_Paper")
sys.path.insert(0, str(REPO))

N_BOOT = 10000
C_FN, C_FP, C_LOC, C_DEF, IOU_THR = 10.0, 1.0, 2.0, 0.5, 0.5
THETA_SWEEP = [0.001, 0.005, 0.01, 0.02, 0.03, 0.04] + [round(0.05*k,3) for k in range(1,20)]

DETECTORS = {
    "YOLOv8m":      ("results/yolov8m_flir_seed0/eval/yolov8m_calib_predictions.json",
                     "results/yolov8m_flir_seed0/eval/yolov8m_test_predictions.json", 0.05),
    "RT-DETR":      ("results/rtdetr_flir_seed0/eval/rtdetr_calib_predictions.json",
                     "results/rtdetr_flir_seed0/eval/rtdetr_test_predictions.json", 0.35),
    "Faster R-CNN": ("results/faster_rcnn_flir_seed0/eval/frcnn_calib_predictions.json",
                     "results/faster_rcnn_flir_seed0/eval/frcnn_test_predictions.json", 0.50),
    "RetinaNet":    ("results/retinanet_flir_seed0/eval/retinanet_calib_predictions.json",
                     "results/retinanet_flir_seed0/eval/retinanet_test_predictions.json", 0.35),
}

# ── Pure-Python IoU + per-image cost (no src/ imports needed) ──────────────
def _iou(a, b):
    ax1,ay1,ax2,ay2 = float(a[0]),float(a[1]),float(a[2]),float(a[3])
    bx1,by1,bx2,by2 = float(b[0]),float(b[1]),float(b[2]),float(b[3])
    iw = max(0., min(ax2,bx2)-max(ax1,bx1))
    ih = max(0., min(ay2,by2)-max(ay1,by1))
    inter = iw*ih
    ua = max(0.,ax2-ax1)*max(0.,ay2-ay1)
    ub = max(0.,bx2-bx1)*max(0.,by2-by1)
    union = ua+ub-inter
    return inter/union if union>0 else 0.

def _per_image_cost(pred, gt, theta, c_fn, c_fp, c_loc, c_def, iou_thr):
    boxes  = pred.get("boxes",[])
    scores = [float(s) for s in pred.get("scores",[])]
    labels = [int(l) for l in pred.get("labels",[])]
    gb     = gt.get("boxes",[])
    gl     = [int(l) for l in gt.get("labels",[])]
    # filter by threshold
    keep = [i for i,s in enumerate(scores) if s >= theta]
    boxes  = [boxes[i] for i in keep]
    scores = [scores[i] for i in keep]
    labels = [labels[i] for i in keep]
    matched_g = [False]*len(gb)
    matched_p = [False]*len(boxes)
    tp_ious = []
    for pi in sorted(range(len(boxes)), key=lambda i:-scores[i]):
        best_iou, best_g = 0., -1
        for gi in range(len(gb)):
            if matched_g[gi] or gl[gi]!=labels[pi]: continue
            iou = _iou(boxes[pi], gb[gi])
            if iou >= iou_thr and iou > best_iou:
                best_iou, best_g = iou, gi
        if best_g >= 0:
            matched_g[best_g] = True
            matched_p[pi] = True
            tp_ious.append(best_iou)
    fn = sum(1 for m in matched_g if not m)
    fp = sum(1 for m in matched_p if not m)
    loc = sum(1.-iou for iou in tp_ious)
    return c_fn*fn + c_fp*fp + c_loc*loc + c_def*0.

def load_records(path):
    return json.loads(Path(path).read_text())

def main():
    rng = random.Random(42)
    results = []
    print(f"Bootstrap CI (n_boot={N_BOOT}, 95% CI, fast precompute)\n")
    header = f"{'Detector':<14} {'theta*':>6} {'Point':>8} {'95% CI':>20}  {'std':>6}"
    print(header); print("-"*60)

    for det, (cp, tp, theta_star) in DETECTORS.items():
        cal_recs = load_records(str(REPO/cp))
        tst_recs = load_records(str(REPO/tp))

        # Precompute per-image costs at theta_star and theta=0.5 for test split
        costs_star = [_per_image_cost(
            {"boxes":r.get("pred_boxes",[]),"scores":r.get("pred_scores",[]),"labels":r.get("pred_labels",[])},
            {"boxes":r.get("gt_boxes",[]),"labels":r.get("gt_labels",[])},
            theta_star, C_FN, C_FP, C_LOC, C_DEF, IOU_THR) for r in tst_recs]
        costs_half = [_per_image_cost(
            {"boxes":r.get("pred_boxes",[]),"scores":r.get("pred_scores",[]),"labels":r.get("pred_labels",[])},
            {"boxes":r.get("gt_boxes",[]),"labels":r.get("gt_labels",[])},
            0.5, C_FN, C_FP, C_LOC, C_DEF, IOU_THR) for r in tst_recs]

        n = len(costs_star)
        point_star = sum(costs_star)
        point_half = sum(costs_half)
        point_red = 100.*(point_half - point_star)/point_half if point_half else 0.

        # Bootstrap by resampling per-image cost vectors
        boot_reds = []
        for _ in range(N_BOOT):
            idx = [rng.randint(0, n-1) for _ in range(n)]
            bs = sum(costs_star[i] for i in idx)
            bh = sum(costs_half[i] for i in idx)
            boot_reds.append(100.*(bh-bs)/bh if bh else 0.)
        boot_reds.sort()
        lo = boot_reds[int(N_BOOT*0.025)]
        hi = boot_reds[int(N_BOOT*0.975)]
        std = (sum((x - point_red)**2 for x in boot_reds)/N_BOOT)**0.5

        gate = "✓" if lo > 0 else "✗"
        print(f"{det:<14} {theta_star:>6.3f} {point_red:>+7.1f}%  [{lo:>+6.1f}%, {hi:>+5.1f}%]  {std:>5.2f}%  {gate}")
        results.append({"detector":det,"theta_star":theta_star,
                        "point_pct":round(point_red,2),"ci_lo":round(lo,2),
                        "ci_hi":round(hi,2),"boot_std":round(std,2)})

    out_csv = REPO/"results/bootstrap_ci.csv"
    with open(out_csv,"w",newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader(); w.writerows(results)
    print(f"\nCSV: {out_csv}")

    # LaTeX
    lines = [
        r"\begin{table}[t]",
        r"\caption{Bootstrap 95\,\% CIs on cost reduction for each detector"
        r" (seed 0, test split $n{=}572$, $n_\text{boot}{=}10{,}000$)."
        r" $\theta^*$ fixed from calibration split. \checkmark\ = CI entirely above~0.}",
        r"\label{tab:bootstrap_ci}\centering",
        r"\begin{tabular}{lrrrr}", r"\toprule",
        r"Detector & $\theta^*$ & Point est. & 95\,\% CI & Std \\", r"\midrule",
    ]
    for r in results:
        gate = r"\checkmark" if r["ci_lo"] > 0 else r"\texttimes"
        lines.append(
            f"{r['detector']} & {r['theta_star']:.3f} & {r['point_pct']:+.1f}\\% & "
            f"[{r['ci_lo']:+.1f}\\%,\\,{r['ci_hi']:+.1f}\\%] & {r['boot_std']:.2f}\\% & {gate} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    out_tex = REPO/"results/bootstrap_ci_latex.tex"
    out_tex.write_text("\n".join(lines)+"\n")
    print(f"LaTeX: {out_tex}")

if __name__=="__main__": main()
