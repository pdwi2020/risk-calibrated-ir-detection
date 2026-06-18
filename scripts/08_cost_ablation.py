"""Cost-ratio ablation: c_FN x c_FP grid, all 4 detectors.
Run from project root: python /tmp/mv_paper_scripts/08_cost_ablation.py
"""
from __future__ import annotations
import csv, json, sys
from pathlib import Path

REPO = Path("/Volumes/Crucial X9/Research Projects/MV_Paper")
sys.path.insert(0, str(REPO))
from src.risk.cost_sensitive import CostSensitiveThreshold

C_LOC, C_DEF, IOU_THR = 2.0, 0.5, 0.5
C_FN_GRID = [5, 10, 20]
C_FP_GRID = [1, 2, 5]
THETA_SWEEP = [0.001, 0.005, 0.01, 0.02, 0.03, 0.04] + [round(0.05*k,3) for k in range(1,20)]

DETECTORS = {
    "YOLOv8m":     ("results/yolov8m_flir_seed0/eval/yolov8m_calib_predictions.json",
                    "results/yolov8m_flir_seed0/eval/yolov8m_test_predictions.json"),
    "RT-DETR":     ("results/rtdetr_flir_seed0/eval/rtdetr_calib_predictions.json",
                    "results/rtdetr_flir_seed0/eval/rtdetr_test_predictions.json"),
    "Faster R-CNN":("results/faster_rcnn_flir_seed0/eval/frcnn_calib_predictions.json",
                    "results/faster_rcnn_flir_seed0/eval/frcnn_test_predictions.json"),
    "RetinaNet":   ("results/retinanet_flir_seed0/eval/retinanet_calib_predictions.json",
                    "results/retinanet_flir_seed0/eval/retinanet_test_predictions.json"),
}

def load_preds(path):
    records = json.loads(Path(path).read_text())
    preds, gts = [], []
    for d in records:
        preds.append({"boxes":d.get("pred_boxes",[]),"scores":d.get("pred_scores",[]),"labels":d.get("pred_labels",[])})
        gts.append({"boxes":d.get("gt_boxes",[]),"labels":d.get("gt_labels",[])})
    return preds, gts

def main():
    rows = []
    for det, (cp, tp) in DETECTORS.items():
        print(f"\n=== {det} ===")
        cal_p, cal_g = load_preds(str(REPO/cp))
        tst_p, tst_g = load_preds(str(REPO/tp))
        for c_fn in C_FN_GRID:
            for c_fp in C_FP_GRID:
                cost = CostSensitiveThreshold(c_fn=c_fn,c_fp=c_fp,c_loc=C_LOC,c_def=C_DEF,iou_threshold=IOU_THR)
                th, _ = cost.optimize_threshold(cal_p, cal_g, THETA_SWEEP)
                c_star = cost.compute_cost(tst_p, tst_g, th)
                c_half = cost.compute_cost(tst_p, tst_g, 0.5)
                red = 100.0*(c_half-c_star)/c_half if c_half else 0.0
                print(f"  c_FN={c_fn:2d} c_FP={c_fp}  theta*={th:.3f}  red={red:+.1f}%")
                rows.append({"detector":det,"c_fn":c_fn,"c_fp":c_fp,"ratio":c_fn/c_fp,
                             "theta_star":th,"c_test_at_half":round(c_half,1),
                             "c_test_at_star":round(c_star,1),"cost_reduction_pct":round(red,2)})
    out = REPO/"results/cost_ablation.csv"
    with open(out,"w",newline="") as f:
        w = csv.DictWriter(f,fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nCSV: {out}")
    _latex(rows)

def _latex(rows):
    lut = {(r["detector"],r["c_fn"],r["c_fp"]):r for r in rows}
    dets = ["YOLOv8m","RT-DETR","Faster R-CNN","RetinaNet"]
    lines = [
        r"\begin{table}[t]",
        r"\caption{Cost-ratio ablation. Each cell: $\theta^*$ (cost red.\,\%). Paper baseline ($c_\text{FN}{=}10$, $c_\text{FP}{=}1$) \textbf{bold}. $c_\text{Loc}{=}2$, $c_\text{Def}{=}0.5$ fixed.}",
        r"\label{tab:cost_ablation}\centering\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{l"+"c"*9+"}",r"\toprule",
        r"& \multicolumn{3}{c}{$c_\text{FN}=5$} & \multicolumn{3}{c}{$c_\text{FN}=10$} & \multicolumn{3}{c}{$c_\text{FN}=20$}\\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}",
        r"Detector & $c_P\!=\!1$ & $c_P\!=\!2$ & $c_P\!=\!5$ & $c_P\!=\!1$ & $c_P\!=\!2$ & $c_P\!=\!5$ & $c_P\!=\!1$ & $c_P\!=\!2$ & $c_P\!=\!5$\\",
        r"\midrule",
    ]
    for det in dets:
        cells=[]
        for c_fn in [5,10,20]:
            for c_fp in [1,2,5]:
                r=lut[(det,c_fn,c_fp)]
                c=f"{r['theta_star']:.2f} ({r['cost_reduction_pct']:+.0f}\\%)"
                if c_fn==10 and c_fp==1: c=r"\textbf{"+c+"}"
                cells.append(c)
        lines.append(det+" & "+" & ".join(cells)+r"\\")
    lines+=[r"\bottomrule",r"\end{tabular}",r"\end{table}"]
    out = REPO/"results/cost_ablation_latex.tex"
    out.write_text("\n".join(lines)+"\n")
    print(f"LaTeX: {out}")

if __name__=="__main__": main()
