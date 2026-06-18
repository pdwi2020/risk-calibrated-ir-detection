"""7.3 Corruption-Aware and Class-Wise Risk-Calibrated Thresholds.

Extends the global cost-sensitive threshold θ* to:
  (a) per-class θ* (person / bike / car)
  (b) per-corruption-type θ*

Shows that a single global θ* is suboptimal under corruption shift; condition-
al θ* further reduces E[C].  Explicitly frames this as the IR+corruption-
conditional extension of Sbeyti UAI 2024 + Elkan 2001.

Inputs:
  results/corruption_preds/{model}_{type}_{sev}.json   (from 14a)
  results/{model}_flir_seed0/eval/{prefix}_test_predictions.json  (clean test)
  results/{model}_flir_seed0/eval/{prefix}_calib_predictions.json (clean calib)

Outputs:
  results/conditional_thresholds.csv  — per (model, condition, class) θ* and E[C]
  results/fig_conditional_thresholds.pdf — bar chart: global vs class-wise vs corruption θ*

Usage:
    python scripts/17_conditional_thresholds.py [--root <project_root>]
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
from src.risk.cost_sensitive import CostSensitiveThreshold
from src.corruption.corruption_pipeline import CORRUPTION_REGISTRY

DETECTORS = {
    "yolov8m":     "yolov8m",
    "yolov11m":    "yolov11m",
    "rtdetr":      "rtdetr",
    "faster_rcnn": "frcnn",
    "retinanet":   "retinanet",
}
CLASS_NAMES = {0: "person", 1: "bike", 2: "car"}
THETA_SWEEP = [round(0.05 * i, 2) for i in range(1, 20)]
C_FN, C_FP = 10.0, 1.0


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
    return [{"boxes":  r["gt_boxes"], "labels": r["gt_labels"]}
            for r in json.loads(path.read_text())]


def filter_class(preds: List[Dict], gt: List[Dict], cls: int
                 ) -> Tuple[List[Dict], List[Dict]]:
    """Keep only detections / GT of a single class."""
    fp = [{"boxes":  [b for b, l in zip(p["boxes"], p["labels"]) if l == cls],
           "scores": [s for s, l in zip(p["scores"], p["labels"]) if l == cls],
           "labels": [l for l in p["labels"] if l == cls]} for p in preds]
    fg = [{"boxes":  [b for b, l in zip(g["boxes"], g["labels"]) if l == cls],
           "labels": [l for l in g["labels"] if l == cls]} for g in gt]
    return fp, fg


def run_analysis(repo: Path) -> List[Dict]:
    preds_dir = repo / "results/corruption_preds"
    rows = []

    for model_name, prefix in DETECTORS.items():
        eval_dir = repo / f"results/{model_name}_flir_seed0/eval"
        cs = CostSensitiveThreshold(c_fn=C_FN, c_fp=C_FP)

        # ── Clean global baseline ──────────────────────────────────────────
        calib_preds = load_preds(eval_dir / f"{prefix}_calib_predictions.json")
        test_preds  = load_preds(eval_dir / f"{prefix}_test_predictions.json")
        calib_gt    = load_gt(eval_dir / f"{prefix}_calib_predictions.json")
        test_gt     = load_gt(eval_dir / f"{prefix}_test_predictions.json")
        if not calib_preds:
            print(f"  [{model_name}] missing eval files — skipping.")
            continue

        theta_global, _ = cs.optimize_threshold(calib_preds, calib_gt, THETA_SWEEP)
        cost_global  = cs.compute_cost(test_preds, test_gt, theta_global)
        cost_05      = cs.compute_cost(test_preds, test_gt, 0.5)
        rows.append({"model": model_name, "condition": "clean", "class": "all",
                     "theta_star": round(theta_global, 2),
                     "cost_test":  round(cost_global, 2),
                     "cost_05":    round(cost_05, 2),
                     "cost_red_vs_global_pct": 0.0})
        print(f"  [{model_name}] global θ*={theta_global:.2f}  E[C]={cost_global:.2f}")

        # ── Clean per-class ────────────────────────────────────────────────
        for cls, cname in CLASS_NAMES.items():
            cp, cg = filter_class(calib_preds, calib_gt, cls)
            tp, tg = filter_class(test_preds, test_gt, cls)
            if sum(len(p["scores"]) for p in cp) < 5:
                continue
            theta_c, _ = cs.optimize_threshold(cp, cg, THETA_SWEEP)
            cost_c = cs.compute_cost(tp, tg, theta_c)
            red = (cost_global - cost_c) / max(cost_global, 1e-6) * 100.0
            rows.append({"model": model_name, "condition": "clean", "class": cname,
                         "theta_star": round(theta_c, 2),
                         "cost_test":  round(cost_c, 2),
                         "cost_05":    round(cost_05, 2),
                         "cost_red_vs_global_pct": round(red, 2)})

        # ── Corruption-aware θ* (averaged over severities per type) ────────
        if not preds_dir.exists():
            print(f"  corruption_preds dir not found — skipping corruption-aware θ*")
            continue

        for cname in CORRUPTION_REGISTRY:
            # Pool across severities for calib; evaluate per severity
            pool_calib_preds, pool_calib_gt = [], []
            for sev in [1, 2, 3, 4]:
                rec_path = preds_dir / f"{model_name}_{cname}_{sev}.json"
                if rec_path.exists():
                    recs = json.loads(rec_path.read_text())
                    pool_calib_preds += [{"boxes": r["pred_boxes"], "scores": r["pred_scores"],
                                          "labels": r["pred_labels"]} for r in recs]
                    pool_calib_gt    += [{"boxes": r["gt_boxes"], "labels": r["gt_labels"]}
                                         for r in recs]
            if not pool_calib_preds:
                continue
            theta_corr, _ = cs.optimize_threshold(pool_calib_preds, pool_calib_gt,
                                                   THETA_SWEEP)
            # Evaluate on sev=4 (hardest)
            rec_sev4 = preds_dir / f"{model_name}_{cname}_4.json"
            if not rec_sev4.exists():
                continue
            recs4 = json.loads(rec_sev4.read_text())
            test_c_preds = [{"boxes": r["pred_boxes"], "scores": r["pred_scores"],
                              "labels": r["pred_labels"]} for r in recs4]
            test_c_gt    = [{"boxes": r["gt_boxes"], "labels": r["gt_labels"]}
                             for r in recs4]
            cost_global_on_corr = cs.compute_cost(test_c_preds, test_c_gt, theta_global)
            cost_corr_opt       = cs.compute_cost(test_c_preds, test_c_gt, theta_corr)
            red = (cost_global_on_corr - cost_corr_opt) / max(cost_global_on_corr, 1e-6) * 100.0
            rows.append({
                "model": model_name, "condition": cname, "class": "all",
                "theta_star": round(theta_corr, 2),
                "cost_test":  round(cost_corr_opt, 2),
                "cost_05":    round(cs.compute_cost(test_c_preds, test_c_gt, 0.5), 2),
                "cost_red_vs_global_pct": round(red, 2),
            })
            print(f"    corr-aware [{cname}] θ*={theta_corr:.2f}  "
                  f"E[C]={cost_corr_opt:.2f}  red={red:.1f}%")

    return rows


def make_figure(rows: List[Dict], out_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    models = list(dict.fromkeys(r["model"] for r in rows))
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4), squeeze=False)
    for mi, model in enumerate(models):
        ax = axes[0][mi]
        model_rows = [r for r in rows if r["model"] == model]
        labels_plot = [f"{r['condition']}\n({r['class']})" for r in model_rows]
        reds = [r["cost_red_vs_global_pct"] for r in model_rows]
        colors = ["tab:blue" if r["condition"] == "clean" else
                  ("tab:orange" if r["class"] != "all" else "tab:green")
                  for r in model_rows]
        ax.barh(range(len(labels_plot)), reds, color=colors, height=0.7)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_yticks(range(len(labels_plot))); ax.set_yticklabels(labels_plot, fontsize=6)
        ax.set_xlabel("Cost reduction vs global θ* (%)")
        ax.set_title(model, fontsize=9)

    plt.tight_layout()
    plt.savefig(str(out_path), bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(REPO))
    args = ap.parse_args()
    repo = Path(args.root)

    print("Running conditional threshold analysis ...")
    rows = run_analysis(repo)
    if not rows:
        sys.exit("No data.")

    out_csv = repo / "results/conditional_thresholds.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"Saved: {out_csv}  ({len(rows)} rows)")
    make_figure(rows, repo / "results/fig_conditional_thresholds.pdf")


if __name__ == "__main__":
    main()
