"""7.5 Cost-ratio sensitivity grid.

Sweeps c_FN ∈ {2, 5, 10, 20, 50} × c_FP ∈ {1, 2, 5} (15 pairs).
For each pair × each detector: optimise θ*, compute E[C] and RA-AP.
Marks the cell where the best-detector identity changes vs. the baseline
(c_FN=10, c_FP=1).

Inputs  (all local, no GPU needed):
  results/{det}_flir_seed0/eval/{det}_test_predictions.json
  results/{det}_flir_seed0/eval/{det}_calib_predictions.json
  results/{det}_flir_seed0/eval/{det}_test_map.json  (for mAP50 baseline)

Outputs:
  results/cost_ratio_grid.csv     — θ*, E[C], RA-AP per (model, c_FN, c_FP)
  results/fig_cost_ratio_grid.pdf — 3-panel heatmap (θ*, E[C], best-detector)

Usage:
    python scripts/16_cost_ratio_grid.py [--root <project_root>]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from itertools import product
from pathlib import Path
from typing import Dict, List, Tuple

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np
from src.risk.cost_sensitive import CostSensitiveThreshold

# ── detector registry ──────────────────────────────────────────────────────────
DETECTORS = {
    "yolov8m":     {"dir": "yolov8m_flir_seed0",    "prefix": "yolov8m"},
    "yolov11m":    {"dir": "yolov11m_flir_seed0",   "prefix": "yolov11m"},
    "rtdetr":      {"dir": "rtdetr_flir_seed0",     "prefix": "rtdetr"},
    "faster_rcnn": {"dir": "faster_rcnn_flir_seed0", "prefix": "frcnn"},
    "retinanet":   {"dir": "retinanet_flir_seed0",  "prefix": "retinanet"},
}

C_FN_GRID = [2, 5, 10, 20, 50]
C_FP_GRID = [1, 2, 5]
THETA_SWEEP = [round(0.05 * i, 2) for i in range(1, 20)]   # 0.05 … 0.95


def load_preds(path: Path) -> List[Dict]:
    """Load predictions JSON → list of dicts with boxes/scores/labels keys."""
    raw = json.loads(path.read_text())
    out = []
    for r in raw:
        # normalise key names (calib_predictions use pred_* prefix)
        out.append({
            "boxes":  r.get("pred_boxes",  r.get("boxes",  [])),
            "scores": r.get("pred_scores", r.get("scores", [])),
            "labels": r.get("pred_labels", r.get("labels", [])),
        })
    return out


def load_gt(path: Path) -> List[Dict]:
    raw = json.loads(path.read_text())
    return [{"boxes": r["gt_boxes"], "labels": r["gt_labels"]} for r in raw]


def load_map50(path: Path) -> float:
    data = json.loads(path.read_text())
    return float(data.get("mAP50", data.get("map50", 0.0)))


def run_grid(repo: Path) -> Tuple[List[Dict], Dict[str, float]]:
    results = []
    map50_baseline: Dict[str, float] = {}

    for det_name, cfg in DETECTORS.items():
        det_dir = repo / "results" / cfg["dir"] / "eval"
        prefix = cfg["prefix"]

        calib_json = det_dir / f"{prefix}_calib_predictions.json"
        test_json  = det_dir / f"{prefix}_test_predictions.json"
        map_json   = det_dir / f"{prefix}_test_map.json"

        if not (calib_json.exists() and test_json.exists()):
            print(f"  [{det_name}] missing eval files — skipping.")
            continue

        calib_preds = load_preds(calib_json)
        calib_gt    = load_gt(calib_json)
        test_preds  = load_preds(test_json)
        test_gt     = load_gt(test_json)
        map50 = load_map50(map_json) if map_json.exists() else 0.0
        map50_baseline[det_name] = map50
        print(f"  [{det_name}] {len(test_preds)} test imgs  mAP50={map50:.4f}")

        for c_fn, c_fp in product(C_FN_GRID, C_FP_GRID):
            cs = CostSensitiveThreshold(c_fn=c_fn, c_fp=c_fp, c_loc=2.0, c_def=0.5)
            theta_star, _ = cs.optimize_threshold(calib_preds, calib_gt,
                                                   threshold_sweep=THETA_SWEEP)
            n = max(len(test_preds), 1)
            cost_test = cs.compute_cost(test_preds, test_gt, theta_star) / n
            cost_05   = cs.compute_cost(test_preds, test_gt, 0.5) / n
            cost_red  = (cost_05 - cost_test) / max(cost_05, 1e-6) * 100.0
            raap = cs.risk_adjusted_ap(map50, cost_test)
            results.append({
                "model":   det_name,
                "c_fn":    c_fn,
                "c_fp":    c_fp,
                "theta_star": round(theta_star, 2),
                "cost_test":  round(cost_test, 2),
                "cost_05":    round(cost_05, 2),
                "cost_red_pct": round(cost_red, 2),
                "raap":    round(raap, 4),
                "map50":   round(map50, 4),
            })

    return results, map50_baseline


def write_csv(rows: List[Dict], out_path: Path) -> None:
    if not rows:
        return
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Saved: {out_path}")


def make_heatmaps(rows: List[Dict], out_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except ImportError:
        print("matplotlib not available — skipping figure generation.")
        return

    models = list(dict.fromkeys(r["model"] for r in rows))
    c_fn_vals = sorted(set(r["c_fn"] for r in rows))
    c_fp_vals = sorted(set(r["c_fp"] for r in rows))

    def _grid(model: str, key: str):
        g = np.zeros((len(c_fn_vals), len(c_fp_vals)))
        lut = {(r["c_fn"], r["c_fp"]): r[key] for r in rows if r["model"] == model}
        for i, fn in enumerate(c_fn_vals):
            for j, fp in enumerate(c_fp_vals):
                g[i, j] = lut.get((fn, fp), np.nan)
        return g

    # Best-detector at each (c_fn, c_fp) by lowest cost
    best_det_grid = np.empty((len(c_fn_vals), len(c_fp_vals)), dtype=object)
    for i, fn in enumerate(c_fn_vals):
        for j, fp in enumerate(c_fp_vals):
            candidates = [(r["model"], r["cost_test"])
                          for r in rows if r["c_fn"] == fn and r["c_fp"] == fp]
            if candidates:
                best_det_grid[i, j] = min(candidates, key=lambda x: x[1])[0]

    fig, axes = plt.subplots(len(models) + 1, 2, figsize=(10, 4 * (len(models) + 1)))
    axes = np.atleast_2d(axes)

    # Per-model: θ* and E[C] heatmaps
    for mi, model in enumerate(models):
        for ci, (key, label) in enumerate([("theta_star", "θ*"), ("cost_test", "E[C]")]):
            ax = axes[mi, ci]
            g = _grid(model, key)
            im = ax.imshow(g, aspect="auto", cmap="viridis")
            ax.set_xticks(range(len(c_fp_vals))); ax.set_xticklabels([f"c_fp={v}" for v in c_fp_vals])
            ax.set_yticks(range(len(c_fn_vals))); ax.set_yticklabels([f"c_fn={v}" for v in c_fn_vals])
            ax.set_title(f"{model} — {label}", fontsize=9)
            plt.colorbar(im, ax=ax, shrink=0.8)
            for i in range(len(c_fn_vals)):
                for j in range(len(c_fp_vals)):
                    ax.text(j, i, f"{g[i,j]:.2f}", ha="center", va="center",
                            fontsize=7, color="white" if g[i,j] > g.mean() else "black")

    # Best-detector map
    ax = axes[len(models), 0]
    model_idx = {m: i for i, m in enumerate(models)}
    num_grid = np.vectorize(lambda x: model_idx.get(x, -1))(best_det_grid)
    cmap = plt.cm.get_cmap("tab10", len(models))
    im = ax.imshow(num_grid.astype(float), aspect="auto", cmap=cmap,
                   vmin=-0.5, vmax=len(models) - 0.5)
    ax.set_xticks(range(len(c_fp_vals))); ax.set_xticklabels([f"c_fp={v}" for v in c_fp_vals])
    ax.set_yticks(range(len(c_fn_vals))); ax.set_yticklabels([f"c_fn={v}" for v in c_fn_vals])
    ax.set_title("Best detector (lowest E[C])", fontsize=9)
    plt.colorbar(im, ax=ax, ticks=range(len(models)),
                 shrink=0.8).set_ticklabels(models)
    for i in range(len(c_fn_vals)):
        for j in range(len(c_fp_vals)):
            ax.text(j, i, best_det_grid[i, j][:4] if best_det_grid[i, j] else "?",
                    ha="center", va="center", fontsize=6)

    # RA-AP comparison (baseline c_fn=10, c_fp=1) vs highest-penalty
    ax = axes[len(models), 1]
    baseline_raap = [r["raap"] for r in rows if r["c_fn"] == 10 and r["c_fp"] == 1]
    highpen_raap  = [r["raap"] for r in rows if r["c_fn"] == 50 and r["c_fp"] == 5]
    mods_b = [r["model"] for r in rows if r["c_fn"] == 10 and r["c_fp"] == 1]
    x = np.arange(len(baseline_raap))
    ax.bar(x - 0.2, baseline_raap, 0.4, label="c_fn=10,c_fp=1")
    ax.bar(x + 0.2, highpen_raap,  0.4, label="c_fn=50,c_fp=5")
    ax.set_xticks(x); ax.set_xticklabels(mods_b, fontsize=7, rotation=20)
    ax.set_ylabel("RA-AP"); ax.set_title("RA-AP: baseline vs high-penalty", fontsize=9)
    ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(str(out_path), bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(REPO))
    args = ap.parse_args()
    repo = Path(args.root)

    print("Running cost-ratio sensitivity grid ...")
    rows, _ = run_grid(repo)
    if not rows:
        sys.exit("No results — check that eval prediction JSONs exist.")

    out_csv = repo / "results/cost_ratio_grid.csv"
    write_csv(rows, out_csv)

    out_fig = repo / "results/fig_cost_ratio_grid.pdf"
    make_heatmaps(rows, out_fig)
    print(f"\nCost-ratio grid complete. {len(rows)} rows written.")


if __name__ == "__main__":
    main()
