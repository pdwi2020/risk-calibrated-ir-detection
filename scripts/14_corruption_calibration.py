"""7.1 Corruption-Aware Calibration Analysis (★★★ — strongest novel finding).

For each detector × corruption type × severity, computes ECE under four protocols:
  (a) no-cal     — raw detector confidence
  (b) clean-cal  — isotonic regressor fit on clean calibration split (our current method)
  (c) pooled-cal — isotonic regressor fit on pooled clean + all-corruption predictions
  (d) per-type-cal — isotonic regressor fit on same-type (oracle: knows the corruption)

Demonstrates that clean-cal silently fails under thermal IR corruption, while
pooled-cal recovers most of the ECE gap without corruption labels.

Inputs (from 14a_corruption_infer.py GPU cache):
  results/corruption_preds/{model}_clean.json      — clean predictions
  results/corruption_preds/{model}_{type}_{sev}.json  — per-condition predictions
  results/{model}_flir_seed0/eval/{prefix}_calib_predictions.json  — calib split preds

Outputs:
  results/corruption_calibration.csv  — ECE for 4 models × 6 types × 4 sev × 4 protocols
  results/fig_corruption_calibration.pdf  — reliability diagram grid (4 corruptions @ sev=4)

Usage:
    python scripts/14_corruption_calibration.py [--root <project_root>]
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

from src.corruption.corruption_pipeline import CORRUPTION_REGISTRY

DETECTORS = {
    "yolov8m":     "yolov8m",
    "rtdetr":      "rtdetr",
    "faster_rcnn": "frcnn",
    "retinanet":   "retinanet",
}

IOU_THRESH = 0.5
N_BINS = 15


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_preds_json(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text())


def cache_path(preds_dir: Path, model: str, split: str,
               cname=None, sev: int = 0) -> Path:
    """Per-split corruption cache path, falling back to the legacy (test-only)
    un-split name so this still runs against pre-P3 caches if needed."""
    if cname is None:
        p = preds_dir / f"{model}_{split}_clean.json"
        return p if p.exists() else preds_dir / f"{model}_clean.json"
    p = preds_dir / f"{model}_{split}_{cname}_{sev}.json"
    return p if p.exists() else preds_dir / f"{model}_{cname}_{sev}.json"


def extract_scores_labels(records: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    """Return (scores, is_TP) arrays using IoU matching."""
    from src.calibration.cach import match_detections_to_gt
    all_s, all_tp = [], []
    for rec in records:
        s, tp = match_detections_to_gt(
            rec["pred_boxes"], rec["pred_scores"], rec["pred_labels"],
            rec["gt_boxes"], rec["gt_labels"], IOU_THRESH)
        all_s.append(s); all_tp.append(tp)
    if not all_s:
        return np.array([]), np.array([])
    return np.concatenate(all_s), np.concatenate(all_tp)


def ece_score(scores: np.ndarray, is_tp: np.ndarray, n_bins: int = N_BINS) -> float:
    if len(scores) == 0:
        return float("nan")
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (scores > lo) & (scores <= hi)
        if mask.sum() == 0:
            continue
        ece += abs(is_tp[mask].mean() - scores[mask].mean()) * mask.mean()
    return float(ece)


def fit_isotonic(scores: np.ndarray, is_tp: np.ndarray) -> IsotonicRegression:
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(scores, is_tp)
    return ir


def reliability_bins(scores: np.ndarray, is_tp: np.ndarray, n_bins: int = 10):
    """Return (bin_centers, mean_conf, fraction_tp, counts) for reliability diagram."""
    bins = np.linspace(0, 1, n_bins + 1)
    centers, conf_means, frac_tp, counts = [], [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (scores > lo) & (scores <= hi)
        cnt = mask.sum()
        counts.append(cnt)
        centers.append((lo + hi) / 2)
        if cnt == 0:
            conf_means.append(np.nan); frac_tp.append(np.nan)
        else:
            conf_means.append(scores[mask].mean())
            frac_tp.append(is_tp[mask].mean())
    return (np.array(centers), np.array(conf_means),
            np.array(frac_tp), np.array(counts))


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_analysis(repo: Path) -> List[Dict]:
    preds_dir = repo / "results/corruption_preds"
    if not preds_dir.exists():
        sys.exit(f"ERROR: {preds_dir} not found. Run 14a_corruption_infer.py first.")

    rows = []
    # Store calibrators and arrays for figure
    fig_data = {}

    for model_name, prefix in DETECTORS.items():
        eval_dir   = repo / f"results/{model_name}_flir_seed0/eval"
        calib_json = eval_dir / f"{prefix}_calib_predictions.json"
        # Gate on the held-out TEST clean cache (eval is reported on TEST).
        test_clean = cache_path(preds_dir, model_name, "test")

        if not test_clean.exists():
            print(f"  [{model_name}] test clean preds missing — skipping.")
            continue

        # ── clean-cal isotonic: fit on the CLEAN calibration split ─────────
        if calib_json.exists():
            c_recs = load_preds_json(calib_json)
            c_scores, c_tp = extract_scores_labels(c_recs)
            ir_clean = fit_isotonic(c_scores, c_tp) if len(c_scores) else None
        else:
            ir_clean = None
            print(f"  [{model_name}] calib predictions not found; clean-cal unavailable.")

        # ── pooled isotonic: FIT on CALIBRATION (clean + all corruptions),
        #    leakage-free w.r.t. the test conditions it is scored on ─────────
        pool_s, pool_tp = [], []
        for cname in CORRUPTION_REGISTRY:
            for sev in [1, 2, 3, 4]:
                rec = load_preds_json(cache_path(preds_dir, model_name, "calibration", cname, sev))
                if rec:
                    s, tp = extract_scores_labels(rec)
                    pool_s.append(s); pool_tp.append(tp)
        cal_clean = load_preds_json(cache_path(preds_dir, model_name, "calibration"))
        if cal_clean:
            s0, tp0 = extract_scores_labels(cal_clean)
            pool_s.append(s0); pool_tp.append(tp0)
        if pool_s:
            ps = np.concatenate(pool_s); pp = np.concatenate(pool_tp)
            if len(ps) == 0:
                print(f"  [{model_name}] pooled cal: 0 detections — skipping model.")
                continue
            ir_pooled = fit_isotonic(ps, pp)
        else:
            ir_pooled = None
            ps = np.array([])

        print(f"  [{model_name}] pooled cal (calibration fit): {len(ps):,} detections")

        for cname in CORRUPTION_REGISTRY:
            # Per-type isotonic (oracle): FIT on CALIBRATION corrupted of this type.
            type_s, type_tp = [], []
            for sev in [1, 2, 3, 4]:
                rec = load_preds_json(cache_path(preds_dir, model_name, "calibration", cname, sev))
                if rec:
                    s, tp = extract_scores_labels(rec)
                    type_s.append(s); type_tp.append(tp)
            if type_s:
                ts = np.concatenate(type_s); ttp = np.concatenate(type_tp)
                ir_pertype = fit_isotonic(ts, ttp)
            else:
                ir_pertype = None

            for sev in [1, 2, 3, 4]:
                # EVALUATE every protocol on the held-out TEST corrupted condition.
                rec = load_preds_json(cache_path(preds_dir, model_name, "test", cname, sev))
                if not rec:
                    continue
                scores, is_tp = extract_scores_labels(rec)
                if len(scores) == 0:
                    continue

                ece_nocal  = ece_score(scores, is_tp)
                ece_clean  = ece_score(ir_clean.predict(scores), is_tp)  if ir_clean  else float("nan")
                ece_pooled = ece_score(ir_pooled.predict(scores), is_tp) if ir_pooled else float("nan")
                ece_oracle = ece_score(ir_pertype.predict(scores), is_tp) if ir_pertype else float("nan")

                rows.append({
                    "model": model_name, "corruption": cname, "severity": sev,
                    "ece_nocal":  round(ece_nocal,  4),
                    "ece_clean":  round(ece_clean,  4),
                    "ece_pooled": round(ece_pooled, 4),
                    "ece_oracle": round(ece_oracle, 4),
                    "n_dets": len(scores),
                })

                # store sev=4 data for reliability figures
                if sev == 4 and cname in ["gaussian_noise", "motion_blur",
                                           "fog", "resolution"]:
                    fig_data[(model_name, cname)] = {
                        "scores":       scores,
                        "is_tp":        is_tp,
                        "cal_clean":    ir_clean.predict(scores)  if ir_clean  else scores,
                        "cal_pooled":   ir_pooled.predict(scores) if ir_pooled else scores,
                    }

    return rows, fig_data


def make_figure(fig_data: dict, out_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping figure.")
        return

    models = list(dict.fromkeys(k[0] for k in fig_data))
    corrs  = ["gaussian_noise", "motion_blur", "fog", "resolution"]
    n_corr = len(corrs)
    n_mods = len(models)
    fig, axes = plt.subplots(n_mods, n_corr, figsize=(3.5 * n_corr, 3.5 * n_mods),
                              squeeze=False)

    for mi, model in enumerate(models):
        for ci, cname in enumerate(corrs):
            ax = axes[mi][ci]
            key = (model, cname)
            if key not in fig_data:
                ax.axis("off"); continue
            d = fig_data[key]
            _, _, frac_raw,    _ = reliability_bins(d["scores"],     d["is_tp"])
            _, _, frac_clean,  _ = reliability_bins(d["cal_clean"],  d["is_tp"])
            _, _, frac_pooled, _ = reliability_bins(d["cal_pooled"], d["is_tp"])
            x = np.linspace(0.05, 0.95, len(frac_raw))
            ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="perfect")
            ax.plot(x, np.nan_to_num(frac_raw),    "r-",  lw=1.2, label="no-cal")
            ax.plot(x, np.nan_to_num(frac_clean),  "b--", lw=1.2, label="clean-cal")
            ax.plot(x, np.nan_to_num(frac_pooled), "g-",  lw=1.2, label="pooled-cal")
            ax.set_title(f"{model}\n{cname.replace('_',' ')} sev=4", fontsize=7)
            ax.set_xlabel("Confidence", fontsize=7)
            ax.set_ylabel("Fraction TP", fontsize=7)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.tick_params(labelsize=6)
            if mi == 0 and ci == 0:
                ax.legend(fontsize=6)

    plt.tight_layout()
    plt.savefig(str(out_path), bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(REPO))
    args = ap.parse_args()
    repo = Path(args.root)

    print("Running corruption calibration analysis ...")
    rows, fig_data = run_analysis(repo)
    if not rows:
        sys.exit("No data — run 14a_corruption_infer.py first.")

    out_csv = repo / "results/corruption_calibration.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"Saved: {out_csv}  ({len(rows)} rows)")

    out_fig = repo / "results/fig_corruption_calibration.pdf"
    make_figure(fig_data, out_fig)

    # Print summary: average ECE across models and severities
    import pandas as pd
    try:
        df = pd.DataFrame(rows)
        summary = df.groupby("corruption")[["ece_nocal","ece_clean","ece_pooled","ece_oracle"]].mean()
        print("\nMean ECE by corruption (averaged over models and severities):")
        print(summary.to_string())
        gain = df["ece_clean"].mean() - df["ece_pooled"].mean()
        print(f"\nPooled-cal ECE reduction vs clean-cal: {gain:.4f} ({gain/df['ece_clean'].mean()*100:.1f}%)")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
