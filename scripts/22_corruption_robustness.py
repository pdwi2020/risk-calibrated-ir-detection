"""22_corruption_robustness.py — Unbiased corruption-robustness metrics.

Recomputes corruption robustness from the seeded TEST prediction caches
(results/corruption_preds/{model}_test_{corr}_{sev}.json) — no detector re-run.

Why this replaces the previous metric
-------------------------------------
The old robustness table used a *self-relative* mean Corruption Error
    mCE_c = E_c / E_clean ,   E = 1 - mAP@0.5
where each detector is normalised by its OWN clean error.  That biases the
comparison: a detector with a weak clean mAP (large E_clean) gets a flattering
mCE even if its absolute corrupted performance is poor.  We report instead:

  * map_clean, map_corr   — raw mAP@0.5 (clean and per-corruption, sev-averaged)
  * delta_map  = map_clean - map_corr            (raw absolute degradation, ↓ better)
  * mce_self   = E_c / E_clean                   (legacy self-relative, kept + flagged)
  * ce_shared  = E_c / E_clean_bar               (shared-baseline relative error)

where E_clean_bar is the MEAN clean error across all detectors — a common
denominator, so cross-detector robustness is comparable without the self-norm
bias.  mAP is computed with the same pycocotools wrapper (_compute_map) used
everywhere else in the codebase, for protocol consistency.

Outputs:
  results/corruption_robustness.csv          — per (detector, corruption) summary
  results/corruption_robustness_detail.csv   — per (detector, corruption, severity)
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.corruption.corruption_pipeline import CORRUPTION_REGISTRY, _compute_map

DETECTORS = ["yolov8m", "rtdetr", "faster_rcnn", "retinanet"]
SEVERITIES = [1, 2, 3, 4]


def cache_path(preds_dir: Path, model: str, cname: str | None = None,
               sev: int | None = None) -> Path:
    if cname is None:
        return preds_dir / f"{model}_test_clean.json"
    return preds_dir / f"{model}_test_{cname}_{sev}.json"


def split_pg(recs):
    """records -> (preds, gts) in _compute_map's expected form."""
    preds = [{"boxes": r["pred_boxes"], "scores": r["pred_scores"],
              "labels": r["pred_labels"]} for r in recs]
    gts = [{"boxes": r["gt_boxes"], "labels": r["gt_labels"]} for r in recs]
    return gts, preds


def map50_from_cache(path: Path) -> float | None:
    if not path.exists():
        return None
    recs = json.loads(path.read_text())
    if not recs:
        return None
    gts, preds = split_pg(recs)
    return _compute_map(gts, preds, iou_threshold=0.5)["mAP50"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(REPO))
    args = ap.parse_args()
    repo = Path(args.root)
    preds_dir = repo / "results/corruption_preds"
    if not preds_dir.exists():
        sys.exit(f"ERROR: {preds_dir} not found")

    # ── Pass 1: clean mAP per detector → shared baseline error ───────────────
    clean_map = {}
    for m in DETECTORS:
        v = map50_from_cache(cache_path(preds_dir, m))
        if v is None:
            print(f"[warn] no clean test cache for {m}")
            continue
        clean_map[m] = v
        print(f"  [{m}] clean mAP@0.5 = {v:.4f}")
    if not clean_map:
        sys.exit("No clean caches found.")
    e_clean_bar = sum(1.0 - v for v in clean_map.values()) / len(clean_map)
    print(f"\nShared-baseline mean clean error  E_clean_bar = {e_clean_bar:.4f}\n")

    detail_rows, summ_rows = [], []
    for m in DETECTORS:
        if m not in clean_map:
            continue
        e_clean = 1.0 - clean_map[m]
        for cname in CORRUPTION_REGISTRY:
            sev_maps = []
            for sev in SEVERITIES:
                v = map50_from_cache(cache_path(preds_dir, m, cname, sev))
                if v is None:
                    continue
                sev_maps.append(v)
                detail_rows.append({"detector": m, "corruption": cname,
                                    "severity": sev, "map50": round(v, 4)})
            if not sev_maps:
                continue
            map_corr = sum(sev_maps) / len(sev_maps)
            e_c = 1.0 - map_corr
            summ_rows.append({
                "detector": m, "corruption": cname,
                "map_clean": round(clean_map[m], 4),
                "map_corr":  round(map_corr, 4),
                "delta_map": round(clean_map[m] - map_corr, 4),
                "mce_self":  round(e_c / e_clean, 4) if e_clean > 1e-9 else float("nan"),
                "ce_shared": round(e_c / e_clean_bar, 4),
            })
            print(f"  [{m}] {cname:14s} map_corr={map_corr:.3f} "
                  f"Δ={clean_map[m]-map_corr:+.3f}  mce_self={e_c/max(e_clean,1e-9):.3f}  "
                  f"ce_shared={e_c/e_clean_bar:.3f}")

    # ── per-detector mean over corruptions ───────────────────────────────────
    for m in DETECTORS:
        rows = [r for r in summ_rows if r["detector"] == m]
        if not rows:
            continue
        n = len(rows)
        summ_rows.append({
            "detector": m, "corruption": "MEAN",
            "map_clean": round(clean_map[m], 4),
            "map_corr":  round(sum(r["map_corr"] for r in rows) / n, 4),
            "delta_map": round(sum(r["delta_map"] for r in rows) / n, 4),
            "mce_self":  round(sum(r["mce_self"] for r in rows) / n, 4),
            "ce_shared": round(sum(r["ce_shared"] for r in rows) / n, 4),
        })

    out_sum = repo / "results/corruption_robustness.csv"
    with open(out_sum, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["detector", "corruption", "map_clean",
                                          "map_corr", "delta_map", "mce_self", "ce_shared"])
        w.writeheader(); w.writerows(summ_rows)
    out_det = repo / "results/corruption_robustness_detail.csv"
    with open(out_det, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["detector", "corruption", "severity", "map50"])
        w.writeheader(); w.writerows(detail_rows)
    print(f"\nSaved: {out_sum}  ({len(summ_rows)} rows)")
    print(f"Saved: {out_det}  ({len(detail_rows)} rows)")


if __name__ == "__main__":
    main()
