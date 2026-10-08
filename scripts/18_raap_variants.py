"""7.4 RA-AP Variants and mAP-vs-RA-AP Ranking Divergence.

Computes RA-AP under four conditions:
  RA-AP-global   — clean test split, global cost (c_FN=10, c_FP=1)
  RA-AP-person   — person-class weighted (c_FN=15 for person safety)
  RA-AP-corrupt  — averaged over 6×4 corruption conditions
  RA-AP-domain   — FLIR→LLVIP zero-shot condition

Main result: Table showing mAP rank vs RA-AP rank — at least one inversion
expected (model that ranks best by mAP is not best by RA-AP under shift).

Inputs:
  results/corruption_preds/ (from 14a)
  results/llvip_transfer_preds.json (from 14a)
  results/{det}_flir_seed0/eval/{prefix}_test_{predictions,map}.json

Outputs:
  results/raap_variants.csv        — RA-AP values per (model, variant)
  results/ranking_divergence.csv   — rank tables: mAP vs RA-AP per condition
  (no separate figure — ranking_divergence is the paper table)

Usage:
    python scripts/18_raap_variants.py [--root <project_root>]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np
from src.risk.cost_sensitive import CostSensitiveThreshold
from src.corruption.corruption_pipeline import CORRUPTION_REGISTRY, _compute_map

ALL_VARIANTS = ["global", "person_weighted", "corruption_avg", "domain_llvip"]


def _corr_cache(preds_dir: Path, model: str, split: str, cname: str, sev: int) -> Path:
    """Per-split corrupted cache path with fallback to the legacy (test-only) name."""
    p = preds_dir / f"{model}_{split}_{cname}_{sev}.json"
    return p if p.exists() else preds_dir / f"{model}_{cname}_{sev}.json"


def _split_pg(recs):
    """records -> (preds, gts) in {boxes,scores,labels} / {boxes,labels} form."""
    preds = [{"boxes": r["pred_boxes"], "scores": r["pred_scores"],
              "labels": r["pred_labels"]} for r in recs]
    gts = [{"boxes": r["gt_boxes"], "labels": r["gt_labels"]} for r in recs]
    return preds, gts


DETECTORS = {
    "yolov8m":     "yolov8m",
    "rtdetr":      "rtdetr",
    "faster_rcnn": "frcnn",
    "retinanet":   "retinanet",
}
THETA_SWEEP = [round(0.05 * i, 2) for i in range(1, 20)]


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


def load_map50(path: Path) -> float:
    if not path.exists():
        return 0.0
    data = json.loads(path.read_text())
    return float(data.get("mAP50", data.get("map50", 0.0)))


def compute_raap(cs: CostSensitiveThreshold,
                 calib_preds, calib_gt,
                 test_preds, test_gt, map50: float) -> Dict:
    """Leakage-free RA-AP: fit theta* on the CALIBRATION (preds,gt), evaluate the
    per-image cost on the disjoint TEST split, and discount mAP by the normalised
    risk rho = per_image_cost / (c_FN * mean_gt_per_image) (density-invariant)."""
    theta, _ = cs.optimize_threshold(calib_preds, calib_gt, THETA_SWEEP)
    cost_total = cs.compute_cost(test_preds, test_gt, theta)
    n = max(len(test_preds), 1)
    cost_per_img = cost_total / n
    gbar = cs.mean_gt_per_image(test_gt)
    raap = cs.risk_adjusted_ap(map50, cost_per_img, gbar)
    rho = cost_per_img / (cs.c_fn * max(gbar, 1e-9))
    return {"theta_star": round(theta, 2), "cost": round(cost_per_img, 4),
            "rho": round(rho, 4), "gbar": round(gbar, 3),
            "raap": round(raap, 4), "map50": round(map50, 4)}


def run_analysis(repo: Path, variants=None):
    variants = set(variants or ALL_VARIANTS)
    preds_dir = repo / "results/corruption_preds"
    llvip_path = repo / "results/llvip_transfer_preds.json"

    raap_rows = []
    rank_rows = []

    # ── Collect per-model metrics for each variant ─────────────────────────
    variant_results: Dict[str, Dict[str, Dict]] = {m: {} for m in DETECTORS}

    for model_name, prefix in DETECTORS.items():
        eval_dir = repo / f"results/{model_name}_flir_seed0/eval"
        calib_p  = load_preds(eval_dir / f"{prefix}_calib_predictions.json")
        test_p   = load_preds(eval_dir / f"{prefix}_test_predictions.json")
        calib_g  = load_gt(eval_dir / f"{prefix}_calib_predictions.json")
        test_g   = load_gt(eval_dir / f"{prefix}_test_predictions.json")
        map50    = load_map50(eval_dir / f"{prefix}_test_map.json")

        if not calib_p:
            print(f"  [{model_name}] missing eval files — skipping."); continue

        # 1. Global (clean, standard cost): theta* on clean calib, eval clean test
        cs_std = CostSensitiveThreshold(c_fn=10, c_fp=1)
        if "global" in variants:
            v_global = compute_raap(cs_std, calib_p, calib_g, test_p, test_g, map50)
            variant_results[model_name]["global"] = v_global
            print(f"  [{model_name}] global  mAP={map50:.4f}  RA-AP={v_global['raap']:.4f} "
                  f"(rho={v_global['rho']}, gbar={v_global['gbar']})")

        # 2. Person-weighted (c_FN=15)
        if "person_weighted" in variants:
            cs_person = CostSensitiveThreshold(c_fn=15, c_fp=1)
            v_person = compute_raap(cs_person, calib_p, calib_g, test_p, test_g, map50)
            variant_results[model_name]["person_weighted"] = v_person
            print(f"  [{model_name}] person  RA-AP={v_person['raap']:.4f}")

        # 3. Corruption (mean RA-AP over 6x4 conditions, LEAKAGE-FREE):
        #    theta* on the CALIBRATION corrupted cache, cost + corrupted mAP on TEST.
        if "corruption_avg" in variants and preds_dir.exists():
            raap_corr_vals, corr_maps = [], []
            for cname in CORRUPTION_REGISTRY:
                for sev in [1, 2, 3, 4]:
                    cal_path  = _corr_cache(preds_dir, model_name, "calibration", cname, sev)
                    test_path = _corr_cache(preds_dir, model_name, "test", cname, sev)
                    if not cal_path.exists() or not test_path.exists():
                        continue
                    cal_cp, cal_cg = _split_pg(json.loads(cal_path.read_text()))
                    tst_cp, tst_cg = _split_pg(json.loads(test_path.read_text()))
                    corr_map = _compute_map(tst_cg, tst_cp, 0.5)["mAP50"]
                    v = compute_raap(cs_std, cal_cp, cal_cg, tst_cp, tst_cg, corr_map)
                    raap_corr_vals.append(v["raap"]); corr_maps.append(corr_map)
            if raap_corr_vals:
                variant_results[model_name]["corruption_avg"] = {
                    "raap": round(float(np.mean(raap_corr_vals)), 4),
                    "map50": round(float(np.mean(corr_maps)), 4),
                    "theta_star": None, "cost": None}
                print(f"  [{model_name}] corruption-avg  mAP={np.mean(corr_maps):.4f}  "
                      f"RA-AP={np.mean(raap_corr_vals):.4f}")

        # 4. Domain (LLVIP zero-shot)
        if "domain_llvip" in variants and llvip_path.exists():
            recs_llvip = json.loads(llvip_path.read_text())
            # Use only test split portion
            split_json = json.loads(
                (repo / "configs/llvip_val_split.json").read_text())
            test_ids = set(split_json["test"])
            llvip_test_p = [{"boxes": r["pred_boxes"], "scores": r["pred_scores"],
                              "labels": r["pred_labels"]}
                             for r in recs_llvip if str(r["image_id"]) in test_ids]
            llvip_test_g = [{"boxes": r["gt_boxes"], "labels": r["gt_labels"]}
                             for r in recs_llvip if str(r["image_id"]) in test_ids]
            if llvip_test_p:
                # mAP on LLVIP test (person-only, from domain_transfer results)
                d1 = repo / "results/domain_transfer/transfer_D1.json"
                llvip_map = json.loads(d1.read_text()).get("ap50_person", 0.5)
                # Use FLIR calib predictions as "source" calib for θ*
                v_domain = compute_raap(cs_std, calib_p, calib_g,
                                         llvip_test_p, llvip_test_g, llvip_map)
                variant_results[model_name]["domain_llvip"] = v_domain
                print(f"  [{model_name}] domain(LLVIP) RA-AP={v_domain['raap']:.4f}")

    # ── Build raap_variants rows ───────────────────────────────────────────
    for model_name in DETECTORS:
        for variant in ALL_VARIANTS:
            v = variant_results[model_name].get(variant)
            if v is None:
                continue
            raap_rows.append({
                "model": model_name, "variant": variant,
                "map50":  v["map50"],  "raap":       v["raap"],
                "theta_star": v.get("theta_star"), "cost": v.get("cost"),
            })

    # ── Build ranking divergence table ────────────────────────────────────
    for variant in ALL_VARIANTS:
        vdata = [(m, variant_results[m].get(variant, {}).get("raap"),
                  variant_results[m].get(variant, {}).get("map50"))
                 for m in DETECTORS if variant_results[m].get(variant)]
        if len(vdata) < 2:
            continue
        vdata = [(m, r, mp) for m, r, mp in vdata if r is not None and mp is not None]
        if not vdata:
            continue
        models_v = [x[0] for x in vdata]
        raaps_v  = [x[1] for x in vdata]
        map50s_v = [x[2] for x in vdata]

        # A ranking requires variation in the reference metric.  Some variants
        # (e.g. domain_llvip) use a single shared transfer AP rather than a
        # per-detector mAP measurement; then every detector ties at the same
        # mAP, the mAP ranking is undefined, and any apparent "inversion" is an
        # artifact of comparing a varying RA-AP against a constant.  Skip these.
        if len({round(mp, 4) for mp in map50s_v}) < 2:
            print(f"  [{variant}] mAP identical across detectors "
                  f"(={map50s_v[0]:.4f}) — mAP ranking undefined, variant "
                  f"excluded from divergence table.")
            continue

        map_ranks  = [sorted(map50s_v, reverse=True).index(mp) + 1 for mp in map50s_v]
        raap_ranks = [sorted(raaps_v,  reverse=True).index(rp) + 1 for rp in raaps_v]
        divergence = sum(abs(m - r) for m, r in zip(map_ranks, raap_ranks))

        for model_name, mp, rp, mr, rr in zip(models_v, map50s_v, raaps_v,
                                               map_ranks, raap_ranks):
            rank_rows.append({
                "variant": variant, "model": model_name,
                "map50": mp, "raap": rp,
                "map_rank": mr, "raap_rank": rr,
                "rank_inversion": int(mr != rr),
                "total_divergence": divergence,
            })
        n_inversions = sum(int(m != r) for m, r in zip(map_ranks, raap_ranks))
        print(f"  [{variant}] rank inversions={n_inversions}/{len(models_v)}  "
              f"total_divergence={divergence}")

    return raap_rows, rank_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(REPO))
    ap.add_argument("--variants", nargs="+", default=ALL_VARIANTS,
                    choices=ALL_VARIANTS,
                    help="Which RA-AP variants to compute. Clean variants "
                         "(global, person_weighted) need no corrupted caches; "
                         "corruption_avg/domain_llvip require the P3 caches.")
    args = ap.parse_args()
    repo = Path(args.root)

    print(f"Running RA-AP variants {args.variants} + ranking divergence ...")
    raap_rows, rank_rows = run_analysis(repo, variants=args.variants)

    if not raap_rows:
        sys.exit("No data.")

    out_raap = repo / "results/raap_variants.csv"
    with open(out_raap, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(raap_rows[0].keys()))
        w.writeheader(); w.writerows(raap_rows)
    print(f"Saved: {out_raap}")

    out_rank = repo / "results/ranking_divergence.csv"
    if rank_rows:
        with open(out_rank, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rank_rows[0].keys()))
            w.writeheader(); w.writerows(rank_rows)
        print(f"Saved: {out_rank}")

        total_inv = sum(r["rank_inversion"] for r in rank_rows)
        print(f"\nTotal rank inversions across all variants: {total_inv}")


if __name__ == "__main__":
    main()
