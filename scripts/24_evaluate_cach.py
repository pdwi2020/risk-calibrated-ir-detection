"""8.3 CACH Evaluation — compare vs {no-cal, clean-cal, pooled-cal, per-type oracle, CACH}.

For each model × corruption × severity:
  1. no-cal     — raw detector confidence
  2. clean-cal  — isotonic fit on clean calib split
  3. pooled-cal — isotonic fit on clean + all corruptions
  4. oracle     — isotonic fit on same-type corruption (oracle knows the type)
  5. CACH       — our learned corruption-adaptive head (no corruption label)

Outputs:
  results/cach_eval.csv        — ECE per condition per protocol
  results/fig_cach.pdf         — ECE comparison bar chart + line plot

Usage:
    python scripts/24_evaluate_cach.py [--root <project_root>]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import cv2
import numpy as np
import torch
from sklearn.isotonic import IsotonicRegression

from src.calibration.cach import CACH, preprocess_patch, match_detections_to_gt
from src.corruption.corruption_pipeline import CORRUPTION_REGISTRY

DETECTORS = ["yolov8m", "rtdetr", "faster_rcnn", "retinanet"]
IOU_THRESH = 0.5
N_BINS = 15
_CORR_IDX = {n: i for i, n in enumerate(CORRUPTION_REGISTRY)}


def cache_path(preds_dir: Path, model: str, split: str,
               cname=None, sev: int = 0) -> Path:
    """Per-split corruption cache path, falling back to the legacy (test-only,
    un-split) names so this still runs against pre-P3 caches if needed."""
    if cname is None:
        p = preds_dir / f"{model}_{split}_clean.json"
        return p if p.exists() else preds_dir / f"{model}_clean.json"
    p = preds_dir / f"{model}_{split}_{cname}_{sev}.json"
    return p if p.exists() else preds_dir / f"{model}_{cname}_{sev}.json"


# ---------------------------------------------------------------------------
# Helpers shared with script 14
# ---------------------------------------------------------------------------

def load_json(path: Path):
    if not path.exists():
        return []
    return json.loads(path.read_text())


def extract_scores_labels(records):
    all_s, all_tp = [], []
    for rec in records:
        s, tp = match_detections_to_gt(
            rec["pred_boxes"], rec["pred_scores"], rec["pred_labels"],
            rec["gt_boxes"], rec["gt_labels"], IOU_THRESH)
        all_s.append(s); all_tp.append(tp)
    if not all_s:
        return np.array([]), np.array([])
    return np.concatenate(all_s), np.concatenate(all_tp)


def ece(scores: np.ndarray, is_tp: np.ndarray, n_bins: int = N_BINS) -> float:
    if len(scores) == 0:
        return float("nan")
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (scores > lo) & (scores <= hi)
        if not mask.any():
            continue
        e += abs(is_tp[mask].mean() - scores[mask].mean()) * mask.mean()
    return float(e)


def fit_iso(scores, is_tp):
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(scores, is_tp)
    return ir


# ---------------------------------------------------------------------------
# CACH calibration helper
# ---------------------------------------------------------------------------

def build_cach_calib_fn(cach_model: CACH, flir_root: Path, device: str):
    """Return a function scores → cal_scores using CACH with a mean embedding."""
    data_dir = flir_root / "images_thermal_val" / "data"
    paths = sorted(data_dir.glob("*.jpg"))[:50]
    patches = []
    for p in paths:
        g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if g is not None:
            patches.append(preprocess_patch(g))
    if not patches:
        return None
    patch_stack = torch.cat(patches).to(device)

    cach_model.eval()
    with torch.no_grad():
        embeds = cach_model.embed_net(patch_stack)  # (N, embed_dim)
        mean_embed = embeds.mean(0, keepdim=True)   # (1, embed_dim)

    def calibrate_fn(scores_arr: np.ndarray) -> np.ndarray:
        s = torch.tensor(scores_arr, dtype=torch.float32).to(device)
        e = mean_embed.expand(len(s), -1)
        with torch.no_grad():
            cal = cach_model.calib_head(s, e)
        return cal.cpu().numpy()

    return calibrate_fn


def _build_id2path(flir_root: Path) -> dict:
    """Build COCO image_id → absolute Path mapping from the val COCO JSON."""
    coco_json = flir_root / "images_thermal_val" / "coco.json"
    if not coco_json.exists():
        return {}
    data = json.loads(coco_json.read_text())
    base = flir_root / "images_thermal_val"
    return {img["id"]: base / img["file_name"] for img in data["images"]}


def build_cach_per_image_fn(cach_model: CACH, flir_root: Path, records,
                             device: str, id2path: dict = None,
                             corr_name=None, severity: int = 0):
    """Apply CACH per-image using the COCO ID→file mapping.

    CRITICAL FIX: when evaluating a corrupted condition, the SAME corruption
    (``corr_name`` at ``severity``) is applied to the image BEFORE it is fed to
    CorruptionEmbedNet -- otherwise CACH sees a clean image and cannot infer the
    corruption it is meant to adapt to. The corruption is seeded reproducibly
    (per image/type/severity), an independent draw from the same family that
    produced the cached predictions.
    """
    if id2path is None:
        id2path = _build_id2path(flir_root)
    # Fallback: first image in the directory if no mapping available
    data_dir = flir_root / "images_thermal_val" / "data"
    fallback_paths = sorted(data_dir.glob("*.jpg"))
    cfn = CORRUPTION_REGISTRY[corr_name] if corr_name is not None else None
    cach_model.eval()

    all_s, all_tp = [], []
    for rec in records:
        img_id = rec.get("image_id")
        img_path = id2path.get(img_id) if img_id is not None else None
        if img_path is None or not img_path.exists():
            if fallback_paths:
                img_path = fallback_paths[0]
            else:
                continue
        g = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        if cfn is not None:
            key = f"{img_path}|{corr_name}|{severity}".encode()
            s_seed = int(hashlib.md5(key).hexdigest()[:8], 16)
            g = cfn(g, severity, np.random.default_rng(s_seed))
        patch = preprocess_patch(g).to(device)

        scores = np.array(rec.get("pred_scores", []), dtype=np.float32)
        if len(scores) == 0:
            continue
        _, is_tp = match_detections_to_gt(
            rec["pred_boxes"], rec["pred_scores"], rec["pred_labels"],
            rec["gt_boxes"], rec["gt_labels"], IOU_THRESH)
        if len(is_tp) == 0:
            continue

        s_t = torch.tensor(scores).to(device)
        with torch.no_grad():
            cal = cach_model.calibrate(patch, s_t)
        all_s.append(cal.cpu().numpy())
        all_tp.append(is_tp)

    if not all_s:
        return np.array([]), np.array([])
    return np.concatenate(all_s), np.concatenate(all_tp)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root",  default=str(REPO))
    ap.add_argument("--cach",  default="results/cach/cach_best.pt")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    repo = Path(args.root)
    preds_dir  = repo / "results/corruption_preds"
    flir_root  = repo / "datasets/flir_adas_v2/FLIR_ADAS_v2"
    # --cach is kept for backward compatibility but is no longer used; each
    # detector loads its own per-detector checkpoint inside the loop below.

    if not preds_dir.exists():
        sys.exit(f"ERROR: {preds_dir} not found")

    id2path = _build_id2path(flir_root)
    print(f"COCO ID→path: {len(id2path)} images")

    rows = []

    for model_name in DETECTORS:
        # ── Load per-detector CACH checkpoint ─────────────────────────────
        ckpt_path = repo / "results/cach" / f"{model_name}_cach_best.pt"
        if not ckpt_path.exists():
            print(f"[skip] no CACH checkpoint for {model_name} "
                  f"(expected: {ckpt_path})")
            continue
        print(f"\nLoading CACH for {model_name} from {ckpt_path} ...")
        cach = CACH.load(str(ckpt_path), map_location=args.device)
        cach.eval()
        print(f"  Parameters: {cach.param_count():,}")
        eval_dir   = repo / f"results/{model_name}_flir_seed0/eval"
        prefix_map = {"yolov8m": "yolov8m", "rtdetr": "rtdetr",
                      "faster_rcnn": "frcnn", "retinanet": "retinanet"}
        prefix     = prefix_map[model_name]
        calib_json = eval_dir / f"{prefix}_calib_predictions.json"
        test_clean = cache_path(preds_dir, model_name, "test")

        if not test_clean.exists():
            print(f"  [{model_name}] test clean preds missing — skipping.")
            continue

        # ── Clean-cal isotonic (fit on CLEAN calibration) ─────────────────
        ir_clean = None
        if calib_json.exists():
            c_recs = load_json(calib_json)
            c_s, c_tp = extract_scores_labels(c_recs)
            if len(c_s) > 0:
                ir_clean = fit_iso(c_s, c_tp)

        # ── Pooled-cal isotonic: FIT on CALIBRATION (clean + all corruptions),
        #    leakage-free w.r.t. the test conditions it is scored on. ────────
        pool_s, pool_tp = [], []
        for cname in CORRUPTION_REGISTRY:
            for sev in [1, 2, 3, 4]:
                recs = load_json(cache_path(preds_dir, model_name, "calibration", cname, sev))
                if recs:
                    s, tp = extract_scores_labels(recs)
                    pool_s.append(s); pool_tp.append(tp)
        recs0 = load_json(cache_path(preds_dir, model_name, "calibration"))
        if recs0:
            s0, tp0 = extract_scores_labels(recs0)
            pool_s.append(s0); pool_tp.append(tp0)
        ps = np.concatenate(pool_s) if pool_s else np.array([])
        pp = np.concatenate(pool_tp) if pool_tp else np.array([])
        ir_pooled = fit_iso(ps, pp) if len(ps) > 0 else None
        print(f"  [{model_name}] pooled (calibration fit): {len(ps):,} dets")

        for cname in CORRUPTION_REGISTRY:
            # Per-type oracle isotonic: FIT on CALIBRATION corrupted of this type.
            ts, ttp_list = [], []
            for sev in [1, 2, 3, 4]:
                recs = load_json(cache_path(preds_dir, model_name, "calibration", cname, sev))
                if recs:
                    s, tp = extract_scores_labels(recs)
                    ts.append(s); ttp_list.append(tp)
            ir_oracle = fit_iso(np.concatenate(ts), np.concatenate(ttp_list)) \
                if ts else None

            for sev in [1, 2, 3, 4]:
                # EVALUATE on the held-out TEST corrupted condition.
                recs = load_json(cache_path(preds_dir, model_name, "test", cname, sev))
                if not recs:
                    continue
                scores, is_tp = extract_scores_labels(recs)
                if len(scores) == 0:
                    continue

                ece_nocal  = ece(scores, is_tp)
                ece_clean  = ece(ir_clean.predict(scores), is_tp) if ir_clean else float("nan")
                ece_pooled = ece(ir_pooled.predict(scores), is_tp) if ir_pooled else float("nan")
                ece_oracle = ece(ir_oracle.predict(scores), is_tp) if ir_oracle else float("nan")

                # CACH per-image calibration: corruption APPLIED to the image.
                cach_s, cach_tp = build_cach_per_image_fn(
                    cach, flir_root, recs, args.device, id2path=id2path,
                    corr_name=cname, severity=sev)
                ece_cach = ece(cach_s, cach_tp) if len(cach_s) > 0 else float("nan")

                rows.append({
                    "model": model_name, "corruption": cname, "severity": sev,
                    "n_dets": len(scores),
                    "ece_nocal":  round(ece_nocal,  4),
                    "ece_clean":  round(ece_clean,  4),
                    "ece_pooled": round(ece_pooled, 4),
                    "ece_oracle": round(ece_oracle, 4),
                    "ece_cach":   round(ece_cach,   4),
                })

        print(f"  [{model_name}] done ({len(CORRUPTION_REGISTRY)*4} conditions)")

    if not rows:
        sys.exit("No data rows produced.")

    out_csv = repo / "results/cach_eval.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nSaved: {out_csv}  ({len(rows)} rows)")

    _make_figure(rows, repo / "results/fig_cach.pdf")


def _make_figure(rows, out_path: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError:
        print("matplotlib/pandas not available — skipping figure.")
        return

    df = pd.DataFrame(rows)
    protocols = ["ece_nocal", "ece_clean", "ece_pooled", "ece_oracle", "ece_cach"]
    labels    = ["No-cal", "Clean-cal", "Pooled-cal", "Oracle", "CACH (ours)"]
    colors    = ["#9E9E9E", "#F44336", "#FF9800", "#4CAF50", "#2196F3"]

    # ── Fig 1: mean ECE by protocol, grouped by corruption ──────────────
    summary = df.groupby("corruption")[protocols].mean()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(summary))
    width = 0.15
    for i, (proto, lbl, col) in enumerate(zip(protocols, labels, colors)):
        axes[0].bar(x + i * width, summary[proto], width, label=lbl, color=col)
    axes[0].set_xticks(x + width * 2)
    axes[0].set_xticklabels(
        [c.replace("_", "\n") for c in summary.index], fontsize=8)
    axes[0].set_ylabel("ECE (↓ better)")
    axes[0].set_title("ECE by Corruption Type\n(averaged over models & severities)")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.3)

    # ── Fig 2: ECE vs severity, averaged over models & corruptions ──────
    by_sev = df.groupby("severity")[protocols].mean()
    for proto, lbl, col in zip(protocols, labels, colors):
        axes[1].plot(by_sev.index, by_sev[proto], "o-", label=lbl, color=col, lw=1.5)
    axes[1].set_xlabel("Corruption severity")
    axes[1].set_ylabel("ECE (↓ better)")
    axes[1].set_title("ECE vs Severity\n(averaged over models & corruptions)")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(out_path), bbox_inches="tight")
    plt.savefig(str(out_path.with_suffix(".png")), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

    # Print summary
    mean_by_proto = df[protocols].mean()
    print("\nMean ECE across all conditions:")
    for proto, lbl in zip(protocols, labels):
        print(f"  {lbl:<15}: {mean_by_proto[proto]:.4f}")

    cach_vs_oracle = mean_by_proto["ece_cach"] - mean_by_proto["ece_oracle"]
    cach_vs_pooled = mean_by_proto["ece_pooled"] - mean_by_proto["ece_cach"]
    print(f"\nCACH vs oracle gap  : +{cach_vs_oracle:.4f}")
    print(f"CACH improvement over pooled-cal: {cach_vs_pooled:.4f} "
          f"({cach_vs_pooled/mean_by_proto['ece_pooled']*100:.1f}%)")


if __name__ == "__main__":
    main()
