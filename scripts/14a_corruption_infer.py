"""GPU enabling run — per-detection logging under corruption + LLVIP inference.

Produces the per-image prediction caches needed for all Phase-7 calibration,
threshold, RA-AP, and deferral analysis (scripts 14–19).  Run once on vast.ai;
results rsynced back to X9.

Outputs (all JSON, list of per-image dicts):
  results/corruption_preds/{model}_{corr_type}_{sev}.json
    — 5 models × 6 corruption types × 4 severities = 120 files
    — each file: list of {image_id, pred_boxes, pred_scores, pred_labels,
                          gt_boxes, gt_labels}
  results/corruption_preds/{model}_clean.json
    — clean baseline predictions (per model)
  results/llvip_transfer_preds.json
    — YOLOv8m-FLIR on LLVIP test split (1731 imgs); per-image preds + GT

Also rewrites results/corruption_comparison.csv with updated mCE.

Usage (from project root on GPU node):
    python scripts/14a_corruption_infer.py \\
        --flir-root /workspace/data/flir/FLIR_ADAS_v2 \\
        --llvip-root /workspace/data/llvip/LLVIP-YOLO \\
        --split-json configs/flir_val_split.json \\
        --llvip-split-json configs/llvip_val_split.json \\
        [--models yolov8m rtdetr faster_rcnn retinanet] \\
        [--corruptions gaussian_noise poisson_noise ...] \\
        [--dry-run]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np

try:
    import cv2
    _CV2 = True
except ImportError:
    _CV2 = False

from src.corruption.corruption_pipeline import CORRUPTION_REGISTRY, image_snr_db

# ---------------------------------------------------------------------------
# Detector configs (weights relative to project root; adjusted for /workspace)
# ---------------------------------------------------------------------------
# weights_rel paths match the actual X9 checkpoint names; the pod must receive
# results/<det>/... at the same relative path (rsync preserves it). Fallbacks
# cover the on-pod training layout (weights/best.pt) from the original run.
DETECTOR_CONFIGS = {
    "yolov8m": {
        "type": "ultralytics",
        "weights_rel": "results/yolov8m_flir_seed0/yolov8m_flir_final.pt",
        "weights_fallbacks": ["results/yolov8m_flir_seed0/weights/best.pt"],
    },
    "yolov11m": {
        "type": "ultralytics",
        "weights_rel": "results/yolov11m_flir_seed0/weights/best.pt",
        "weights_fallbacks": ["results/yolov11m_flir_seed0/yolov11m_flir_final.pt"],
    },
    "rtdetr": {
        "type": "ultralytics_rtdetr",
        "weights_rel": "results/rtdetr_flir_seed0/rtdetr_flir_final.pt",
        "weights_fallbacks": ["results/rtdetr_flir_seed0/rtdetr_flir/weights/best.pt"],
    },
    "faster_rcnn": {
        "type": "frcnn",
        "weights_rel": "results/faster_rcnn_flir_seed0/weights/frcnn_best.pth",
        "weights_fallbacks": ["results/faster_rcnn_flir_seed0/frcnn_best.pth"],
    },
    "retinanet": {
        "type": "retinanet",
        "weights_rel": "results/retinanet_flir_seed0/retinanet_best.pth",
        "weights_fallbacks": [],
    },
}

CLASS_MAP = {1: 0, 2: 1, 3: 2}   # FLIR COCO category IDs → 0-indexed
VALID_CATS = set(CLASS_MAP.keys())


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def _load_gray(path: str) -> np.ndarray:
    if _CV2:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(path)
        return img
    from PIL import Image
    return np.array(Image.open(path).convert("L"))


def _gray_to_3ch(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return np.stack([img] * 3, axis=-1)
    if img.ndim == 3 and img.shape[2] == 1:
        return np.concatenate([img] * 3, axis=-1)
    return img


# ---------------------------------------------------------------------------
# FLIR test split loader
# ---------------------------------------------------------------------------

def load_flir_split(flir_root: Path, split_json: Path,
                    which: str = "test") -> Tuple[List[str], List[Dict], List[int]]:
    """Return (image_paths, gt_records, image_ids) for the FLIR ``which`` split.

    ``which`` in {'calibration', 'test'}; only GT-bearing images of the kept
    classes are returned (background-only frames carry no detection signal).
    """
    split = json.loads(split_json.read_text())
    sel_ids = set(split[which])
    coco_ann = flir_root / "images_thermal_val/coco.json"
    gt_data = json.loads(coco_ann.read_text())
    id2file = {img["id"]: img["file_name"] for img in gt_data["images"]}
    ann_map: Dict[int, list] = defaultdict(list)
    for ann in gt_data["annotations"]:
        if ann["image_id"] in sel_ids and ann["category_id"] in VALID_CATS:
            ann_map[ann["image_id"]].append(ann)

    base = flir_root / "images_thermal_val"
    image_paths, gt_records, image_ids = [], [], []
    for img_id in sorted(sel_ids):
        if img_id not in id2file:
            continue
        anns = ann_map[img_id]
        if not anns:
            continue
        fpath = base / id2file[img_id]
        if not fpath.exists():
            continue
        boxes = [[a["bbox"][0], a["bbox"][1],
                  a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]]
                 for a in anns]
        labels = [CLASS_MAP[a["category_id"]] for a in anns]
        image_paths.append(str(fpath))
        gt_records.append({"boxes": boxes, "labels": labels})
        image_ids.append(img_id)

    return image_paths, gt_records, image_ids


# ---------------------------------------------------------------------------
# LLVIP test split loader
# ---------------------------------------------------------------------------

def load_llvip_test(llvip_root: Path, split_json: Path) -> Tuple[List[str], List[Dict], List[str]]:
    """Return (image_paths, gt_records, image_ids) for LLVIP test split."""
    split = json.loads(split_json.read_text())
    test_ids = set(split["test"])  # string IDs like '190001'

    img_dir = llvip_root / "infrared/test"
    if not img_dir.exists():
        img_dir = llvip_root / "images/test"  # alternate layout
    lbl_dir = llvip_root / "labels/test"

    image_paths, gt_records, image_ids = [], [], []
    for img_id in sorted(test_ids):
        # Try .jpg then .png
        fpath = None
        for ext in (".jpg", ".png"):
            p = img_dir / f"{img_id}{ext}"
            if p.exists():
                fpath = p
                break
        if fpath is None:
            continue
        lbl_file = lbl_dir / f"{img_id}.txt"
        if not lbl_file.exists():
            continue
        img = _load_gray(str(fpath))
        h, w = img.shape[:2]
        boxes, labels = [], []
        for line in lbl_file.read_text().strip().splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            cx, cy, bw, bh = (float(p) for p in parts[1:5])
            x1 = (cx - bw / 2) * w
            y1 = (cy - bh / 2) * h
            x2 = (cx + bw / 2) * w
            y2 = (cy + bh / 2) * h
            boxes.append([x1, y1, x2, y2])
            labels.append(cls)
        if not boxes:
            continue
        image_paths.append(str(fpath))
        gt_records.append({"boxes": boxes, "labels": labels})
        image_ids.append(img_id)

    return image_paths, gt_records, image_ids


# ---------------------------------------------------------------------------
# Detector factory
# ---------------------------------------------------------------------------

def build_predict_fn(det_name: str, repo: Path):
    cfg = DETECTOR_CONFIGS[det_name]
    weights = repo / cfg["weights_rel"]
    if not weights.exists():
        for fb in cfg.get("weights_fallbacks", []):
            if (repo / fb).exists():
                weights = repo / fb
                break
    if not weights.exists():
        sys.exit(f"ERROR: {det_name} weights not found at {cfg['weights_rel']} "
                 f"or fallbacks {cfg.get('weights_fallbacks', [])}")
    print(f"  [{det_name}] weights: {weights}")
    dtype = cfg["type"]

    if dtype in ("ultralytics", "ultralytics_rtdetr"):
        from ultralytics import YOLO
        model = YOLO(str(weights))

        def predict(imgs: List[np.ndarray]) -> List[Dict]:
            out = []
            for img in imgs:
                rgb = _gray_to_3ch(img)
                r = model.predict(rgb, conf=0.001, verbose=False)[0]
                if len(r.boxes):
                    boxes  = r.boxes.xyxy.cpu().numpy().tolist()
                    scores = r.boxes.conf.cpu().numpy().tolist()
                    labels = r.boxes.cls.cpu().numpy().astype(int).tolist()
                else:
                    boxes, scores, labels = [], [], []
                out.append({"boxes": boxes, "scores": scores, "labels": labels})
            return out
        return predict

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dtype == "frcnn":
        from src.detectors.faster_rcnn_wrapper import FasterRCNNDetector
        wrapper = FasterRCNNDetector(pretrained=False)
    else:
        from src.detectors.retinanet_wrapper import RetinaNetDetector
        wrapper = RetinaNetDetector(pretrained=False)
    state = torch.load(str(weights), map_location=device, weights_only=True)
    wrapper.model.load_state_dict(state)
    wrapper.model.to(device).eval()

    def predict(imgs: List[np.ndarray]) -> List[Dict]:
        import torchvision.transforms.functional as TF
        out = []
        for img in imgs:
            rgb = _gray_to_3ch(img)
            tensor = TF.to_tensor(rgb).to(device)
            with torch.no_grad():
                pred = wrapper.model([tensor])[0]
            boxes  = pred["boxes"].cpu().numpy().tolist()
            scores = pred["scores"].cpu().numpy().tolist()
            labels = (pred["labels"].cpu().numpy() - 1).tolist()
            out.append({"boxes": boxes, "scores": scores, "labels": labels})
        return out
    return predict


# ---------------------------------------------------------------------------
# mAP helper
# ---------------------------------------------------------------------------

def compute_map50(gt_list: List[Dict], pred_list: List[Dict]) -> float:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    gt_coco = {"images": [], "annotations": [],
               "categories": [{"id": i + 1, "name": n}
                               for i, n in enumerate(["person", "bicycle", "car"])]}
    dt_coco, ann_id = [], 1
    for img_id, (gt, _) in enumerate(zip(gt_list, pred_list), 1):
        gt_coco["images"].append({"id": img_id})
        for box, lab in zip(gt["boxes"], gt["labels"]):
            x1, y1, x2, y2 = [float(v) for v in box]
            gt_coco["annotations"].append({
                "id": ann_id, "image_id": img_id,
                "category_id": int(lab) + 1,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "area": max(0.0, x2 - x1) * max(0.0, y2 - y1), "iscrowd": 0,
            })
            ann_id += 1
    for img_id, pred in enumerate(pred_list, 1):
        for box, score, lab in zip(pred["boxes"], pred["scores"], pred["labels"]):
            x1, y1, x2, y2 = [float(v) for v in box]
            dt_coco.append({"image_id": img_id, "category_id": int(lab) + 1,
                            "bbox": [x1, y1, x2 - x1, y2 - y1],
                            "score": float(score)})
    if not dt_coco:
        return 0.0
    coco_gt = COCO(); coco_gt.dataset = gt_coco; coco_gt.createIndex()
    coco_dt = coco_gt.loadRes(dt_coco)
    ev = COCOeval(coco_gt, coco_dt, "bbox")
    ev.evaluate(); ev.accumulate(); ev.summarize()
    return float(ev.stats[1])


# ---------------------------------------------------------------------------
# Core: run one model on one (image_list, corruption, severity) condition
# ---------------------------------------------------------------------------

def run_condition(
    predict_fn,
    image_paths: List[str],
    gt_records: List[Dict],
    image_ids,
    corruption_fn=None,  # None = clean
    severity: int = 0,
    base_seed: Optional[int] = None,   # enables reproducible seeded corruption
    corr_idx: int = 0,                 # stable per-corruption-type index
) -> Tuple[List[Dict], float]:
    """Return (per_image_records, mAP50).

    When ``base_seed`` is set, each image gets an independent but reproducible
    noise realisation seeded by (base_seed, corr_idx, severity, image position).
    The seed is INDEPENDENT of the detector, so every model sees pixel-identical
    corrupted inputs -- essential for fair cross-detector comparison and for the
    corruption-conditioned CACH head.
    """
    records = []
    preds_for_map = []
    snr_vals: List[float] = []

    for i, (img_path, gt, img_id) in enumerate(zip(image_paths, gt_records, image_ids)):
        clean = _load_gray(img_path)
        img = clean
        if corruption_fn is not None:
            if base_seed is not None:
                ss = np.random.SeedSequence([base_seed, corr_idx, severity, i])
                img = corruption_fn(clean, severity, np.random.default_rng(ss))
            else:
                img = corruption_fn(clean, severity)
            snr = image_snr_db(clean, img)
            if np.isfinite(snr):
                snr_vals.append(snr)
        result = predict_fn([img])[0]
        records.append({
            "image_id": img_id,
            "pred_boxes":  result["boxes"],
            "pred_scores": result["scores"],
            "pred_labels": result["labels"],
            "gt_boxes":    gt["boxes"],
            "gt_labels":   gt["labels"],
        })
        preds_for_map.append(result)

    map50 = compute_map50(gt_records, preds_for_map)
    mean_snr = float(np.mean(snr_vals)) if snr_vals else None
    return records, map50, mean_snr


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flir-root", default=None,
                    help="Path to FLIR_ADAS_v2 root. Autodetected if absent.")
    ap.add_argument("--llvip-root", default=None,
                    help="Path to LLVIP-YOLO root. Autodetected if absent.")
    ap.add_argument("--split-json", default=str(REPO / "configs/flir_val_split.json"))
    ap.add_argument("--llvip-split-json", default=str(REPO / "configs/llvip_val_split.json"))
    ap.add_argument("--models", nargs="+",
                    default=list(DETECTOR_CONFIGS.keys()),
                    choices=list(DETECTOR_CONFIGS.keys()))
    ap.add_argument("--corruptions", nargs="+",
                    default=list(CORRUPTION_REGISTRY.keys()),
                    choices=list(CORRUPTION_REGISTRY.keys()))
    ap.add_argument("--severities", nargs="+", type=int, default=[1, 2, 3, 4])
    ap.add_argument("--split", choices=["calibration", "test", "both"],
                    default="both",
                    help="FLIR split(s) to regenerate corrupted caches for. "
                         "P3 needs BOTH so calibration can fit and test can report.")
    ap.add_argument("--seed", type=int, default=0,
                    help="Base seed for reproducible corruption (model-independent).")
    ap.add_argument("--skip-llvip", action="store_true",
                    help="Skip LLVIP inference (e.g. data not present).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print plan then exit.")
    args = ap.parse_args()

    # ── resolve paths ──────────────────────────────────────────────────────────
    flir_root = Path(args.flir_root) if args.flir_root else None
    if flir_root is None:
        for candidate in [
            Path("/workspace/data/flir/FLIR_ADAS_v2"),
            Path("/workspace/data/public/FLIR_ADAS_v2"),
            REPO / "datasets/flir_adas_v2/FLIR_ADAS_v2",
        ]:
            if candidate.exists():
                flir_root = candidate
                break
    if flir_root is None or not flir_root.exists():
        sys.exit(f"ERROR: FLIR_ADAS_v2 not found. Pass --flir-root.")
    print(f"FLIR root: {flir_root}")

    llvip_root = Path(args.llvip_root) if args.llvip_root else None
    if llvip_root is None and not args.skip_llvip:
        for candidate in [
            Path("/workspace/data/llvip/LLVIP-YOLO"),
            Path("/workspace/data/public/LLVIP-YOLO"),
            REPO / "datasets/LLVIP-YOLO",
        ]:
            if candidate.exists():
                llvip_root = candidate
                break
    if llvip_root is None and not args.skip_llvip:
        print("WARNING: LLVIP root not found, skipping LLVIP inference. Pass --llvip-root or --skip-llvip.")
        args.skip_llvip = True

    out_dir = REPO / "results/corruption_preds"
    out_dir.mkdir(parents=True, exist_ok=True)
    corr_idx = {n: i for i, n in enumerate(CORRUPTION_REGISTRY)}
    splits = ["calibration", "test"] if args.split == "both" else [args.split]

    n_conditions = (len(splits) * len(args.models)
                    * (1 + len(args.corruptions) * len(args.severities)))
    print(f"\nPlan: splits={splits}  {len(args.models)} models × (1 clean + "
          f"{len(args.corruptions)} corruptions × {len(args.severities)} severities) "
          f"= {n_conditions} conditions  (seed={args.seed})")
    print(f"Output dir: {out_dir}")
    if args.dry_run:
        print("--dry-run: exiting.")
        return

    # mCE / SNR tracking across all (split, model)
    all_mce_rows = []
    snr_rows = []  # SNR is model-independent; recorded once per (split,corr,sev)

    for which in splits:
        print(f"\n########## FLIR split: {which} ##########")
        image_paths, gt_records, image_ids = load_flir_split(
            flir_root, Path(args.split_json), which)
        print(f"  {len(image_paths)} {which} images (GT-bearing)")

        for model_name in args.models:
            print(f"\n{'='*60}\n  [{which}] Model: {model_name}\n{'='*60}")
            predict_fn = build_predict_fn(model_name, REPO)

            # Clean baseline (no seeding needed)
            print("  [clean] ...")
            clean_records, clean_m, _ = run_condition(
                predict_fn, image_paths, gt_records, image_ids)
            out_path = out_dir / f"{model_name}_{which}_clean.json"
            out_path.write_text(json.dumps(clean_records))
            print(f"    mAP@0.5 = {clean_m:.4f}  → {out_path.name}")

            # Corrupted conditions (seeded, model-independent corruption)
            sev_errors: Dict[str, List[float]] = defaultdict(list)
            for cname in args.corruptions:
                cfn = CORRUPTION_REGISTRY[cname]
                for sev in args.severities:
                    out_path = out_dir / f"{model_name}_{which}_{cname}_{sev}.json"
                    print(f"  [{cname} sev={sev}] ...")
                    records, m, snr = run_condition(
                        predict_fn, image_paths, gt_records, image_ids, cfn, sev,
                        base_seed=args.seed, corr_idx=corr_idx[cname])
                    out_path.write_text(json.dumps(records))
                    sev_errors[cname].append(1.0 - m)
                    # SNR is model-independent: record once (first model only)
                    if model_name == args.models[0]:
                        snr_rows.append({
                            "split": which, "corruption": cname, "severity": sev,
                            "snr_db": round(snr, 3) if snr is not None else None,
                        })
                    print(f"    mAP@0.5 = {m:.4f}  SNR = "
                          f"{('%.1f dB' % snr) if snr is not None else 'n/a'}"
                          f"  → {out_path.name}")

            # mCE (own-clean; kept for backward-compat, P4 uses absolute degradation)
            e_clean = max(1.0 - clean_m, 1e-6)
            for cname in args.corruptions:
                mce = float(np.mean(sev_errors[cname])) / e_clean
                all_mce_rows.append({
                    "split": which, "model": model_name, "corruption": cname,
                    "mCE": round(mce, 4), "clean_map50": round(clean_m, 4),
                    **{f"sev{s}_err": round(sev_errors[cname][i], 4)
                       for i, s in enumerate(args.severities)},
                })
            mean_mce = float(np.mean([r["mCE"] for r in all_mce_rows
                                      if r["model"] == model_name and r["split"] == which]))
            print(f"  [{which}/{model_name}] mean mCE = {mean_mce:.4f}")

            del predict_fn  # release GPU memory before next model
            try:
                import torch, gc
                torch.cuda.empty_cache(); gc.collect()
            except Exception:
                pass

    # ── LLVIP inference (YOLOv8m only) ────────────────────────────────────────
    if not args.skip_llvip and llvip_root is not None:
        print(f"\n{'='*60}\n  LLVIP inference (YOLOv8m)\n{'='*60}")
        llvip_paths, llvip_gt, llvip_ids = load_llvip_test(
            llvip_root, Path(args.llvip_split_json))
        print(f"  {len(llvip_paths)} LLVIP test images")
        predict_fn = build_predict_fn("yolov8m", REPO)
        llvip_records = []
        for img_path, gt, img_id in zip(llvip_paths, llvip_gt, llvip_ids):
            img = _load_gray(img_path)
            result = predict_fn([img])[0]
            llvip_records.append({
                "image_id": img_id,
                "pred_boxes":  result["boxes"],
                "pred_scores": result["scores"],
                "pred_labels": result["labels"],
                "gt_boxes":    gt["boxes"],
                "gt_labels":   gt["labels"],
            })
        llvip_out = REPO / "results/llvip_transfer_preds.json"
        llvip_out.write_text(json.dumps(llvip_records))
        llvip_map = compute_map50(llvip_gt,
                                  [{"boxes": r["pred_boxes"], "scores": r["pred_scores"],
                                    "labels": r["pred_labels"]} for r in llvip_records])
        print(f"  LLVIP mAP@0.5 (person) = {llvip_map:.4f}  → {llvip_out}")

    # ── Write mCE + SNR summaries ───────────────────────────────────────────────
    if all_mce_rows:
        mce_csv = REPO / "results/corruption_preds/mce_summary.csv"
        with open(mce_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_mce_rows[0].keys()))
            w.writeheader(); w.writerows(all_mce_rows)
        print(f"\nmCE summary: {mce_csv}")

    if snr_rows:
        snr_csv = REPO / "results/corruption_preds/snr_summary.csv"
        with open(snr_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(snr_rows[0].keys()))
            w.writeheader(); w.writerows(snr_rows)
        print(f"SNR summary: {snr_csv}")

    print("\nDone. rsync results/corruption_preds/ and "
          "results/llvip_transfer_preds.json back to X9.")


if __name__ == "__main__":
    main()
