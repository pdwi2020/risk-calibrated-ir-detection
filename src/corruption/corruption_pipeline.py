"""src/corruption/corruption_pipeline.py — complete implementation.

6 corruption types x 4 severity levels following ImageNet-C conventions,
adapted for 8-bit grayscale thermal IR images.

Corruption types:
  1. gaussian_noise   — additive AWGN
  2. poisson_noise    — shot noise (Poisson-distributed)
  3. impulse_noise    — salt-and-pepper
  4. motion_blur      — linear kernel
  5. fog              — Koschmieder attenuation (intensity-domain)
  6. resolution       — bicubic downscale + upscale (pixelation)

Note on IR fog realism (#24):
  Thermal cameras see emitted radiation, not reflected light. Water-based
  fog (droplet diameter ~10 um) attenuates 8-14 um LWIR by only 10-30%
  per 100m (vs. near-total attenuation in visible). We model this as a
  MILD attenuation (severity 1-4 = transmittance 0.90/0.80/0.65/0.50)
  rather than the full additive fog model used for RGB ImageNet-C.
  This is flagged as a limitation in the paper.

mCE (mean Corruption Error):
  mCE_c = E_c / E_clean   where E = 1 - mAP@0.5 (higher is worse)
  mCE   = (1/C) * sum_c mCE_c  averaged over severities 1-4 per corruption
  A model robust to corruptions has mCE < 1 (better than clean baseline on
  corrupted data, relative to inter-model gap — but we report raw mCE).
"""
from __future__ import annotations

import csv
import json
import math
import os
import random
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import cv2
    _CV2 = True
except ImportError:
    _CV2 = False

try:
    from PIL import Image as PILImage
    _PIL = True
except ImportError:
    _PIL = False


# ---------------------------------------------------------------------------
# Severity parameter tables  (indexed 1..4)
# ---------------------------------------------------------------------------
_GAUSS_STD   = {1: 8,  2: 16, 3: 32, 4: 56}    # uint8 DN
_POISSON_LAM = {1: 60, 2: 30, 3: 15, 4: 8}      # mean photon count (higher = less noise)
_IMPULSE_P   = {1: 0.03, 2: 0.06, 3: 0.12, 4: 0.22}
_BLUR_K      = {1: 5,  2: 9,  3: 13, 4: 19}     # kernel length (pixels)
_FOG_T       = {1: 0.90, 2: 0.80, 3: 0.65, 4: 0.50}  # IR transmittance
_RES_FACTOR  = {1: 2,  2: 3,  3: 4,  4: 6}      # downscale factor


# ---------------------------------------------------------------------------
# Per-image corruption functions (numpy / pure-Python, no GPU)
# input/output: np.ndarray uint8 HxW or HxWxC
# ---------------------------------------------------------------------------

def gaussian_noise(img: np.ndarray, severity: int) -> np.ndarray:
    std = _GAUSS_STD[severity]
    noise = np.random.normal(0, std, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def poisson_noise(img: np.ndarray, severity: int) -> np.ndarray:
    lam = _POISSON_LAM[severity]
    # Scale image to [0, lam], apply Poisson, scale back
    scale = lam / 255.0
    noisy = np.random.poisson(img.astype(np.float32) * scale).astype(np.float32)
    return np.clip(noisy / scale, 0, 255).astype(np.uint8)


def impulse_noise(img: np.ndarray, severity: int) -> np.ndarray:
    p = _IMPULSE_P[severity]
    out = img.copy()
    mask = np.random.random(img.shape[:2]) < p
    vals = np.random.choice([0, 255], size=mask.sum())
    if img.ndim == 3:
        out[mask] = vals[:, None]
    else:
        out[mask] = vals
    return out


def motion_blur(img: np.ndarray, severity: int) -> np.ndarray:
    k = _BLUR_K[severity]
    kernel = np.zeros((k, k), dtype=np.float32)
    kernel[k // 2, :] = 1.0 / k  # horizontal
    if _CV2:
        out = cv2.filter2D(img, -1, kernel)
    else:
        # Pure numpy: convolve each channel
        from scipy.ndimage import convolve
        if img.ndim == 3:
            out = np.stack([convolve(img[:, :, c].astype(np.float32), kernel)
                            for c in range(img.shape[2])], axis=-1).astype(np.uint8)
        else:
            out = convolve(img.astype(np.float32), kernel).astype(np.uint8)
    return out


def fog(img: np.ndarray, severity: int) -> np.ndarray:
    """Koschmieder model adapted for IR: I_corrupt = T*I + (1-T)*128.
    T < 1 attenuates contrast; mean is preserved near 128 (midpoint).
    This is more realistic for LWIR thermal than additive white fog.
    """
    T = _FOG_T[severity]
    mid = 128.0
    out = T * img.astype(np.float32) + (1 - T) * mid
    return np.clip(out, 0, 255).astype(np.uint8)


def resolution(img: np.ndarray, severity: int) -> np.ndarray:
    """Bicubic downsample by factor, then upsample back."""
    f = _RES_FACTOR[severity]
    h, w = img.shape[:2]
    small_h, small_w = max(1, h // f), max(1, w // f)
    if _CV2:
        small = cv2.resize(img, (small_w, small_h), interpolation=cv2.INTER_CUBIC)
        out = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    elif _PIL:
        mode = "L" if img.ndim == 2 else "RGB"
        pil = PILImage.fromarray(img, mode=mode)
        small = pil.resize((small_w, small_h), PILImage.BICUBIC)
        out = np.array(small.resize((w, h), PILImage.BICUBIC))
    else:
        # Fallback: simple nearest
        out = np.repeat(np.repeat(img[::f, ::f], f, axis=0), f, axis=1)[:h, :w]
    return np.clip(out, 0, 255).astype(np.uint8)


CORRUPTION_REGISTRY: Dict[str, Any] = {
    "gaussian_noise": gaussian_noise,
    "poisson_noise":  poisson_noise,
    "impulse_noise":  impulse_noise,
    "motion_blur":    motion_blur,
    "fog":            fog,
    "resolution":     resolution,
}


# ---------------------------------------------------------------------------
# mAP computation (requires pycocotools)
# ---------------------------------------------------------------------------

def _compute_map(gt_list: List[Dict], pred_list: List[Dict],
                 iou_threshold: float = 0.5) -> Dict[str, float]:
    """Wrapper around pycocotools COCOeval.
    gt_list / pred_list: lists of dicts with keys boxes/labels(/scores).
    Returns {"mAP50": float, "mAP50-95": float}.
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    gt_coco: Dict = {"images": [], "annotations": [], "categories": [
        {"id": 1, "name": "person"}, {"id": 2, "name": "bicycle"},
        {"id": 3, "name": "car"},
    ]}
    dt_coco: List = []
    ann_id = 1

    for img_id, (gt, _) in enumerate(zip(gt_list, pred_list), 1):
        gt_coco["images"].append({"id": img_id})
        for box, lab in zip(gt.get("boxes", []), gt.get("labels", [])):
            x1, y1, x2, y2 = [float(v) for v in box]
            w, h = x2 - x1, y2 - y1
            gt_coco["annotations"].append({
                "id": ann_id, "image_id": img_id,
                "category_id": int(lab) + 1,
                "bbox": [x1, y1, w, h], "area": w * h, "iscrowd": 0,
            })
            ann_id += 1

    for img_id, pred in enumerate(pred_list, 1):
        for box, score, lab in zip(
            pred.get("boxes", []), pred.get("scores", []), pred.get("labels", [])
        ):
            x1, y1, x2, y2 = [float(v) for v in box]
            w, h = x2 - x1, y2 - y1
            dt_coco.append({
                "image_id": img_id, "category_id": int(lab) + 1,
                "bbox": [x1, y1, w, h], "score": float(score),
            })

    coco_gt = COCO()
    coco_gt.dataset = gt_coco
    coco_gt.createIndex()

    if not dt_coco:
        return {"mAP50": 0.0, "mAP50-95": 0.0}

    coco_dt = coco_gt.loadRes(dt_coco)
    ev = COCOeval(coco_gt, coco_dt, "bbox")
    ev.evaluate(); ev.accumulate(); ev.summarize()
    return {"mAP50": float(ev.stats[1]), "mAP50-95": float(ev.stats[0])}


# ---------------------------------------------------------------------------
# Main evaluation driver
# ---------------------------------------------------------------------------

def run_corruption_eval(
    detector_predict_fn,
    image_paths: List[str],
    gt_records: List[Dict],
    output_dir: str,
    corruption_types: Optional[List[str]] = None,
    severities: Sequence[int] = (1, 2, 3, 4),
    iou_threshold: float = 0.5,
    conf_threshold: float = 0.001,
    clean_map: Optional[float] = None,
) -> Dict:
    """Run corruption robustness evaluation.

    Args:
        detector_predict_fn: callable(images: List[np.ndarray]) -> List[Dict]
            Each dict has keys: boxes (List[[x1,y1,x2,y2]]), scores, labels.
        image_paths: test split image file paths (same order as gt_records).
        gt_records: per-image GT dicts with keys: boxes, labels.
        output_dir: results are written here.
        corruption_types: subset of CORRUPTION_REGISTRY keys; None = all 6.
        severities: which severity levels to evaluate (default 1-4).
        iou_threshold: IoU threshold for mAP computation.
        conf_threshold: prediction confidence threshold (use 0.001 to keep all).
        clean_map: clean-condition mAP@0.5 for mCE normalisation.
                   If None, computes it from un-corrupted images first.
    Returns:
        results dict with structure:
          {"clean": {"mAP50": float, "mAP50-95": float},
           "corruptions": {name: {severity: {"mAP50": float, "mAP50-95": float}}},
           "mce": {name: float},
           "mean_mce": float}
    """
    corr_names = corruption_types or list(CORRUPTION_REGISTRY.keys())
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    def _load_image(p: str) -> np.ndarray:
        if _CV2:
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(p)
            return img
        elif _PIL:
            return np.array(PILImage.open(p).convert("L"))
        else:
            raise RuntimeError("Need cv2 or Pillow to load images")

    def _predict_batch(imgs: List[np.ndarray]) -> List[Dict]:
        # Detector expects uint8 numpy arrays; wrap single if needed
        raw = detector_predict_fn(imgs)
        preds = []
        for r in raw:
            boxes = r.get("boxes", [])
            scores = r.get("scores", [])
            labels = r.get("labels", [])
            # apply confidence threshold
            keep = [i for i, s in enumerate(scores) if float(s) >= conf_threshold]
            preds.append({
                "boxes":  [boxes[i] for i in keep],
                "scores": [scores[i] for i in keep],
                "labels": [labels[i] for i in keep],
            })
        return preds

    # ── 1. Clean baseline ───────────────────────────────────────────────────
    print("  [clean] running detector on clean images ...")
    clean_imgs = [_load_image(p) for p in image_paths]
    clean_preds = _predict_batch(clean_imgs)
    clean_metrics = _compute_map(gt_records, clean_preds, iou_threshold)
    if clean_map is None:
        clean_map = clean_metrics["mAP50"]
    print(f"    clean mAP@0.5 = {clean_metrics['mAP50']:.4f}")

    # ── 2. Corrupted conditions ─────────────────────────────────────────────
    corruption_results: Dict[str, Dict[int, Dict]] = {}
    mce_per_corr: Dict[str, float] = {}

    for cname in corr_names:
        fn = CORRUPTION_REGISTRY[cname]
        corruption_results[cname] = {}
        sev_errors = []

        for sev in severities:
            print(f"  [{cname} sev={sev}] corrupting + predicting ...")
            corr_imgs = [fn(img, sev) for img in clean_imgs]
            corr_preds = _predict_batch(corr_imgs)
            metrics = _compute_map(gt_records, corr_preds, iou_threshold)
            corruption_results[cname][sev] = metrics
            sev_errors.append(1.0 - metrics["mAP50"])
            print(f"    mAP@0.5 = {metrics['mAP50']:.4f}")

        # mCE = mean(E_corrupted / E_clean) across severities
        e_clean = max(1.0 - clean_map, 1e-6)
        mce = float(np.mean(sev_errors)) / e_clean
        mce_per_corr[cname] = round(mce, 4)
        print(f"  [{cname}] mCE = {mce:.4f}")

    mean_mce = float(np.mean(list(mce_per_corr.values())))

    results = {
        "clean": clean_metrics,
        "corruptions": {k: {str(s): v for s, v in sv.items()}
                        for k, sv in corruption_results.items()},
        "mce": mce_per_corr,
        "mean_mce": round(mean_mce, 4),
    }

    # ── 3. Write outputs ────────────────────────────────────────────────────
    json_path = out / "corruption_results.json"
    json_path.write_text(json.dumps(results, indent=2))
    print(f"\n  Results JSON: {json_path}")

    csv_path = out / "mce_summary.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["corruption"] + [f"mAP50_sev{s}" for s in severities] + ["mCE"])
        for cname in corr_names:
            row = [cname]
            for s in severities:
                row.append(round(corruption_results[cname][s]["mAP50"], 4))
            row.append(mce_per_corr[cname])
            w.writerow(row)
        w.writerow(["mean_mCE"] + [""] * len(severities) + [round(mean_mce, 4)])
    print(f"  mCE CSV: {csv_path}")

    return results


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Corruption pipeline self-test (synthetic 64x64 grayscale image)")
    img = np.full((64, 64), 128, dtype=np.uint8)
    img[20:45, 20:45] = 200  # bright box (simulates thermal signature)

    for name, fn in CORRUPTION_REGISTRY.items():
        for s in [1, 2, 3, 4]:
            out = fn(img, s)
            assert out.shape == img.shape, f"{name} sev={s}: shape mismatch"
            assert out.dtype == np.uint8, f"{name} sev={s}: dtype mismatch"
        print(f"  {name}: OK (sev 1-4)")
    print("All corruption functions pass shape/dtype checks.")
