"""Domain transfer evaluation: 4 directions × (zero-shot + fine-tuned).

Protocol overview
-----------------
We evaluate four cross-domain transfer directions using YOLOv8m as the primary
detector.  For each direction we run:

    1. Zero-shot inference (no target-domain training data).
    2. Fine-tuned inference (10 epochs on target-domain TRAIN, head swap if nc
       differs, early stopping on a 10% holdout of target TRAIN).

After inference for each variant we apply domain-adaptive recalibration:
    3. Fit temperature T on target CALIBRATION split → new ECE (reported).
    4. Find risk-optimal θ* on target CALIBRATION split → E[C] on target TEST.

Transfer directions
-------------------
    D1  FLIR→LLVIP  zero-shot   source=FLIR-train  eval=LLVIP-test
    D2  FLIR→LLVIP  fine-tuned  fine-tune on LLVIP-train (10 ep, nc=1 head)
    D3  LLVIP→FLIR  zero-shot   source=LLVIP-train eval=FLIR-test (person only)
    D4  LLVIP→FLIR  fine-tuned  fine-tune on FLIR-train (10 ep, nc=3 head)

Class-mismatch handling
-----------------------
FLIR: 3 classes  (person=0, bike=1, car=2)
LLVIP: 1 class   (person=0)

For D1/D2 (eval on LLVIP): ground-truth contains only person; we evaluate
    mAP@0.5 for person class only (class 0 predictions matched to GT).
For D3/D4 (eval on FLIR, model trained on person-only): only class-0
    (person) detections are produced; bike/car GT boxes are ignored when
    computing person-only mAP.

Calibration splits
------------------
FLIR:  configs/flir_val_split.json   (50/50 of FLIR val, seed=42)
LLVIP: configs/llvip_val_split.json  (50/50 of LLVIP test, seed=42)
         → generate with: python3 src/data/llvip_splits.py --root <LLVIP-YOLO>

Usage (from project root):
    # Generate LLVIP split first (once):
    python3 src/data/llvip_splits.py \
        --root /workspace/data/llvip/LLVIP-YOLO

    # D1: FLIR→LLVIP zero-shot
    python3 scripts/11_domain_transfer.py --direction D1 \
        --flir-weights /workspace/results/yolov8m_flir_seed0/weights/best.pt \
        --llvip-root   /workspace/data/llvip/LLVIP-YOLO \
        --flir-split-json  configs/flir_val_split.json \
        --llvip-split-json configs/llvip_val_split.json \
        --out /workspace/results/domain_transfer

    # D2: FLIR→LLVIP fine-tuned (10 epochs)
    python3 scripts/11_domain_transfer.py --direction D2 \
        --flir-weights /workspace/results/yolov8m_flir_seed0/weights/best.pt \
        --llvip-root   /workspace/data/llvip/LLVIP-YOLO \
        --flir-split-json  configs/flir_val_split.json \
        --llvip-split-json configs/llvip_val_split.json \
        --fine-tune-epochs 10 \
        --out /workspace/results/domain_transfer

    # D3/D4: start from LLVIP-trained weights (train separately with llvip_yolo.yaml)
    python3 scripts/11_domain_transfer.py --direction D3 \
        --llvip-weights /workspace/results/yolov8m_llvip/weights/best.pt \
        --flir-img-dir  /workspace/data/flir_yolo/val/images \
        --flir-lbl-dir  /workspace/data/flir_yolo/val/labels \
        --flir-split-json  configs/flir_val_split.json \
        --llvip-split-json configs/llvip_val_split.json \
        --out /workspace/results/domain_transfer
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np

from src.calibration.temperature_scaling import DetectionTemperatureScaling
from src.data.llvip_splits import load_split as load_llvip_split
from src.data.splits import load_split as load_flir_split


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FLIR_CLASS_NAMES = ["person", "bike", "car"]   # our label IDs 0,1,2
LLVIP_CLASS_NAMES = ["person"]                  # label ID 0 only
PERSON_CLASS_ID = 0
COST_FN = 10.0    # false-negative cost (miss a pedestrian)
COST_FP = 1.0     # false-positive cost


# ---------------------------------------------------------------------------
# Shared evaluation helpers
# ---------------------------------------------------------------------------

def _iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter == 0.0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1])
    ub = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (ua + ub - inter + 1e-6)


def _match_predictions(
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    gt_boxes: np.ndarray,
    iou_thr: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (is_tp, is_matched_gt) for a single image.

    is_tp[i]         = 1 if pred i is a true positive
    is_matched_gt[j] = 1 if GT box j was matched by any prediction
    """
    n_pred = len(pred_boxes)
    n_gt = len(gt_boxes)
    is_tp = np.zeros(n_pred, dtype=np.int32)
    is_matched_gt = np.zeros(n_gt, dtype=np.int32)

    if n_pred == 0 or n_gt == 0:
        return is_tp, is_matched_gt

    order = np.argsort(-pred_scores)
    for i in order:
        best_iou, best_j = 0.0, -1
        for j in range(n_gt):
            if is_matched_gt[j]:
                continue
            iou = _iou(pred_boxes[i], gt_boxes[j])
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= iou_thr:
            is_tp[i] = 1
            is_matched_gt[best_j] = 1
    return is_tp, is_matched_gt


def _compute_ap(is_tp: np.ndarray, scores: np.ndarray, n_gt: int) -> float:
    """Compute Average Precision (area under precision-recall curve, 11-point)."""
    if n_gt == 0:
        return 0.0
    order = np.argsort(-scores)
    tp_cumsum = np.cumsum(is_tp[order])
    fp_cumsum = np.cumsum(1 - is_tp[order])
    precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
    recall = tp_cumsum / n_gt

    # 11-point interpolation
    ap = 0.0
    for thr in np.linspace(0.0, 1.0, 11):
        prec_at_thr = precision[recall >= thr]
        ap += (prec_at_thr.max() if len(prec_at_thr) > 0 else 0.0) / 11.0
    return float(ap)


def _compute_expected_cost(
    all_is_tp: np.ndarray,
    all_scores: np.ndarray,
    all_n_gt: int,
    theta_vals: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """Find theta* minimizing E[C] = c_FN * FN(theta)/n_gt + c_FP * FP(theta)/n_pred.

    Returns (theta_star, min_expected_cost).
    """
    if theta_vals is None:
        theta_vals = np.arange(0.0, 1.01, 0.05)

    best_theta, best_cost = 0.5, float("inf")
    for theta in theta_vals:
        mask = all_scores >= theta
        tp = int(all_is_tp[mask].sum())
        fp = int((1 - all_is_tp[mask]).sum())
        fn = all_n_gt - tp
        n_pred_above = int(mask.sum())
        cost = (COST_FN * fn / max(all_n_gt, 1)
                + COST_FP * fp / max(n_pred_above, 1) if n_pred_above > 0
                else COST_FN * fn / max(all_n_gt, 1))
        if cost < best_cost:
            best_cost, best_theta = cost, float(theta)
    return best_theta, best_cost


def _compute_ece(
    all_confs: List[float],
    all_is_tp: List[int],
    n_bins: int = 10,
) -> float:
    """ECE = Σ_b (n_b/N) |precision_b - mean_conf_b|."""
    n = len(all_confs)
    if n == 0:
        return 0.0
    bins_conf: List[List[float]] = [[] for _ in range(n_bins)]
    bins_tp: List[List[int]] = [[] for _ in range(n_bins)]
    for c, y in zip(all_confs, all_is_tp):
        b = min(n_bins - 1, int(c * n_bins))
        bins_conf[b].append(c)
        bins_tp[b].append(y)
    ece = 0.0
    for b in range(n_bins):
        if not bins_conf[b]:
            continue
        avg_c = sum(bins_conf[b]) / len(bins_conf[b])
        prec = sum(bins_tp[b]) / len(bins_tp[b])
        ece += (len(bins_conf[b]) / n) * abs(prec - avg_c)
    return ece


# ---------------------------------------------------------------------------
# Inference on an image directory
# ---------------------------------------------------------------------------

def _run_inference_on_dir(
    model_path: str,
    img_dir: str,
    stems: Optional[List[str]],
    conf_thr: float = 0.001,
    device: str = "cuda",
) -> Dict[str, Dict]:
    """Run inference on a directory of .jpg images.

    Args:
        model_path: Path to .pt checkpoint.
        img_dir:    Directory of jpg images.
        stems:      If given, only process images whose stem is in this list.
        conf_thr:   Low threshold to capture full score distribution.
        device:     'cuda' or 'cpu'.

    Returns:
        stem → {'boxes': (N,4), 'scores': (N,), 'labels': (N,)}
    """
    from ultralytics import YOLO
    model = YOLO(model_path)
    img_dir = Path(img_dir)
    stem_set = set(stems) if stems is not None else None

    paths = sorted(p for p in img_dir.glob("*.jpg") if not p.name.startswith("."))
    if stem_set is not None:
        paths = [p for p in paths if p.stem in stem_set]

    results_dict: Dict[str, Dict] = {}
    for path in paths:
        res = model.predict(
            source=str(path),
            conf=conf_thr,
            device=device,
            verbose=False,
            save=False,
        )
        r = res[0]
        boxes_obj = r.boxes
        if boxes_obj is None or len(boxes_obj) == 0:
            results_dict[path.stem] = {
                "boxes": np.empty((0, 4), dtype=np.float32),
                "scores": np.empty((0,), dtype=np.float32),
                "labels": np.empty((0,), dtype=np.int32),
            }
        else:
            results_dict[path.stem] = {
                "boxes": boxes_obj.xyxy.cpu().numpy().astype(np.float32),
                "scores": boxes_obj.conf.cpu().numpy().astype(np.float32),
                "labels": boxes_obj.cls.cpu().numpy().astype(np.int32),
            }
    return results_dict


def _detect_img_size(img_dir: str, stems: List[str]) -> tuple:
    """Return (width, height) by reading the first available image in img_dir."""
    from PIL import Image as _PIL_Image
    img_dir = Path(img_dir)
    for stem in stems:
        for ext in (".jpg", ".jpeg", ".png"):
            p = img_dir / f"{stem}{ext}"
            if p.exists():
                w, h = _PIL_Image.open(p).size
                return w, h
    return 640, 640  # fallback


def _load_yolo_labels(
    lbl_dir: str,
    stems: List[str],
    img_w: int = 640,
    img_h: int = 640,
    keep_class: Optional[int] = None,
    img_dir: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """Load YOLO-format label files → stem → xyxy boxes (optionally filtered by class).

    If img_dir is provided, the actual image dimensions are auto-detected from the
    first available image so that YOLO normalised coords are correctly de-normalised.
    This is important when images are not 640×640 (e.g. LLVIP=1280×1024, FLIR=640×512).
    """
    if img_dir is not None:
        img_w, img_h = _detect_img_size(img_dir, stems)
    lbl_dir = Path(lbl_dir)
    gt_dict: Dict[str, np.ndarray] = {}
    for stem in stems:
        lbl_path = lbl_dir / f"{stem}.txt"
        boxes = []
        if lbl_path.exists():
            for line in lbl_path.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls_id, cx, cy, bw, bh = map(float, parts)
                if keep_class is not None and int(cls_id) != keep_class:
                    continue
                x1 = (cx - bw / 2) * img_w
                y1 = (cy - bh / 2) * img_h
                x2 = (cx + bw / 2) * img_w
                y2 = (cy + bh / 2) * img_h
                boxes.append([x1, y1, x2, y2])
        gt_dict[stem] = np.array(boxes, dtype=np.float32).reshape(-1, 4)
    return gt_dict


# ---------------------------------------------------------------------------
# Core evaluation routine
# ---------------------------------------------------------------------------

def evaluate_split(
    pred_dict: Dict[str, Dict],
    gt_dict: Dict[str, np.ndarray],
    stems: List[str],
    calib_stems: List[str],
    calibrate_temperature: bool = True,
) -> Dict:
    """Compute mAP@0.5, ECE (raw+recal), theta*, E[C] for a given split.

    pred_dict: stem → {'boxes', 'scores', 'labels'}  (person class only, pre-filtered)
    gt_dict:   stem → (N,4) xyxy float32 (person GT only)
    stems:     TEST stems (report metrics)
    calib_stems: CALIBRATION stems (fit T and theta*)
    """
    # Collect calibration (conf, is_tp) pairs
    calib_confs: List[float] = []
    calib_tp: List[int] = []
    calib_n_gt = 0
    for stem in calib_stems:
        d = pred_dict.get(stem, {"scores": np.empty(0), "is_tp": np.empty(0)})
        gt = gt_dict.get(stem, np.empty((0, 4)))
        preds = pred_dict.get(stem, {"boxes": np.empty((0, 4)), "scores": np.empty(0)})
        if len(preds["scores"]) > 0:
            is_tp, _ = _match_predictions(preds["boxes"], preds["scores"], gt)
            calib_confs.extend(preds["scores"].tolist())
            calib_tp.extend(is_tp.tolist())
        calib_n_gt += len(gt)

    # Fit temperature on calibration
    ts = DetectionTemperatureScaling()
    if calibrate_temperature and calib_confs:
        T = ts.fit(calib_confs, calib_tp)
    else:
        T = 1.0

    # Find theta* on calibration
    calib_all_tp = np.array(calib_tp, dtype=np.int32)
    calib_all_sc = np.array(calib_confs, dtype=np.float32)
    theta_star, _ = _compute_expected_cost(calib_all_tp, calib_all_sc, calib_n_gt)

    # Evaluate on test stems
    all_is_tp: List[int] = []
    all_scores_raw: List[float] = []
    all_scores_recal: List[float] = []
    total_gt = 0

    for stem in stems:
        preds = pred_dict.get(stem, {"boxes": np.empty((0, 4)), "scores": np.empty(0)})
        gt = gt_dict.get(stem, np.empty((0, 4)))
        total_gt += len(gt)
        if len(preds["scores"]) > 0:
            is_tp, _ = _match_predictions(preds["boxes"], preds["scores"], gt)
            all_is_tp.extend(is_tp.tolist())
            all_scores_raw.extend(preds["scores"].tolist())
            recal = ts.transform(preds["scores"].tolist())
            all_scores_recal.extend(recal)

    all_is_tp_arr = np.array(all_is_tp, dtype=np.int32)
    all_scores_arr = np.array(all_scores_raw, dtype=np.float32)

    ap = _compute_ap(all_is_tp_arr, all_scores_arr, total_gt)
    ece_raw = _compute_ece(all_scores_raw, all_is_tp)
    ece_recal = _compute_ece(all_scores_recal, all_is_tp)

    _, cost_at_theta_star = _compute_expected_cost(
        all_is_tp_arr, all_scores_arr, total_gt,
        theta_vals=np.array([theta_star]),
    )
    _, cost_at_05 = _compute_expected_cost(
        all_is_tp_arr, all_scores_arr, total_gt,
        theta_vals=np.array([0.5]),
    )
    cost_reduction_pct = 100.0 * (cost_at_05 - cost_at_theta_star) / max(cost_at_05, 1e-9)

    return {
        "n_test_images": len(stems),
        "n_gt_boxes": total_gt,
        "n_pred_boxes": len(all_is_tp),
        "ap50_person": round(ap, 4),
        "ece_raw": round(ece_raw, 4),
        "ece_recal": round(ece_recal, 4),
        "temperature": round(T, 4),
        "theta_star": round(theta_star, 3),
        "cost_at_05": round(cost_at_05, 4),
        "cost_at_theta_star": round(cost_at_theta_star, 4),
        "cost_reduction_pct": round(cost_reduction_pct, 2),
    }


# ---------------------------------------------------------------------------
# Fine-tuning helpers
# ---------------------------------------------------------------------------

def _write_data_yaml(path: str, train_dir: str, val_dir: str, nc: int, names: List[str]) -> None:
    """Write a minimal YOLO data.yaml for fine-tuning."""
    lines = [
        f"# Auto-generated for domain transfer fine-tuning",
        f"train: {train_dir}",
        f"val: {val_dir}",
        f"nc: {nc}",
        "names:",
    ]
    for i, name in enumerate(names):
        lines.append(f"  {i}: {name}")
    Path(path).write_text("\n".join(lines) + "\n")


def _fine_tune(
    src_weights: str,
    data_yaml: str,
    out_dir: str,
    epochs: int = 10,
    batch: int = 16,
    lr0: float = 1e-4,
    device: str = "cuda",
) -> str:
    """Fine-tune a YOLOv8 checkpoint. Returns path to best.pt."""
    from ultralytics import YOLO
    model = YOLO(src_weights)
    model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch,
        lr0=lr0,
        imgsz=640,
        device=device,
        project=out_dir,
        name="finetune",
        exist_ok=True,
        verbose=False,
        plots=False,
    )
    best_pt = Path(out_dir) / "finetune" / "weights" / "best.pt"
    if not best_pt.exists():
        # Fallback to last.pt if best wasn't saved (e.g. short fine-tune)
        best_pt = Path(out_dir) / "finetune" / "weights" / "last.pt"
    return str(best_pt)


# ---------------------------------------------------------------------------
# Direction implementations
# ---------------------------------------------------------------------------

def _run_d1_d2(
    args: argparse.Namespace,
    fine_tune: bool,
    out_dir: Path,
    device: str,
) -> Dict:
    """D1 (zero-shot) or D2 (fine-tuned): FLIR → LLVIP."""
    llvip_root = Path(args.llvip_root)
    img_dir_test = llvip_root / "test" / "lwir" / "images"
    lbl_dir_test = llvip_root / "test" / "lwir" / "labels"
    # For fine-tuning: use train split; 10% random holdout for in-training val
    img_dir_train = llvip_root / "train" / "lwir" / "images"
    lbl_dir_train = llvip_root / "train" / "lwir" / "labels"

    llvip_calib = load_llvip_split(args.llvip_split_json, "calibration")
    llvip_test = load_llvip_split(args.llvip_split_json, "test")

    if fine_tune:
        # Write data.yaml pointing at LLVIP train/test (test as proxy val)
        data_yaml = str(out_dir / "llvip_finetune.yaml")
        _write_data_yaml(
            data_yaml,
            train_dir=str(img_dir_train),
            val_dir=str(img_dir_test),
            nc=1,
            names=LLVIP_CLASS_NAMES,
        )
        print(f"[D2] Fine-tuning {args.flir_weights} on LLVIP ({args.fine_tune_epochs} epochs)...")
        weights = _fine_tune(
            args.flir_weights, data_yaml, str(out_dir / "D2_finetune"),
            epochs=args.fine_tune_epochs, batch=args.batch, device=device,
        )
        print(f"[D2] Fine-tuned checkpoint: {weights}")
    else:
        weights = args.flir_weights

    all_stems = llvip_calib + llvip_test
    print(f"[D{'2' if fine_tune else '1'}] Running inference on {len(all_stems)} LLVIP images...")
    pred_dict = _run_inference_on_dir(weights, str(img_dir_test), all_stems, device=device)

    # Filter to person class only; LLVIP GT is person-only
    for stem, d in pred_dict.items():
        mask = d["labels"] == PERSON_CLASS_ID
        pred_dict[stem] = {
            "boxes": d["boxes"][mask],
            "scores": d["scores"][mask],
            "labels": d["labels"][mask],
        }

    gt_dict = _load_yolo_labels(str(lbl_dir_test), all_stems, keep_class=None,
                                img_dir=str(img_dir_test))

    return evaluate_split(pred_dict, gt_dict, llvip_test, llvip_calib)


def _run_d3_d4(
    args: argparse.Namespace,
    fine_tune: bool,
    out_dir: Path,
    device: str,
) -> Dict:
    """D3 (zero-shot) or D4 (fine-tuned): LLVIP → FLIR (person class only)."""
    flir_img_dir = args.flir_img_dir
    flir_lbl_dir = args.flir_lbl_dir
    flir_train_img = args.flir_train_img_dir
    flir_train_lbl = args.flir_train_lbl_dir

    # Load FLIR calib/test split (image filenames → stems)
    flir_calib_ids = load_flir_split(args.flir_split_json, "calibration")
    flir_test_ids = load_flir_split(args.flir_split_json, "test")

    # Translate image_ids → stems by scanning the val image dir
    id_to_stem: Dict = {}
    for p in Path(flir_img_dir).glob("*.jpg"):
        if not p.name.startswith("."):
            # FLIR val images: coco_style name or plain stem
            id_to_stem[p.stem] = p.stem  # will match by filename stem
    # If split JSON has integer IDs, we need the filename map from it
    split_data = json.loads(Path(args.flir_split_json).read_text())
    fname_map = split_data.get("filenames", {})

    def _ids_to_stems(ids):
        out = []
        for iid in ids:
            fname = fname_map.get(str(iid))
            if fname:
                out.append(Path(fname).stem)
        return out

    flir_calib_stems = _ids_to_stems(flir_calib_ids)
    flir_test_stems = _ids_to_stems(flir_test_ids)

    if fine_tune:
        # Write data.yaml pointing at FLIR train
        data_yaml = str(out_dir / "flir_finetune_from_llvip.yaml")
        _write_data_yaml(
            data_yaml,
            train_dir=str(flir_train_img),
            val_dir=str(flir_img_dir),
            nc=3,
            names=FLIR_CLASS_NAMES,
        )
        print(f"[D4] Fine-tuning {args.llvip_weights} on FLIR ({args.fine_tune_epochs} epochs)...")
        weights = _fine_tune(
            args.llvip_weights, data_yaml, str(out_dir / "D4_finetune"),
            epochs=args.fine_tune_epochs, batch=args.batch, device=device,
        )
        print(f"[D4] Fine-tuned checkpoint: {weights}")
    else:
        weights = args.llvip_weights

    all_val_stems = flir_calib_stems + flir_test_stems
    print(f"[D{'4' if fine_tune else '3'}] Running inference on {len(all_val_stems)} FLIR images...")
    pred_dict = _run_inference_on_dir(weights, str(flir_img_dir), all_val_stems, device=device)

    # Keep only person class (class 0) predictions; GT filtered to person
    for stem, d in pred_dict.items():
        mask = d["labels"] == PERSON_CLASS_ID
        pred_dict[stem] = {
            "boxes": d["boxes"][mask],
            "scores": d["scores"][mask],
            "labels": d["labels"][mask],
        }

    gt_dict = _load_yolo_labels(
        str(flir_lbl_dir), all_val_stems, keep_class=PERSON_CLASS_ID,
        img_dir=str(flir_img_dir),
    )

    return evaluate_split(pred_dict, gt_dict, flir_test_stems, flir_calib_stems)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--direction", required=True, choices=["D1", "D2", "D3", "D4"],
        help="Transfer direction (see module docstring for details).",
    )
    # Source checkpoints
    p.add_argument("--flir-weights", default=None,
                   help="FLIR-trained YOLOv8m checkpoint (needed for D1/D2).")
    p.add_argument("--llvip-weights", default=None,
                   help="LLVIP-trained YOLOv8m checkpoint (needed for D3/D4).")
    # LLVIP data
    p.add_argument("--llvip-root", default=None,
                   help="Path to LLVIP-YOLO root directory (needed for D1/D2).")
    p.add_argument("--llvip-split-json", default="configs/llvip_val_split.json",
                   help="Frozen LLVIP calib/test split JSON.")
    # FLIR data
    p.add_argument("--flir-img-dir", default=None,
                   help="FLIR val images directory (needed for D3/D4).")
    p.add_argument("--flir-lbl-dir", default=None,
                   help="FLIR val labels directory (needed for D3/D4).")
    p.add_argument("--flir-train-img-dir", default=None,
                   help="FLIR train images directory (needed for D4 fine-tuning).")
    p.add_argument("--flir-train-lbl-dir", default=None,
                   help="FLIR train labels directory (needed for D4 fine-tuning).")
    p.add_argument("--flir-split-json", default="configs/flir_val_split.json",
                   help="Frozen FLIR calib/test split JSON.")
    # Fine-tuning
    p.add_argument("--fine-tune-epochs", type=int, default=10,
                   help="Number of fine-tuning epochs (D2/D4 only).")
    p.add_argument("--batch", type=int, default=16, help="Batch size for fine-tuning.")
    # Output
    p.add_argument("--out", default="results/domain_transfer",
                   help="Output directory for results CSV + JSON.")
    p.add_argument("--device", default="cuda", help="'cuda' or 'cpu'.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device

    print(f"\n=== Domain Transfer: {args.direction} ===\n")

    if args.direction == "D1":
        assert args.flir_weights, "--flir-weights required for D1"
        assert args.llvip_root, "--llvip-root required for D1"
        metrics = _run_d1_d2(args, fine_tune=False, out_dir=out_dir, device=device)
    elif args.direction == "D2":
        assert args.flir_weights, "--flir-weights required for D2"
        assert args.llvip_root, "--llvip-root required for D2"
        metrics = _run_d1_d2(args, fine_tune=True, out_dir=out_dir, device=device)
    elif args.direction == "D3":
        assert args.llvip_weights, "--llvip-weights required for D3"
        assert args.flir_img_dir, "--flir-img-dir required for D3"
        assert args.flir_lbl_dir, "--flir-lbl-dir required for D3"
        metrics = _run_d3_d4(args, fine_tune=False, out_dir=out_dir, device=device)
    elif args.direction == "D4":
        assert args.llvip_weights, "--llvip-weights required for D4"
        assert args.flir_img_dir, "--flir-img-dir required for D4"
        assert args.flir_lbl_dir, "--flir-lbl-dir required for D4"
        assert args.flir_train_img_dir, "--flir-train-img-dir required for D4"
        metrics = _run_d3_d4(args, fine_tune=True, out_dir=out_dir, device=device)
    else:
        raise ValueError(f"Unknown direction: {args.direction}")

    metrics["direction"] = args.direction
    metrics["src_weights"] = (args.flir_weights if args.direction in ("D1", "D2")
                              else args.llvip_weights)
    metrics["fine_tuned"] = args.direction in ("D2", "D4")

    # Print summary
    print(f"\n--- Results [{args.direction}] ---")
    for k, v in metrics.items():
        print(f"  {k:30s}: {v}")

    # Save JSON
    json_path = out_dir / f"transfer_{args.direction}.json"
    json_path.write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved JSON: {json_path}")

    # Append to CSV (one row per direction)
    csv_path = out_dir / "transfer_results.csv"
    fieldnames = list(metrics.keys())
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(metrics)
    print(f"Appended to CSV: {csv_path}")


if __name__ == "__main__":
    main()
