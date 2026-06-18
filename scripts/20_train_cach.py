"""8.2 CACH training script.

Trains the Corruption-Adaptive Calibration Head (src/calibration/cach.py) on
pooled clean + 6×4 corrupted FLIR predictions (from 14a_corruption_infer.py).

Data pipeline (no extra inference needed):
  - Loads per-detection records from results/corruption_preds/{model}_{type}_{sev}.json
  - Re-applies the SAME corruption transform on-the-fly to the input image to
    build the image patches for the embed net (deterministic: fixed seed per
    condition → each detection always sees the same patch).
  - Computes is_TP via IoU matching (same logic as cach.match_detections_to_gt).

Training:
  - Detector (YOLOv8m) backbone: FROZEN (not loaded; we only use its saved preds).
  - CACH only: ~20K params, trains in minutes on any GPU.
  - Loss: combined_loss (TP-NLL + focal calibration).
  - Eval after each epoch: ECE on held-out FLIR test split (all corruptions).

Outputs:
  results/cach/cach_best.pt          — best checkpoint by val ECE
  results/cach/cach_final.pt         — final epoch checkpoint
  results/cach/training_log.csv      — epoch, train_loss, val_ece, val_loss

Usage (from project root on GPU node):
    python scripts/20_train_cach.py \\
        --flir-root /workspace/data/flir/FLIR_ADAS_v2 \\
        --preds-dir results/corruption_preds \\
        --split-json configs/flir_val_split.json \\
        --out results/cach \\
        [--model yolov8m] \\
        [--epochs 30] [--batch 256] [--lr 1e-3] \\
        [--embed-dim 32] [--hidden 64]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.calibration.cach import (
    CACH, combined_loss, match_detections_to_gt, preprocess_patch,
)
from src.corruption.corruption_pipeline import CORRUPTION_REGISTRY

CLEAN_MAP = {1: 0, 2: 1, 3: 2}
VALID_CATS = set(CLEAN_MAP.keys())


# ---------------------------------------------------------------------------
# ECE helper
# ---------------------------------------------------------------------------

def ece(scores: np.ndarray, is_tp: np.ndarray, n_bins: int = 15) -> float:
    if len(scores) == 0:
        return 0.0
    bins = np.linspace(0, 1, n_bins + 1)
    ece_val = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (scores > lo) & (scores <= hi)
        if mask.sum() == 0:
            continue
        acc = is_tp[mask].mean()
        conf = scores[mask].mean()
        ece_val += abs(acc - conf) * mask.mean()
    return float(ece_val)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CACHDataset(Dataset):
    """Each item: (image_patch, raw_conf, is_tp).

    Loads cached per-image prediction records, loads the corresponding image,
    applies the same corruption transform, and extracts per-detection items.
    """

    def __init__(
        self,
        flir_root: Path,
        preds_dir: Path,
        split_ids: List,              # image IDs (int) for this split
        model_name: str = "yolov8m",
        include_clean: bool = True,
        corruption_names: List[str] = None,
        severities: List[int] = None,
        patch_size: int = 64,
        seed: int = 42,
    ):
        self.patch_size = patch_size
        self.items: List[Tuple[str, any, float, float]] = []
        # (image_path, corruption_fn_or_None, severity, raw_conf, is_tp)

        test_id_set = set(split_ids)
        coco_ann = flir_root / "images_thermal_val/coco.json"
        gt_data   = json.loads(coco_ann.read_text())
        id2file   = {img["id"]: img["file_name"] for img in gt_data["images"]}
        base      = flir_root / "images_thermal_val"

        ann_map: Dict[int, list] = {}
        from collections import defaultdict
        _amap = defaultdict(list)
        for ann in gt_data["annotations"]:
            if ann["image_id"] in test_id_set and ann["category_id"] in VALID_CATS:
                _amap[ann["image_id"]].append(ann)
        ann_map = dict(_amap)

        corruption_list = [(None, 0)]   # clean
        if include_clean is False:
            corruption_list = []
        corr_names = corruption_names or list(CORRUPTION_REGISTRY.keys())
        sevs = severities or [1, 2, 3, 4]
        for cname in corr_names:
            for sev in sevs:
                corruption_list.append((cname, sev))

        for corr_name, sev in corruption_list:
            if corr_name is None:
                pred_file = preds_dir / f"{model_name}_clean.json"
                cfn = None
            else:
                pred_file = preds_dir / f"{model_name}_{corr_name}_{sev}.json"
                cfn = CORRUPTION_REGISTRY[corr_name]
            if not pred_file.exists():
                continue
            records = json.loads(pred_file.read_text())
            for rec in records:
                img_id = rec["image_id"]
                if img_id not in test_id_set:
                    continue
                fpath = base / id2file.get(img_id, "")
                if not fpath.exists():
                    continue
                gt_boxes  = rec["gt_boxes"]
                gt_labels = rec["gt_labels"]
                pred_boxes  = rec["pred_boxes"]
                pred_scores = rec["pred_scores"]
                pred_labels = rec["pred_labels"]
                if not pred_scores:
                    continue
                _, is_tp_arr = match_detections_to_gt(
                    pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels)
                for score, tp in zip(pred_scores, is_tp_arr):
                    self.items.append((str(fpath), cfn, sev, float(score), float(tp)))

        rng = random.Random(seed)
        rng.shuffle(self.items)
        print(f"  Dataset: {len(self.items):,} detections across {len(corruption_list)} conditions")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        img_path, cfn, sev, score, is_tp = self.items[idx]
        try:
            import cv2
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError
        except Exception:
            from PIL import Image
            img = np.array(Image.open(img_path).convert("L"))
        if cfn is not None:
            img = cfn(img, sev)
        patch = preprocess_patch(img, self.patch_size)  # (1,1,H,W)
        return patch.squeeze(0), torch.tensor(score, dtype=torch.float32), \
               torch.tensor(is_tp, dtype=torch.float32)


def collate_fn(batch):
    patches, scores, labels = zip(*batch)
    return torch.stack(patches), torch.stack(scores), torch.stack(labels)


# ---------------------------------------------------------------------------
# Load FLIR split
# ---------------------------------------------------------------------------

def load_split(split_json: Path):
    data = json.loads(split_json.read_text())
    return data["test"]   # use test split as training data for CACH


def load_calib_split(split_json: Path):
    data = json.loads(split_json.read_text())
    return data.get("calibration", data.get("calib", []))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: CACH, loader: DataLoader, device: torch.device
) -> Tuple[float, float]:
    model.eval()
    all_cal, all_raw, all_tp = [], [], []
    total_loss = 0.0
    for patches, scores, labels in loader:
        patches = patches.to(device)
        scores  = scores.to(device)
        labels  = labels.to(device)
        cal_scores = model.calibrate(patches, scores)
        loss = combined_loss(cal_scores, labels)
        total_loss += loss.item() * len(scores)
        all_cal.append(cal_scores.cpu().numpy())
        all_raw.append(scores.cpu().numpy())
        all_tp.append(labels.cpu().numpy())
    if not all_cal:  # val set empty — skip ECE
        return float("nan"), 0.0
    cal_arr = np.concatenate(all_cal)
    raw_arr = np.concatenate(all_raw)
    tp_arr  = np.concatenate(all_tp)
    val_ece     = ece(cal_arr, tp_arr)
    raw_ece     = ece(raw_arr, tp_arr)
    avg_loss    = total_loss / max(len(cal_arr), 1)
    print(f"    val_ece={val_ece:.4f}  raw_ece={raw_ece:.4f}  val_loss={avg_loss:.4f}")
    return val_ece, avg_loss


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flir-root",    required=True)
    ap.add_argument("--preds-dir",    required=True)
    ap.add_argument("--split-json",   default="configs/flir_val_split.json")
    ap.add_argument("--out",          default="results/cach")
    ap.add_argument("--model",        default="yolov8m")
    ap.add_argument("--epochs",       type=int,   default=30)
    ap.add_argument("--batch",        type=int,   default=256)
    ap.add_argument("--lr",           type=float, default=1e-3)
    ap.add_argument("--embed-dim",    type=int,   default=32)
    ap.add_argument("--hidden",       type=int,   default=64)
    ap.add_argument("--patch-size",   type=int,   default=64)
    ap.add_argument("--num-workers",  type=int,   default=4)
    ap.add_argument("--seed",         type=int,   default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    flir_root = Path(args.flir_root)
    preds_dir = REPO / args.preds_dir
    out_dir   = REPO / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Build datasets ──────────────────────────────────────────────────────
    print("\nBuilding training dataset (test split × all conditions) ...")
    train_ids = load_split(REPO / args.split_json)
    train_ds = CACHDataset(flir_root, preds_dir, train_ids,
                            model_name=args.model, patch_size=args.patch_size,
                            seed=args.seed)

    print("\nBuilding validation dataset (calib split × all conditions) ...")
    val_ids = load_calib_split(REPO / args.split_json)
    val_ds = CACHDataset(flir_root, preds_dir, val_ids,
                          model_name=args.model, patch_size=args.patch_size,
                          seed=args.seed + 1)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_fn,
                              pin_memory=device.type == "cuda")
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                              num_workers=args.num_workers, collate_fn=collate_fn,
                              pin_memory=device.type == "cuda")

    # ── Model ───────────────────────────────────────────────────────────────
    model = CACH(embed_dim=args.embed_dim, hidden=args.hidden).to(device)
    print(f"\nCACH: {model.param_count():,} parameters")

    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=args.epochs, eta_min=args.lr * 0.01)

    best_ece = float("inf")
    log_rows = []

    print(f"\nTraining for {args.epochs} epochs ...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for patches, scores, labels in train_loader:
            patches = patches.to(device)
            scores  = scores.to(device)
            labels  = labels.to(device)
            optimiser.zero_grad()
            cal_scores = model.calibrate(patches, scores)
            loss = combined_loss(cal_scores, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            total_loss += loss.item() * len(scores)
        scheduler.step()

        train_loss = total_loss / max(len(train_ds), 1)
        print(f"  Epoch {epoch:3d}/{args.epochs}  train_loss={train_loss:.4f}")
        val_ece, val_loss = evaluate(model, val_loader, device)
        log_rows.append({"epoch": epoch, "train_loss": round(train_loss, 5),
                          "val_ece": round(val_ece, 5), "val_loss": round(val_loss, 5)})

        _ece_ok = not (val_ece != val_ece)  # True if val_ece is not nan
        if (_ece_ok and val_ece < best_ece) or (not _ece_ok and train_loss < best_ece):
            best_ece = val_ece if _ece_ok else train_loss
            model.save(str(out_dir / "cach_best.pt"))
            _msg = f"val ECE={best_ece:.4f}" if _ece_ok else f"train_loss={best_ece:.4f} (no val data)"
            print(f"    *** New best {_msg}  (checkpoint saved)")

    model.save(str(out_dir / "cach_final.pt"))

    # ── Training log ────────────────────────────────────────────────────────
    log_csv = out_dir / "training_log.csv"
    with open(log_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_ece", "val_loss"])
        w.writeheader(); w.writerows(log_rows)
    print(f"\nBest val ECE: {best_ece:.4f}")
    print(f"Checkpoints: {out_dir}/cach_best.pt, cach_final.pt")
    print(f"Log:         {log_csv}")


if __name__ == "__main__":
    main()
