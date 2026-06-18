"""20b_train_cach_balanced.py — CACH retrain with balanced TP/FP sampling + all 4 detectors.

Key differences from 20_train_cach.py:
  1. Multi-model: pools predictions from all 4 detectors (yolov8m, rtdetr,
     faster_rcnn, retinanet) → richer, more balanced training signal.
  2. Balanced sampler: WeightedRandomSampler enforces ~50/50 TP/FP in each
     minibatch, preventing the positive-confidence bias (b≈0.54) seen when
     TP-heavy data from sparse-TP detectors (RT-DETR, RetinaNet) dominates.
  3. 100 epochs with cosine LR schedule.

Usage (GPU node, from project root):
    python scripts/20b_train_cach_balanced.py \
        --flir-root /workspace/data/flir/FLIR_ADAS_v2 \
        --preds-dir results/corruption_preds \
        --split-json configs/flir_val_split.json \
        --out results/cach_v2 \
        --epochs 120 --batch 256 --lr 1e-3
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from src.calibration.cach import (
    CACH, combined_loss, match_detections_to_gt, preprocess_patch,
)
from src.corruption.corruption_pipeline import CORRUPTION_REGISTRY

DETECTORS = ["yolov8m", "rtdetr", "faster_rcnn", "retinanet"]
CLEAN_MAP = {1: 0, 2: 1, 3: 2}
VALID_CATS = set(CLEAN_MAP.keys())


# ---------------------------------------------------------------------------
# ECE helper
# ---------------------------------------------------------------------------

def ece(scores: np.ndarray, is_tp: np.ndarray, n_bins: int = 15) -> float:
    if len(scores) == 0:
        return 0.0
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (scores > lo) & (scores <= hi)
        if not mask.any():
            continue
        e += abs(is_tp[mask].mean() - scores[mask].mean()) * mask.mean()
    return float(e)


# ---------------------------------------------------------------------------
# Multi-model balanced dataset
# ---------------------------------------------------------------------------

class BalancedCACHDataset(Dataset):
    """Pools predictions from multiple detectors; records TP/FP label for sampler."""

    def __init__(
        self,
        flir_root: Path,
        preds_dir: Path,
        split_ids: List,
        model_names: List[str] = None,
        patch_size: int = 64,
        seed: int = 42,
    ):
        self.patch_size = patch_size
        # Each item: (img_path, cfn, sev, score, is_tp)
        self.items: List[Tuple] = []
        self.is_tp_flags: List[float] = []   # parallel, for sampler weights

        model_names = model_names or DETECTORS
        test_id_set = set(split_ids)

        coco_ann = flir_root / "images_thermal_val/coco.json"
        gt_data  = json.loads(coco_ann.read_text())
        id2file  = {img["id"]: img["file_name"] for img in gt_data["images"]}
        base     = flir_root / "images_thermal_val"

        from collections import defaultdict
        ann_map: Dict[int, list] = defaultdict(list)
        for ann in gt_data["annotations"]:
            if ann["image_id"] in test_id_set and ann["category_id"] in VALID_CATS:
                ann_map[ann["image_id"]].append(ann)

        corruption_list = [(None, 0)]
        for cname in CORRUPTION_REGISTRY:
            for sev in [1, 2, 3, 4]:
                corruption_list.append((cname, sev))

        for model_name in model_names:
            for corr_name, sev in corruption_list:
                if corr_name is None:
                    pred_file = preds_dir / f"{model_name}_clean.json"
                    cfn = None
                else:
                    pred_file = preds_dir / f"{model_name}_{corr_name}_{sev}.json"
                    cfn = CORRUPTION_REGISTRY.get(corr_name)
                if not pred_file.exists():
                    continue
                records = json.loads(pred_file.read_text())
                for rec in records:
                    img_id = rec.get("image_id")
                    if img_id not in test_id_set:
                        continue
                    fpath = base / id2file.get(img_id, "")
                    if not fpath.exists():
                        continue
                    if not rec.get("pred_scores"):
                        continue
                    _, is_tp_arr = match_detections_to_gt(
                        rec["pred_boxes"], rec["pred_scores"], rec["pred_labels"],
                        rec["gt_boxes"],  rec["gt_labels"])
                    for score, tp in zip(rec["pred_scores"], is_tp_arr):
                        self.items.append((str(fpath), cfn, sev, float(score), float(tp)))
                        self.is_tp_flags.append(float(tp))

        rng = random.Random(seed)
        order = list(range(len(self.items)))
        rng.shuffle(order)
        self.items = [self.items[i] for i in order]
        self.is_tp_flags = [self.is_tp_flags[i] for i in order]

        n_tp = sum(self.is_tp_flags)
        n_fp = len(self.is_tp_flags) - n_tp
        print(f"  Dataset: {len(self.items):,} dets "
              f"(TP={int(n_tp):,}  FP={int(n_fp):,}  "
              f"TP-ratio={n_tp/max(len(self.items),1):.3f}) "
              f"across {len(model_names)} detectors × {len(corruption_list)} conditions")

    def __len__(self) -> int:
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
        return (patch.squeeze(0),
                torch.tensor(score, dtype=torch.float32),
                torch.tensor(is_tp, dtype=torch.float32))

    def get_sample_weights(self) -> List[float]:
        """Return per-sample weights for WeightedRandomSampler (50/50 TP/FP)."""
        n_tp = max(sum(self.is_tp_flags), 1)
        n_fp = max(len(self.is_tp_flags) - n_tp, 1)
        w_tp = 1.0 / n_tp
        w_fp = 1.0 / n_fp
        return [w_tp if tp > 0.5 else w_fp for tp in self.is_tp_flags]


def collate_fn(batch):
    patches, scores, labels = zip(*batch)
    return torch.stack(patches), torch.stack(scores), torch.stack(labels)


# ---------------------------------------------------------------------------
# Split helpers
# ---------------------------------------------------------------------------

def load_split(split_json: Path) -> List:
    return json.loads(split_json.read_text())["test"]


def load_calib_split(split_json: Path) -> List:
    data = json.loads(split_json.read_text())
    return data.get("calibration", data.get("calib", []))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model: CACH, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
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
    if not all_cal:
        return float("nan"), 0.0
    cal_arr = np.concatenate(all_cal)
    raw_arr = np.concatenate(all_raw)
    tp_arr  = np.concatenate(all_tp)
    val_ece  = ece(cal_arr, tp_arr)
    raw_ece  = ece(raw_arr, tp_arr)
    avg_loss = total_loss / max(len(cal_arr), 1)
    print(f"    val_ece={val_ece:.4f}  raw_ece={raw_ece:.4f}  val_loss={avg_loss:.4f}")
    return val_ece, avg_loss


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flir-root",   required=True)
    ap.add_argument("--preds-dir",   required=True)
    ap.add_argument("--split-json",  default="configs/flir_val_split.json")
    ap.add_argument("--out",         default="results/cach_v2")
    ap.add_argument("--detectors",   nargs="+", default=DETECTORS)
    ap.add_argument("--epochs",      type=int,   default=120)
    ap.add_argument("--batch",       type=int,   default=256)
    ap.add_argument("--lr",          type=float, default=1e-3)
    ap.add_argument("--embed-dim",   type=int,   default=32)
    ap.add_argument("--hidden",      type=int,   default=64)
    ap.add_argument("--patch-size",  type=int,   default=64)
    ap.add_argument("--num-workers", type=int,   default=4)
    ap.add_argument("--seed",        type=int,   default=42)
    ap.add_argument("--no-balance",  action="store_true",
                    help="Disable balanced TP/FP sampler (use for ablation)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Detectors: {args.detectors}")
    print(f"Balanced sampler: {not args.no_balance}")

    flir_root = Path(args.flir_root)
    preds_dir = REPO / args.preds_dir
    out_dir   = REPO / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Build datasets ──────────────────────────────────────────────────────
    print("\nBuilding training dataset ...")
    train_ids = load_split(REPO / args.split_json)
    train_ds = BalancedCACHDataset(
        flir_root, preds_dir, train_ids,
        model_names=args.detectors,
        patch_size=args.patch_size, seed=args.seed)

    print("\nBuilding validation dataset ...")
    val_ids = load_calib_split(REPO / args.split_json)
    val_ds = BalancedCACHDataset(
        flir_root, preds_dir, val_ids,
        model_names=args.detectors,
        patch_size=args.patch_size, seed=args.seed + 1)

    # Balanced sampler for training (not validation)
    if not args.no_balance:
        weights = train_ds.get_sample_weights()
        sampler = WeightedRandomSampler(
            weights=weights, num_samples=len(weights), replacement=True)
        train_loader = DataLoader(
            train_ds, batch_size=args.batch, sampler=sampler,
            num_workers=args.num_workers, collate_fn=collate_fn,
            pin_memory=device.type == "cuda")
    else:
        train_loader = DataLoader(
            train_ds, batch_size=args.batch, shuffle=True,
            num_workers=args.num_workers, collate_fn=collate_fn,
            pin_memory=device.type == "cuda")

    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False,
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

    print(f"\nTraining for {args.epochs} epochs (balanced={not args.no_balance}) ...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n_items = 0
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
            n_items += len(scores)
        scheduler.step()

        train_loss = total_loss / max(n_items, 1)
        print(f"  Epoch {epoch:3d}/{args.epochs}  train_loss={train_loss:.4f}")
        val_ece, val_loss = evaluate(model, val_loader, device)

        log_rows.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "val_ece": round(val_ece, 5) if val_ece == val_ece else "nan",
            "val_loss": round(val_loss, 5),
        })

        is_ok = val_ece == val_ece   # not nan
        metric = val_ece if is_ok else train_loss
        if metric < best_ece:
            best_ece = metric
            model.save(str(out_dir / "cach_best.pt"))
            tag = f"val ECE={best_ece:.4f}" if is_ok else f"train_loss={best_ece:.4f}"
            print(f"    *** New best {tag}  (saved)")

    model.save(str(out_dir / "cach_final.pt"))

    log_csv = out_dir / "training_log.csv"
    with open(log_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_ece", "val_loss"])
        w.writeheader()
        w.writerows(log_rows)

    print(f"\nBest val ECE: {best_ece:.4f}")
    print(f"Checkpoints : {out_dir}/cach_best.pt  cach_final.pt")
    print(f"Log         : {log_csv}")


if __name__ == "__main__":
    main()
