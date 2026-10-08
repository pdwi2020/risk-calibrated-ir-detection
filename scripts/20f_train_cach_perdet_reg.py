"""20e_train_cach_perdet.py — Per-detector CACH trainer, leakage-free + fast.

Derived from 20c_train_cach_efficient.py.  Changes vs 20c:
  1. Cache-file names use the calibration split:
       clean      → {model}_calibration_clean.json
       corrupted  → {model}_calibration_{corr}_{sev}.json
     (20c used {model}_clean.json / {model}_{corr}_{sev}.json — old no-split names)
  2. Training uses CALIBRATION split image-ids (leakage-free).
     Validation uses TEST split image-ids.
  3. --detectors accepts a SINGLE detector name so each model gets its own
     checkpoint; --out is required to be per-detector (caller passes
     --out results/cach_<model>).

Everything else (RAM patch-cache, BalancedBatchSampler, num_workers=0) is
unchanged from 20c.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Sampler, Subset

from src.calibration.cach import (
    CACH, combined_loss, identity_reg_loss, match_detections_to_gt,
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
# Balanced batch sampler (no torch.multinomial, exact 50/50)
# ---------------------------------------------------------------------------

class BalancedBatchSampler(Sampler):
    """Each batch contains exactly batch_size//2 TP and batch_size//2 FP items.

    Iterates all TP indices once per epoch; FP indices are shuffled and
    sampled to match.  Drops the last partial batch.
    """

    def __init__(self, is_tp_flags: List[float], batch_size: int, seed: int = 42):
        self.tp_idx = np.where(np.array(is_tp_flags) > 0.5)[0]
        self.fp_idx = np.where(np.array(is_tp_flags) <= 0.5)[0]
        self.batch_size = batch_size
        self.half = batch_size // 2
        self.rng = np.random.default_rng(seed)

    def __iter__(self) -> Iterator[List[int]]:
        half = self.half
        tp = self.tp_idx.copy(); self.rng.shuffle(tp)
        fp = self.fp_idx.copy(); self.rng.shuffle(fp)
        n_batches = min(len(tp), len(fp)) // half
        for i in range(n_batches):
            batch = np.concatenate([tp[i*half:(i+1)*half],
                                    fp[i*half:(i+1)*half]])
            self.rng.shuffle(batch)
            yield batch.tolist()

    def __len__(self) -> int:
        return min(len(self.tp_idx), len(self.fp_idx)) // self.half


# ---------------------------------------------------------------------------
# Dataset with pre-computed patch cache
# ---------------------------------------------------------------------------

class EfficientCACHDataset(Dataset):
    """Predictions from one (or more) detectors; pre-computes unique patches.

    The pre-computation step uses ThreadPoolExecutor (not fork) to avoid
    the fork-safety issue with cv2 and numpy thread pools.

    Key difference vs 20c:
      - split_ids are the *calibration* ids when building the training set
        (or *test* ids for the validation set).
      - Cache filenames use the {model}_{split}_{corr}_{sev}.json naming
        where split = "calibration" or "test".
    """

    def __init__(
        self,
        flir_root: Path,
        preds_dir: Path,
        split_ids: List,          # calibration image ids for train; test ids for val
        cache_split: str,         # "calibration" or "test" — selects file suffix
        model_names: List[str] = None,
        patch_size: int = 64,
        seed: int = 42,
        n_precompute_workers: int = 8,
    ):
        self.patch_size = patch_size
        model_names = model_names or DETECTORS
        id_set = set(split_ids)

        # ── Load GT annotations ─────────────────────────────────────────────
        coco_ann = flir_root / "images_thermal_val/coco.json"
        gt_data  = json.loads(coco_ann.read_text())
        id2file  = {img["id"]: img["file_name"] for img in gt_data["images"]}
        base     = flir_root / "images_thermal_val"

        from collections import defaultdict
        ann_map: Dict[int, list] = defaultdict(list)
        for ann in gt_data["annotations"]:
            if ann["image_id"] in id_set and ann["category_id"] in VALID_CATS:
                ann_map[ann["image_id"]].append(ann)

        corruption_list = [(None, 0)]
        for cname in CORRUPTION_REGISTRY:
            for sev in [1, 2, 3, 4]:
                corruption_list.append((cname, sev))

        # ── Pass 1: build detection records (no image I/O) ──────────────────
        raw_items: List[Tuple] = []
        is_tp_flags: List[float] = []
        unique_keys_ordered: List[Tuple] = []
        key_to_idx: Dict[Tuple, int] = {}

        for model_name in model_names:
            for corr_name, sev in corruption_list:
                # --- CHANGE 1: per-split cache filenames ---
                if corr_name is None:
                    pred_file = preds_dir / f"{model_name}_{cache_split}_clean.json"
                else:
                    pred_file = preds_dir / f"{model_name}_{cache_split}_{corr_name}_{sev}.json"
                if not pred_file.exists():
                    continue
                records = json.loads(pred_file.read_text())
                for rec in records:
                    img_id = rec.get("image_id")
                    # --- CHANGE 2: filter to the correct split ids ---
                    if img_id not in id_set:
                        continue
                    fpath = base / id2file.get(img_id, "")
                    if not fpath.exists():
                        continue
                    if not rec.get("pred_scores"):
                        continue
                    _, is_tp_arr = match_detections_to_gt(
                        rec["pred_boxes"], rec["pred_scores"], rec["pred_labels"],
                        rec["gt_boxes"],  rec["gt_labels"])

                    key = (str(fpath), corr_name, sev)
                    if key not in key_to_idx:
                        key_to_idx[key] = len(unique_keys_ordered)
                        unique_keys_ordered.append(key)

                    patch_idx = key_to_idx[key]
                    for score, tp in zip(rec["pred_scores"], is_tp_arr):
                        raw_items.append((patch_idx, float(score), float(tp)))
                        is_tp_flags.append(float(tp))

        # Shuffle items
        rng = random.Random(seed)
        order = list(range(len(raw_items)))
        rng.shuffle(order)
        raw_items   = [raw_items[i]   for i in order]
        is_tp_flags = [is_tp_flags[i] for i in order]

        n_tp = sum(is_tp_flags)
        n_fp = len(is_tp_flags) - n_tp
        n_unique = len(unique_keys_ordered)
        print(f"  [{cache_split}] Detections: {len(raw_items):,} "
              f"(TP={int(n_tp):,}  FP={int(n_fp):,}  "
              f"TP-ratio={n_tp/max(len(raw_items),1):.3f}) "
              f"across {len(model_names)} detectors × {len(corruption_list)} conditions")
        print(f"  Unique (image, condition) keys: {n_unique:,}")

        # ── Pass 2: pre-compute unique patches ──────────────────────────────
        print(f"  Pre-computing {n_unique:,} patches "
              f"(parallel, {n_precompute_workers} threads)...", flush=True)

        import cv2  # import here; before fork, but we use threads not fork

        def _compute_patch(key: Tuple) -> np.ndarray:
            img_path, corr_name, sev = key
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                img = np.zeros((512, 640), dtype=np.uint8)
            if corr_name is not None:
                fn = CORRUPTION_REGISTRY[corr_name]
                img = fn(img, sev)
            # Resize to patch_size × patch_size, normalise to [0,1]
            small = cv2.resize(img.astype(np.float32),
                               (patch_size, patch_size),
                               interpolation=cv2.INTER_LINEAR)
            return (small / 255.0).astype(np.float32)   # (64, 64)

        with ThreadPoolExecutor(max_workers=n_precompute_workers) as ex:
            patches_list = list(ex.map(_compute_patch, unique_keys_ordered))

        # patches: (n_unique, 64, 64) float32 — stays in RAM
        if patches_list:
            self.patches = np.stack(patches_list, axis=0)  # (n_unique, H, W)
        else:
            self.patches = np.zeros((0, patch_size, patch_size), dtype=np.float32)
        print(f"  Patch cache: {self.patches.nbytes / 1e6:.1f} MB", flush=True)

        # ── Store final items as compact arrays ──────────────────────────────
        n = len(raw_items)
        self._patch_idx = np.empty(n, dtype=np.int32)
        self._scores    = np.empty(n, dtype=np.float32)
        self._is_tp     = np.empty(n, dtype=np.float32)
        for i, (pidx, score, tp) in enumerate(raw_items):
            self._patch_idx[i] = pidx
            self._scores[i]    = score
            self._is_tp[i]     = tp

        self.is_tp_flags = is_tp_flags   # kept for BalancedBatchSampler

    def __len__(self) -> int:
        return len(self._scores)

    def __getitem__(self, idx: int):
        pidx  = self._patch_idx[idx]
        patch = torch.tensor(self.patches[pidx], dtype=torch.float32).unsqueeze(0)  # (1,H,W)
        score = torch.tensor(self._scores[idx], dtype=torch.float32)
        is_tp = torch.tensor(self._is_tp[idx],  dtype=torch.float32)
        return patch, score, is_tp


def collate_fn(batch):
    patches, scores, labels = zip(*batch)
    return torch.stack(patches), torch.stack(scores), torch.stack(labels)


# ---------------------------------------------------------------------------
# Split helpers
# ---------------------------------------------------------------------------

def load_splits(split_json: Path):
    """Return (calibration_ids, test_ids) from configs/flir_val_split.json."""
    data = json.loads(split_json.read_text())
    calib_ids = data.get("calibration", data.get("calib", []))
    test_ids  = data.get("test", [])
    return calib_ids, test_ids


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
    print(f"    val_ece={val_ece:.4f}  raw_ece={raw_ece:.4f}  val_loss={avg_loss:.4f}",
          flush=True)
    return val_ece, avg_loss


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Train a per-detector CACH model on the calibration split.")
    ap.add_argument("--flir-root",   required=True,
                    help="Path to FLIR_ADAS_v2 root (contains images_thermal_val/)")
    ap.add_argument("--preds-dir",   required=True,
                    help="Dir with {model}_{split}_*.json prediction caches")
    ap.add_argument("--split-json",  default="configs/flir_val_split.json",
                    help="Path to flir_val_split.json (relative to repo root or absolute)")
    ap.add_argument("--out",         default="results/cach_perdet",
                    help="Output directory for checkpoints + log")
    # CHANGE 3: --detectors kept; single-detector mode is the primary use-case
    ap.add_argument("--detectors",   nargs="+", default=DETECTORS,
                    help="Detector(s) to include (default: all four)")
    ap.add_argument("--epochs",      type=int,   default=30)
    ap.add_argument("--batch",       type=int,   default=256)
    ap.add_argument("--lr",          type=float, default=1e-3)
    ap.add_argument("--embed-dim",   type=int,   default=32)
    ap.add_argument("--hidden",      type=int,   default=64)
    ap.add_argument("--patch-size",  type=int,   default=64)
    ap.add_argument("--seed",        type=int,   default=42)
    ap.add_argument("--no-balance",  action="store_true")
    ap.add_argument("--lambda-id",   type=float, default=1.0,
                    help="Identity-anchoring regularization weight (T->1, b->0). "
                         "Higher => CACH stays closer to no-calibration.")
    ap.add_argument("--max-val",     type=int,   default=50000,
                    help="Subsample the validation set to this many detections for "
                         "per-epoch checkpoint selection (final reported ECE always "
                         "uses the full test set via 24_evaluate_cach.py). Keeps "
                         "per-epoch eval fast for high-detection detectors (rtdetr).")
    args = ap.parse_args()

    # Unbuffered output so nohup/tmux log is up-to-date
    sys.stdout.reconfigure(line_buffering=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    print(f"Detectors: {args.detectors}", flush=True)
    print(f"Balanced sampler: {not args.no_balance}", flush=True)

    flir_root = Path(args.flir_root)
    # preds_dir: support absolute paths or repo-relative
    preds_dir = Path(args.preds_dir)
    if not preds_dir.is_absolute():
        preds_dir = REPO / preds_dir
    # split_json: same
    split_json = Path(args.split_json)
    if not split_json.is_absolute():
        split_json = REPO / split_json
    out_dir   = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = REPO / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load splits ──────────────────────────────────────────────────────────
    calib_ids, test_ids = load_splits(split_json)
    print(f"Split: {len(calib_ids)} calibration ids, {len(test_ids)} test ids", flush=True)

    # ── Datasets ─────────────────────────────────────────────────────────────
    print("\nBuilding training dataset (calibration split) ...", flush=True)
    train_ds = EfficientCACHDataset(
        flir_root, preds_dir,
        split_ids=calib_ids,
        cache_split="calibration",
        model_names=args.detectors,
        patch_size=args.patch_size,
        seed=args.seed)

    print("\nBuilding validation dataset (test split) ...", flush=True)
    val_ds = EfficientCACHDataset(
        flir_root, preds_dir,
        split_ids=test_ids,
        cache_split="test",
        model_names=args.detectors,
        patch_size=args.patch_size,
        seed=args.seed + 1)

    # ── Samplers / loaders ────────────────────────────────────────────────────
    if not args.no_balance and len(train_ds) > 0:
        batch_sampler = BalancedBatchSampler(
            train_ds.is_tp_flags, batch_size=args.batch, seed=args.seed)
        train_loader = DataLoader(
            train_ds, batch_sampler=batch_sampler,
            num_workers=0, collate_fn=collate_fn)
        n_batches = len(batch_sampler)
    else:
        train_loader = DataLoader(
            train_ds, batch_size=args.batch, shuffle=True,
            num_workers=0, collate_fn=collate_fn)
        n_batches = len(train_loader)
        batch_sampler = None

    # Mini-val for per-epoch checkpoint selection: subsample to keep the
    # per-epoch eval cheap on high-detection detectors.  The FINAL Table XIII
    # ECE is always computed on the full test split by 24_evaluate_cach.py.
    val_eval_ds = val_ds
    if 0 < args.max_val < len(val_ds):
        rng = np.random.default_rng(args.seed)
        sub_idx = rng.choice(len(val_ds), size=args.max_val, replace=False)
        val_eval_ds = Subset(val_ds, sub_idx.tolist())
        print(f"Mini-val: {len(val_eval_ds):,} / {len(val_ds):,} detections "
              f"for checkpoint selection", flush=True)
    val_loader = DataLoader(
        val_eval_ds, batch_size=args.batch, shuffle=False,
        num_workers=0, collate_fn=collate_fn)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = CACH(embed_dim=args.embed_dim, hidden=args.hidden).to(device)
    print(f"\nCACH: {model.param_count():,} parameters", flush=True)

    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=args.epochs, eta_min=args.lr * 0.01)

    best_ece = float("inf")
    log_rows = []

    print(f"\nTraining for {args.epochs} epochs "
          f"(balanced={not args.no_balance}, "
          f"batches/epoch≈{n_batches}) ...",
          flush=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n_items = 0
        for patches, scores, labels in train_loader:
            patches = patches.to(device)
            scores  = scores.to(device)
            labels  = labels.to(device)
            optimiser.zero_grad()
            cal_scores, T, b = model.calibrate_with_params(patches, scores)
            loss = (combined_loss(cal_scores, labels)
                    + args.lambda_id * identity_reg_loss(T, b))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            total_loss += loss.item() * len(scores)
            n_items += len(scores)
        scheduler.step()

        train_loss = total_loss / max(n_items, 1)
        print(f"  Epoch {epoch:3d}/{args.epochs}  train_loss={train_loss:.4f}", flush=True)
        val_ece, val_loss = evaluate(model, val_loader, device)

        log_rows.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "val_ece": round(val_ece, 5) if val_ece == val_ece else "nan",
            "val_loss": round(val_loss, 5),
        })

        is_ok = val_ece == val_ece
        metric = val_ece if is_ok else train_loss
        if metric < best_ece:
            best_ece = metric
            model.save(str(out_dir / "cach_best.pt"))
            tag = f"val ECE={best_ece:.4f}" if is_ok else f"train_loss={best_ece:.4f}"
            print(f"    *** New best {tag}  (saved)", flush=True)

    model.save(str(out_dir / "cach_final.pt"))

    log_csv = out_dir / "training_log.csv"
    with open(log_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_ece", "val_loss"])
        w.writeheader()
        w.writerows(log_rows)

    print(f"\nBest val ECE: {best_ece:.4f}", flush=True)
    print(f"Checkpoints : {out_dir}/cach_best.pt  cach_final.pt", flush=True)
    print(f"Log         : {log_csv}", flush=True)


if __name__ == "__main__":
    main()
