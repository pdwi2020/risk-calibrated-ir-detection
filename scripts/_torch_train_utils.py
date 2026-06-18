"""Shared training utilities for torchvision detection drivers (FRCNN, RetinaNet).

Provides:
    set_seed(seed)               -- Deterministic seeding across random/numpy/torch/cuda.
    build_loaders(...)           -- Build train + test DataLoaders per MV_Paper convention.
    run_torch_training(...)      -- Generic epoch loop with CSV logging and checkpointing.

Usage (from project root):
    python scripts/02_train_frcnn.py  --flir-root <path> ...
    python scripts/03_train_retina_rtdetr.py --model retinanet ...
"""

from __future__ import annotations

import csv
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Repo-root sys.path shim — ensures `from src.xxx import ...` resolves when
# the scripts are run as `python scripts/02_train_frcnn.py` from the repo root
# or from any directory.
# ---------------------------------------------------------------------------
_SCRIPTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# Public: seeding
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Seed random, numpy, torch (CPU + all CUDA devices), and enable determinism.

    Sets ``torch.backends.cudnn.deterministic = True`` and disables the
    non-deterministic benchmark selection.  Slightly slower, but ensures
    experiment reproducibility across runs with the same seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Public: DataLoader factory
# ---------------------------------------------------------------------------

def build_loaders(
    flir_root: str,
    split_json: str,
    batch: int,
    workers: int,
) -> tuple[DataLoader, DataLoader]:
    """Build train and test DataLoaders for the FLIR ADAS v2 dataset.

    Train loader:
        FLIRDataset(split='train', return_format='coco') — full training split,
        no subset restriction, shuffled.

    Test loader:
        FLIRDataset(split='val', return_format='coco') restricted to the
        'test' partition of split_json via dataset_subset().  This guarantees
        that the calibration half of the FLIR val set is never seen during
        training-time evaluation, eliminating data leakage.

    Both loaders use:
        collate_fn = lambda batch: tuple(zip(*batch))
    which returns (tuple_of_image_tensors, tuple_of_target_dicts) — the exact
    input format expected by torchvision's train_one_epoch / evaluate.

    Images arrive from FLIRDataset as float32 [C, H, W] in [0, 1].  Torchvision
    detection models (Faster R-CNN, RetinaNet) apply their own internal
    normalisation; DO NOT pre-normalize here.

    Args:
        flir_root:  Path to FLIR_ADAS_v2 root directory.
        split_json: Path to configs/flir_val_split.json (frozen partition file).
        batch:      Images per training batch.
        workers:    DataLoader num_workers.

    Returns:
        (train_loader, test_loader)
    """
    from src.data.flir_dataset import FLIRDataset
    from src.data.splits import dataset_subset

    # Detection collate: keeps images/targets as lists/tuples rather than
    # stacking into a single tensor (bounding-box counts differ per image).
    _collate = lambda b: tuple(zip(*b))  # noqa: E731

    # Training set — full 'train' split, shuffle for SGD stochasticity.
    train_ds = FLIRDataset(root=flir_root, split="train", return_format="coco")
    train_loader = DataLoader(
        train_ds,
        batch_size=batch,
        shuffle=True,
        num_workers=workers,
        collate_fn=_collate,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    # Evaluation set — val split restricted to the 'test' partition only.
    val_ds = FLIRDataset(root=flir_root, split="val", return_format="coco")
    test_ds = dataset_subset(val_ds, split_json, "test")
    test_loader = DataLoader(
        test_ds,
        batch_size=batch,
        shuffle=False,
        num_workers=workers,
        collate_fn=_collate,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    return train_loader, test_loader


# ---------------------------------------------------------------------------
# Public: generic training loop
# ---------------------------------------------------------------------------

def run_torch_training(
    detector: Any,
    train_loader: DataLoader,
    test_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    ckpt_dir: str | Path,
    model_name: str,
    eval_every: int,
    scheduler: Optional[Any] = None,
) -> None:
    """Run the standard epoch loop for torchvision detection models.

    Per-epoch behaviour:
        1. Call detector.train_one_epoch(train_loader, optimizer) → train_loss.
        2. Step the optional LR scheduler (CosineAnnealingLR etc.) after each epoch.
        3. Every ``eval_every`` epochs (and on the final epoch): call
           detector.evaluate(test_loader) → {'mAP50', 'mAP50-95'}.
        4. Append a CSV row:  epoch, train_loss, mAP50, mAP50-95.
        5. Save a state_dict checkpoint every 10 epochs.
        6. Save ``{model_name}_best.pth`` whenever mAP50 improves.

    Args:
        detector:    A FasterRCNNDetector or RetinaNetDetector instance.
        train_loader: Training DataLoader (from build_loaders).
        test_loader:  Test DataLoader (from build_loaders).
        optimizer:   Configured torch optimizer.
        epochs:      Total number of training epochs.
        ckpt_dir:    Directory to write checkpoints and the metrics CSV.
        model_name:  Prefix used for all output file names.
        eval_every:  Run evaluate() every this many epochs.
        scheduler:   Optional LR scheduler (step called after each epoch).
    """
    ckpt_path = Path(ckpt_dir)
    ckpt_path.mkdir(parents=True, exist_ok=True)

    csv_file = ckpt_path / f"{model_name}_metrics.csv"
    fieldnames = ["epoch", "train_loss", "mAP50", "mAP50-95"]

    # Open CSV once; append mode so reruns from a checkpoint don't overwrite history.
    csv_existed = csv_file.exists()
    csv_fh = open(csv_file, "a", newline="")
    writer = csv.DictWriter(csv_fh, fieldnames=fieldnames)
    if not csv_existed:
        writer.writeheader()

    best_map50 = 0.0

    for epoch in range(1, epochs + 1):
        # ------------------------------------------------------------------ #
        # 1. Training step
        # ------------------------------------------------------------------ #
        train_loss = detector.train_one_epoch(train_loader, optimizer)

        # ------------------------------------------------------------------ #
        # 2. LR scheduler step
        # ------------------------------------------------------------------ #
        if scheduler is not None:
            scheduler.step()

        # ------------------------------------------------------------------ #
        # 3. Evaluation (every eval_every epochs and on the last epoch)
        # ------------------------------------------------------------------ #
        map50: float = float("nan")
        map50_95: float = float("nan")
        if (epoch % eval_every == 0) or (epoch == epochs):
            metrics: Dict[str, float] = detector.evaluate(test_loader)
            map50 = metrics.get("mAP50", float("nan"))
            map50_95 = metrics.get("mAP50-95", float("nan"))

        # ------------------------------------------------------------------ #
        # 4. CSV logging
        # ------------------------------------------------------------------ #
        writer.writerow(
            {
                "epoch": epoch,
                "train_loss": f"{train_loss:.6f}",
                "mAP50": f"{map50:.6f}" if not _is_nan(map50) else "",
                "mAP50-95": f"{map50_95:.6f}" if not _is_nan(map50_95) else "",
            }
        )
        csv_fh.flush()

        # ------------------------------------------------------------------ #
        # 5. Progress logging
        # ------------------------------------------------------------------ #
        map_str = (
            f"  mAP50={map50:.4f}  mAP50-95={map50_95:.4f}"
            if not _is_nan(map50)
            else ""
        )
        print(
            f"[{model_name}] epoch {epoch:>3d}/{epochs}"
            f"  loss={train_loss:.4f}{map_str}"
        )

        # ------------------------------------------------------------------ #
        # 6. Periodic checkpoint every 10 epochs
        # ------------------------------------------------------------------ #
        if epoch % 10 == 0:
            ckpt_file = ckpt_path / f"{model_name}_epoch{epoch}.pth"
            torch.save(detector.model.state_dict(), ckpt_file)
            print(f"  -> saved checkpoint: {ckpt_file}")

        # ------------------------------------------------------------------ #
        # 7. Best model by mAP50
        # ------------------------------------------------------------------ #
        if not _is_nan(map50) and map50 > best_map50:
            best_map50 = map50
            best_file = ckpt_path / f"{model_name}_best.pth"
            torch.save(detector.model.state_dict(), best_file)
            print(f"  -> new best mAP50={best_map50:.4f}  saved: {best_file}")

    csv_fh.close()
    print(f"\n[{model_name}] training complete.  best mAP50={best_map50:.4f}")
    print(f"  metrics log : {csv_file}")
    print(f"  checkpoints : {ckpt_path}")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_nan(v: float) -> bool:
    """Return True when v is NaN (handles float('nan') without math import)."""
    return v != v
