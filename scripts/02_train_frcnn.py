"""Phase-1 Training Driver — Faster R-CNN (ResNet-50 FPN) on FLIR ADAS v2.

Architecture  : torchvision fasterrcnn_resnet50_fpn, pretrained on COCO.
Optimiser     : SGD (lr=0.005, momentum=0.9, weight_decay=5e-4).
Schedule      : constant LR (no scheduler).
Epochs        : 50   (override with --epochs).
Batch         : 4    (override with --batch).
Eval every    : 5    (override with --eval-every).

Data pipeline (leakage-free):
  Train : FLIRDataset(split='train', return_format='coco')
  Eval  : FLIRDataset(split='val',   return_format='coco')
          restricted to the 'test' partition of configs/flir_val_split.json
          via dataset_subset() — the calibration partition is NEVER used
          for reporting headline metrics.

Collate       : lambda batch: tuple(zip(*batch))
  => yields (tuple_of_image_tensors, tuple_of_target_dicts) per batch,
     exactly as required by FasterRCNNDetector.train_one_epoch / .evaluate.
  Images are float32 [C, H, W] in [0, 1]; torchvision normalises internally.

Label mapping (handled inside FasterRCNNDetector):
  FLIRDataset labels {0,1,2} (person/bike/car) are shifted to {1,2,3}
  inside train_one_epoch / evaluate to satisfy torchvision's convention
  (0 is reserved for background).

Outputs written to --ckpt-dir:
  frcnn_metrics.csv          — epoch, train_loss, mAP50, mAP50-95
  frcnn_epoch{N}.pth         — state_dict every 10 epochs
  frcnn_best.pth             — state_dict at highest mAP50

Usage (from project root):
    python scripts/02_train_frcnn.py \\
        --flir-root /path/to/FLIR_ADAS_v2 \\
        --split-json configs/flir_val_split.json \\
        --ckpt-dir ./checkpoints \\
        --epochs 50 --batch 4 --eval-every 5 --seed 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Repo-root path shim — resolves `from src.xxx` when invoked as
# `python scripts/02_train_frcnn.py` from the project root.
# ---------------------------------------------------------------------------
_SCRIPTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch  # noqa: E402

from scripts._torch_train_utils import (  # noqa: E402
    build_loaders,
    run_torch_training,
    set_seed,
)
from src.detectors.faster_rcnn_wrapper import FasterRCNNDetector  # noqa: E402


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train Faster R-CNN (ResNet-50 FPN) on FLIR ADAS v2.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--flir-root",
        default=(
            "/Volumes/Crucial X9/Research Projects/MV_Paper"
            "/datasets/flir_adas_v2/FLIR_ADAS_v2"
        ),
        help="Path to FLIR_ADAS_v2 root directory.",
    )
    p.add_argument(
        "--data-yaml",
        default="configs/flir_yolo.yaml",
        help="YOLO data.yaml (not used directly by FRCNN; kept for interface parity).",
    )
    p.add_argument(
        "--split-json",
        default="configs/flir_val_split.json",
        help="Frozen calibration/test partition JSON (must exist before training).",
    )
    p.add_argument(
        "--ckpt-dir",
        default="./checkpoints",
        help="Directory for checkpoints and the metrics CSV.",
    )
    p.add_argument("--epochs", type=int, default=50, help="Total training epochs.")
    p.add_argument("--batch", type=int, default=4, help="Images per training batch.")
    p.add_argument(
        "--workers", type=int, default=4, help="DataLoader worker processes."
    )
    p.add_argument("--seed", type=int, default=0, help="Global RNG seed.")
    p.add_argument(
        "--eval-every",
        type=int,
        default=5,
        help="Run detector.evaluate() every N epochs.",
    )
    # Optimiser hyperparameters (exposed as flags for ablations).
    p.add_argument("--lr", type=float, default=0.005, help="SGD base learning rate.")
    p.add_argument("--momentum", type=float, default=0.9, help="SGD momentum.")
    p.add_argument(
        "--weight-decay", type=float, default=5e-4, help="SGD L2 weight decay."
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    # 1. Deterministic seeding.
    set_seed(args.seed)

    print("[02_train_frcnn] configuration:")
    print(f"  flir-root   : {args.flir_root}")
    print(f"  split-json  : {args.split_json}")
    print(f"  ckpt-dir    : {args.ckpt_dir}")
    print(
        f"  epochs={args.epochs}  batch={args.batch}"
        f"  workers={args.workers}  seed={args.seed}"
    )
    print(
        f"  SGD  lr={args.lr}  momentum={args.momentum}"
        f"  weight_decay={args.weight_decay}"
    )
    print(f"  eval-every={args.eval_every}")

    # 2. DataLoaders.
    print("\n[02_train_frcnn] building dataloaders ...")
    train_loader, test_loader = build_loaders(
        flir_root=args.flir_root,
        split_json=args.split_json,
        batch=args.batch,
        workers=args.workers,
    )
    print(
        f"  train batches={len(train_loader)}"
        f"  test batches={len(test_loader)}"
    )

    # 3. Model.
    #    n_classes=4 : background(0) + person(1) + bike(2) + car(3).
    #    The label shift {0,1,2} -> {1,2,3} is performed inside the wrapper.
    detector = FasterRCNNDetector(
        n_classes=4,
        config_path="configs/faster_rcnn.yaml",
        pretrained=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device: {device}")

    # 4. Optimiser — SGD as specified in the project plan.
    optimizer = torch.optim.SGD(
        detector.model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    # 5. Training loop (no LR scheduler for FRCNN; constant LR per plan).
    print("\n[02_train_frcnn] starting training ...")
    run_torch_training(
        detector=detector,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        epochs=args.epochs,
        ckpt_dir=args.ckpt_dir,
        model_name="frcnn",
        eval_every=args.eval_every,
        scheduler=None,
    )


if __name__ == "__main__":
    main()
