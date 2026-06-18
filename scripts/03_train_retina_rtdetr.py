"""Phase-1 Training Driver — RetinaNet or RT-DETR on FLIR ADAS v2.

Select the model with --model {retinanet,rtdetr}  (default: retinanet).

-----------------------------------------------------------------------
RetinaNet path (default)
-----------------------------------------------------------------------
Architecture  : torchvision retinanet_resnet50_fpn, pretrained on COCO.
                Focal loss (alpha=0.25, gamma=2.0) for class imbalance.
Optimiser     : SGD (lr=0.01, momentum=0.9, weight_decay=1e-4).
Schedule      : CosineAnnealingLR(T_max=epochs).
Epochs        : 50   (override with --epochs).
Batch         : 4    (override with --batch).
Eval every    : 5    (override with --eval-every).

Data pipeline (leakage-free):
  Train : FLIRDataset(split='train', return_format='coco')
  Eval  : FLIRDataset(split='val',   return_format='coco')
          restricted to the 'test' partition of configs/flir_val_split.json
          via dataset_subset() — calibration partition never touches eval.

Collate       : lambda batch: tuple(zip(*batch))
  => yields (tuple_of_image_tensors, tuple_of_target_dicts) per batch.
  Images are float32 [C, H, W] in [0, 1]; torchvision normalises internally.

Label mapping (handled inside RetinaNetDetector):
  FLIRDataset labels {0,1,2} shifted to {1,2,3} inside train_one_epoch /
  evaluate (0 is reserved for background in torchvision).

Outputs:
  retinanet_metrics.csv          — epoch, train_loss, mAP50, mAP50-95
  retinanet_epoch{N}.pth         — state_dict every 10 epochs
  retinanet_best.pth             — state_dict at highest mAP50

-----------------------------------------------------------------------
RT-DETR path (--model rtdetr)
-----------------------------------------------------------------------
Architecture  : RTDETRDetector('rtdetr-l.pt') via ultralytics.
Optimiser     : AdamW at lr=1e-4 (set inside RTDETRDetector.train).
Epochs        : 72   (override with --epochs).
Batch         : 8    (override with --batch).
Eval          : ultralytics handles eval internally; this script calls
                .evaluate() once after training for a headline summary.

Outputs written under --ckpt-dir/rtdetr_flir/  (ultralytics project dir).

-----------------------------------------------------------------------
Usage (from project root):
    # RetinaNet (default)
    python scripts/03_train_retina_rtdetr.py \\
        --model retinanet \\
        --flir-root /path/to/FLIR_ADAS_v2 \\
        --split-json configs/flir_val_split.json \\
        --ckpt-dir ./checkpoints \\
        --epochs 50 --batch 4 --eval-every 5 --seed 0

    # RT-DETR
    python scripts/03_train_retina_rtdetr.py \\
        --model rtdetr \\
        --data-yaml configs/flir_yolo.yaml \\
        --ckpt-dir ./checkpoints \\
        --epochs 72 --batch 8 --seed 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Repo-root path shim — resolves `from src.xxx` when invoked as
# `python scripts/03_train_retina_rtdetr.py` from the project root.
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


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train RetinaNet or RT-DETR on FLIR ADAS v2.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--model",
        choices=["retinanet", "rtdetr"],
        default="retinanet",
        help="Detector to train: 'retinanet' (torchvision) or 'rtdetr' (ultralytics).",
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
        help=(
            "YOLO data.yaml — required for RT-DETR; unused by RetinaNet "
            "(kept for interface parity)."
        ),
    )
    p.add_argument(
        "--split-json",
        default="configs/flir_val_split.json",
        help="Frozen calibration/test partition JSON (RetinaNet path only).",
    )
    p.add_argument(
        "--ckpt-dir",
        default="./checkpoints",
        help="Directory for checkpoints and the metrics CSV.",
    )
    # Epoch defaults differ per model; users override via flag.
    p.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Training epochs.  Defaults: retinanet=50, rtdetr=72.",
    )
    p.add_argument(
        "--batch",
        type=int,
        default=None,
        help="Images per training batch.  Defaults: retinanet=4, rtdetr=8.",
    )
    p.add_argument(
        "--workers", type=int, default=4, help="DataLoader worker processes."
    )
    p.add_argument("--seed", type=int, default=0, help="Global RNG seed.")
    p.add_argument(
        "--eval-every",
        type=int,
        default=5,
        help="Run evaluate() every N epochs (RetinaNet path only).",
    )
    # RetinaNet optimiser hyperparameters.
    p.add_argument("--lr", type=float, default=0.01, help="SGD learning rate (RetinaNet).")
    p.add_argument("--momentum", type=float, default=0.9, help="SGD momentum (RetinaNet).")
    p.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="SGD L2 weight decay (RetinaNet).",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# RetinaNet training path
# ---------------------------------------------------------------------------

def _train_retinanet(args: argparse.Namespace) -> None:
    from src.detectors.retinanet_wrapper import RetinaNetDetector

    epochs = args.epochs if args.epochs is not None else 50
    batch = args.batch if args.batch is not None else 4

    print("[03_train_retina_rtdetr] model=retinanet  configuration:")
    print(f"  flir-root   : {args.flir_root}")
    print(f"  split-json  : {args.split_json}")
    print(f"  ckpt-dir    : {args.ckpt_dir}")
    print(
        f"  epochs={epochs}  batch={batch}"
        f"  workers={args.workers}  seed={args.seed}"
    )
    print(
        f"  SGD  lr={args.lr}  momentum={args.momentum}"
        f"  weight_decay={args.weight_decay}"
    )
    print(f"  CosineAnnealingLR(T_max={epochs})")
    print(f"  eval-every={args.eval_every}")

    # DataLoaders.
    print("\n[03_train_retina_rtdetr] building dataloaders ...")
    train_loader, test_loader = build_loaders(
        flir_root=args.flir_root,
        split_json=args.split_json,
        batch=batch,
        workers=args.workers,
    )
    print(
        f"  train batches={len(train_loader)}"
        f"  test batches={len(test_loader)}"
    )

    # Model.
    #    n_classes=3 : public API (3 foreground classes; wrapper adds background internally).
    detector = RetinaNetDetector(n_classes=3, pretrained=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device: {device}")

    # Optimiser — SGD per project plan.
    optimizer = torch.optim.SGD(
        detector.model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    # CosineAnnealingLR schedule — decays to near-zero over T_max epochs.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )

    # Training loop.
    print("\n[03_train_retina_rtdetr] starting RetinaNet training ...")
    run_torch_training(
        detector=detector,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        epochs=epochs,
        ckpt_dir=args.ckpt_dir,
        model_name="retinanet",
        eval_every=args.eval_every,
        scheduler=scheduler,
    )


# ---------------------------------------------------------------------------
# RT-DETR training path
# ---------------------------------------------------------------------------

def _train_rtdetr(args: argparse.Namespace) -> None:
    from src.detectors.rtdetr_wrapper import RTDETRDetector

    epochs = args.epochs if args.epochs is not None else 72
    batch = args.batch if args.batch is not None else 8

    print("[03_train_retina_rtdetr] model=rtdetr  configuration:")
    print(f"  data-yaml   : {args.data_yaml}")
    print(f"  ckpt-dir    : {args.ckpt_dir}")
    print(
        f"  epochs={epochs}  batch={batch}"
        f"  workers={args.workers}  seed={args.seed}"
    )
    print("  optimiser: AdamW lr=1e-4 (set inside RTDETRDetector.train)")

    # Build detector from pretrained RT-DETR-L weights.
    detector = RTDETRDetector(
        model_path="rtdetr-l.pt",
        config_path="configs/yolov8.yaml",
    )

    # Train — ultralytics manages optimiser (AdamW), LR schedule, and val mAP.
    print("\n[03_train_retina_rtdetr] starting RT-DETR training ...")
    results = detector.train(
        data_yaml=args.data_yaml,
        imgsz=640,
        batch=batch,
        epochs=epochs,
        seed=args.seed,
        workers=args.workers,
        project=args.ckpt_dir,
        name="rtdetr_flir",
        exist_ok=True,
    )

    print("\n[03_train_retina_rtdetr] RT-DETR training complete.")
    for key, val in results.items():
        print(f"  {key}: {val}")

    # Standalone evaluation pass.
    print("\n[03_train_retina_rtdetr] running standalone evaluation ...")
    eval_metrics = detector.evaluate(data_yaml=args.data_yaml)
    print("[03_train_retina_rtdetr] evaluation results:")
    for key, val in eval_metrics.items():
        print(f"  {key}: {val}")

    detector.save(str(Path(args.ckpt_dir) / "rtdetr_flir_final.pt"))
    print(f"\n[03_train_retina_rtdetr] model saved to {args.ckpt_dir}/rtdetr_flir_final.pt")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    # Deterministic seeding — applies for both model paths.
    set_seed(args.seed)

    if args.model == "retinanet":
        _train_retinanet(args)
    else:
        _train_rtdetr(args)


if __name__ == "__main__":
    main()
