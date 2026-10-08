"""Phase-1 Training Driver — YOLOv8m on FLIR ADAS v2 thermal images.

Delegates training entirely to ultralytics; the wrapper handles evaluation
internally (stores results under project/name/).  This script:
  1. Seeds the RNG deterministically.
  2. Builds a YOLOv8Detector from the pretrained YOLOv8m weights.
  3. Calls .train() with the configured hyperparameters.
  4. Calls .evaluate() and prints a summary.

Usage (from project root):
    python scripts/01_train_yolo.py \\
        --flir-root /path/to/FLIR_ADAS_v2 \\
        --data-yaml configs/flir_yolo.yaml \\
        --ckpt-dir ./checkpoints \\
        --epochs 100 \\
        --batch  16 \\
        --seed   0

Hyperparameters (from project plan):
    imgsz=640, batch=16, epochs=100
    Ultralytics manages the optimiser, LR schedule, and mAP evaluation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Repo-root path shim — allows `python scripts/01_train_yolo.py` from root.
# ---------------------------------------------------------------------------
_SCRIPTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts._torch_train_utils import set_seed  # noqa: E402
from src.detectors.yolov8_wrapper import YOLOv8Detector  # noqa: E402


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train YOLOv8m on FLIR ADAS v2 thermal images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--flir-root",
        default="datasets/flir_adas_v2/FLIR_ADAS_v2",
        help="Path to FLIR_ADAS_v2 root directory.",
    )
    p.add_argument(
        "--data-yaml",
        default="configs/flir_yolo.yaml",
        help="YOLO data.yaml specifying train/val paths and class names.",
    )
    p.add_argument(
        "--split-json",
        default="configs/flir_val_split.json",
        help="Frozen val/test split JSON (used only for reference; YOLO uses data-yaml).",
    )
    p.add_argument(
        "--ckpt-dir",
        default="./checkpoints",
        help="Directory for ultralytics run outputs.",
    )
    p.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    p.add_argument("--batch", type=int, default=16, help="Batch size.")
    p.add_argument(
        "--workers", type=int, default=4, help="DataLoader worker processes."
    )
    p.add_argument("--seed", type=int, default=0, help="Global RNG seed.")
    p.add_argument(
        "--eval-every",
        type=int,
        default=10,
        help="Evaluate every N epochs (informational; ultralytics runs val internally).",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    # 1. Deterministic seeding.
    set_seed(args.seed)
    print(f"[01_train_yolo] seed={args.seed}")
    print(f"  data-yaml  : {args.data_yaml}")
    print(f"  ckpt-dir   : {args.ckpt_dir}")
    print(f"  epochs={args.epochs}  batch={args.batch}  workers={args.workers}")

    # 2. Build detector.
    detector = YOLOv8Detector(
        model_path="yolov8m.pt",
        config_path="configs/yolov8.yaml",
    )

    # 3. Train — ultralytics manages optimiser, LR schedule, val mAP, and
    #    saves best/last checkpoints under project/name/.
    print("\n[01_train_yolo] starting ultralytics training run ...")
    results = detector.train(
        data_yaml=args.data_yaml,
        imgsz=640,
        batch=args.batch,
        epochs=args.epochs,
        seed=args.seed,
        workers=args.workers,
        project=args.ckpt_dir,
        name="yolov8m_flir",
        exist_ok=True,
    )

    # 4. Report key metrics returned by ultralytics.
    print("\n[01_train_yolo] training complete.")
    for key, val in results.items():
        print(f"  {key}: {val}")

    # 5. Standalone evaluation pass on the val split.
    print("\n[01_train_yolo] running standalone evaluation ...")
    eval_metrics = detector.evaluate(data_yaml=args.data_yaml)
    print("[01_train_yolo] evaluation results:")
    for key, val in eval_metrics.items():
        print(f"  {key}: {val}")

    detector.save(str(Path(args.ckpt_dir) / "yolov8m_flir_final.pt"))
    print(f"\n[01_train_yolo] model saved to {args.ckpt_dir}/yolov8m_flir_final.pt")


if __name__ == "__main__":
    main()
