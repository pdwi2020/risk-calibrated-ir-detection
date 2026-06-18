"""Training driver — YOLOv11m on FLIR ADAS v2 (SOTA YOLO comparison).

Same pipeline as 01_train_yolo.py; only the model checkpoint differs.
YOLOv11m uses the same ultralytics YOLO class as YOLOv8m — the checkpoint
name yolo11m.pt (no 'v') distinguishes it at the API level.

Usage (from project root):
    python scripts/21_train_yolov11.py \\
        --flir-root /workspace/data/flir/FLIR_ADAS_v2 \\
        --data-yaml configs/flir_yolo.yaml \\
        --ckpt-dir  /workspace/checkpoints \\
        --epochs 100 --batch 16 --seed 0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from scripts._torch_train_utils import set_seed
from src.detectors.yolov11_wrapper import YOLOv11Detector


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--flir-root", default="/workspace/data/flir/FLIR_ADAS_v2")
    p.add_argument("--data-yaml", default="configs/flir_yolo.yaml")
    p.add_argument("--ckpt-dir", default="/workspace/checkpoints")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = _parse()
    set_seed(args.seed)
    print(f"[21_train_yolov11] seed={args.seed}  epochs={args.epochs}  batch={args.batch}")

    det = YOLOv11Detector(model_path="yolo11m.pt", config_path="configs/yolov11.yaml")
    results = det.train(
        data_yaml=args.data_yaml,
        imgsz=640,
        batch=args.batch,
        epochs=args.epochs,
        seed=args.seed,
        workers=args.workers,
        project=args.ckpt_dir,
        name="yolov11m_flir",
        exist_ok=True,
    )
    print("[21_train_yolov11] training complete.")
    for k, v in results.items():
        print(f"  {k}: {v}")

    det.save(str(Path(args.ckpt_dir) / "yolov11m_flir_final.pt"))
    print(f"[21_train_yolov11] saved to {args.ckpt_dir}/yolov11m_flir_final.pt")


if __name__ == "__main__":
    main()
