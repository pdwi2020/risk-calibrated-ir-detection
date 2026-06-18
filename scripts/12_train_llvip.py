"""Train YOLOv8m on LLVIP infrared pedestrian dataset.

Used to produce the LLVIP-base checkpoint needed for Phase-5 domain transfer
directions D3 (LLVIP→FLIR zero-shot) and D4 (LLVIP→FLIR fine-tuned).

LLVIP has a single class (person=0):
    train: 12,025 IR images  (LLVIP-YOLO/train/lwir/images/)
    val:    3,463 IR images  (LLVIP-YOLO/test/lwir/images/)

Labels live in lwir/labels/ (symlinked from visible/labels/). Create links
before running:
    cd <LLVIP-YOLO>/train/lwir && ln -s ../visible/labels labels
    cd <LLVIP-YOLO>/test/lwir  && ln -s ../visible/labels labels

Usage (on GPU box):
    python scripts/12_train_llvip.py \\
        --llvip-root /workspace/data/llvip/LLVIP-YOLO \\
        --ckpt-dir   /workspace/results/yolov8m_llvip \\
        --epochs 50  --batch 16  --seed 0
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from scripts._torch_train_utils import set_seed


def _write_data_yaml(llvip_root: Path, out_path: Path) -> None:
    """Write a YOLO data.yaml for LLVIP (person only, lwir/ layout)."""
    content = f"""\
# LLVIP — YOLOv8 training (IR / lwir modality, auto-generated)
# Symlinks required:
#   LLVIP-YOLO/train/lwir/labels -> ../visible/labels
#   LLVIP-YOLO/test/lwir/labels  -> ../visible/labels

path: {llvip_root}
train: train/lwir/images
val:   test/lwir/images

nc: 1
names:
  0: person
"""
    out_path.write_text(content)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train YOLOv8m on LLVIP infrared pedestrian dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--llvip-root", required=True,
        help="Path to LLVIP-YOLO directory (contains train/ and test/).",
    )
    p.add_argument(
        "--ckpt-dir", default="./results/yolov8m_llvip",
        help="Directory for ultralytics run outputs (best.pt saved here).",
    )
    p.add_argument("--epochs", type=int, default=50,
                   help="Training epochs (50 is enough for single-class LLVIP).")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="0", help="CUDA device id or 'cpu'.")
    p.add_argument(
        "--pretrained", action="store_true", default=True,
        help="Start from COCO-pretrained YOLOv8m weights (default: True).",
    )
    p.add_argument(
        "--weights", default=None,
        help="Override: start from this .pt checkpoint instead of COCO pretrain.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    set_seed(args.seed)

    llvip_root = Path(args.llvip_root).resolve()
    ckpt_dir = Path(args.ckpt_dir).resolve()
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Verify IR image directories exist
    for split in ("train", "test"):
        img_dir = llvip_root / split / "lwir" / "images"
        if not img_dir.exists():
            raise FileNotFoundError(
                f"LLVIP {split} images not found: {img_dir}\n"
                f"  → Download LLVIP-YOLO and create lwir/labels symlinks."
            )
        n = sum(1 for p in img_dir.glob("*.jpg") if not p.name.startswith("."))
        print(f"[LLVIP] {split}: {n} images in {img_dir}")

    # Write data.yaml to a temp file (or ckpt_dir)
    data_yaml = ckpt_dir / "llvip_train.yaml"
    _write_data_yaml(llvip_root, data_yaml)
    print(f"[LLVIP] data.yaml → {data_yaml}")

    # Select starting weights
    if args.weights:
        start_weights = args.weights
        print(f"[LLVIP] Starting from: {start_weights}")
    else:
        start_weights = "yolov8m.pt"  # downloads COCO pretrain on first use
        print("[LLVIP] Starting from COCO-pretrained yolov8m.pt")

    from ultralytics import YOLO
    model = YOLO(start_weights)

    print(f"\n[LLVIP] Training YOLOv8m on LLVIP  "
          f"(epochs={args.epochs}, batch={args.batch}, seed={args.seed})")
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=640,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        project=str(ckpt_dir),
        name="yolov8m_llvip",
        exist_ok=True,
        plots=False,
        verbose=True,
    )

    best_pt = ckpt_dir / "yolov8m_llvip" / "weights" / "best.pt"
    if best_pt.exists():
        print(f"\n[LLVIP] ✓ Training complete. Best checkpoint: {best_pt}")
        # Quick post-hoc eval on LLVIP test
        print("[LLVIP] Running validation on LLVIP test ...")
        metrics = model.val(data=str(data_yaml), split="val", verbose=True)
        map50 = float(metrics.box.map50) if hasattr(metrics, "box") else float("nan")
        print(f"[LLVIP] mAP@0.5 (LLVIP test) = {map50:.4f}")
        print(f"\nFor Phase-5 domain transfer, pass:")
        print(f"  --llvip-weights {best_pt}")
    else:
        print(f"\n[LLVIP] WARNING: best.pt not found at expected path {best_pt}")
        print("  Check ckpt_dir for ultralytics output structure.")


if __name__ == "__main__":
    main()
