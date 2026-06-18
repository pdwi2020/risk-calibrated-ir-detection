#!/usr/bin/env bash
set -euo pipefail

cd /workspace/repo

STATUS_DIR=/workspace/status
mkdir -p "$STATUS_DIR" results/retinanet_flir_seed0/eval /workspace/data/public

mark() {
  local name="$1"
  printf "\n==== %s %s ====\n" "$(date -Is)" "$name"
  printf "%s\n" "$name" > "$STATUS_DIR/current"
}

mark runtime_setup
find /workspace/repo -name '._*' -delete
chmod 600 /root/.kaggle/kaggle.json || true
if command -v apt-get >/dev/null 2>&1; then
  apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends \
    libxcb1 libgl1 libglib2.0-0 libx11-6 libxext6 libsm6 libxrender1
fi
python -m pip install -q --upgrade pip
python -m pip install -q 'ultralytics==8.3.253' pycocotools opencv-python-headless albumentations kaggle
python - <<'PY'
import torch, torchvision
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("torchvision", torchvision.__version__)
import cv2, albumentations
print("cv2", cv2.__version__)
print("albumentations", albumentations.__version__)
try:
    import ultralytics
    print("ultralytics", ultralytics.__version__)
except Exception as exc:
    print("ultralytics import failed", repr(exc))
PY
touch "$STATUS_DIR/runtime_setup.done"

mark download_flir_if_needed
if ! find /workspace/data/public -maxdepth 4 -type f -path '*/FLIR_ADAS_v2/images_thermal_train/coco.json' | grep -q .; then
  attempt=1
  while ! find /workspace/data/public -maxdepth 4 -type f -path '*/FLIR_ADAS_v2/images_thermal_train/coco.json' | grep -q .; do
    echo "FLIR download attempt $attempt"
    set +e
    kaggle datasets download -d samdazel/teledyne-flir-adas-thermal-dataset-v2 \
      -p /workspace/data/public --unzip
    rc=$?
    set -e
    if find /workspace/data/public -maxdepth 4 -type f -path '*/FLIR_ADAS_v2/images_thermal_train/coco.json' | grep -q .; then
      break
    fi
    echo "FLIR download attempt $attempt failed with rc=$rc; retrying after 120s"
    attempt=$((attempt + 1))
    sleep 120
  done
fi
FLIR_ROOT="$(find /workspace/data/public -maxdepth 4 -type d -name FLIR_ADAS_v2 | head -1)"
if [[ -z "$FLIR_ROOT" ]]; then
  echo "FLIR_ADAS_v2 not found under /workspace/data/public" >&2
  exit 10
fi
printf "%s\n" "$FLIR_ROOT" > "$STATUS_DIR/flir_root.txt"
echo "FLIR_ROOT=$FLIR_ROOT"
touch "$STATUS_DIR/download_flir_if_needed.done"

mark prepare_yolo_layout
mkdir -p /workspace/data/flir_yolo/train /workspace/data/flir_yolo/val
rm -rf /workspace/data/flir_yolo/train/images /workspace/data/flir_yolo/val/images
ln -s "$FLIR_ROOT/images_thermal_train/data" /workspace/data/flir_yolo/train/images
ln -s "$FLIR_ROOT/images_thermal_val/data" /workspace/data/flir_yolo/val/images
python - <<'PY'
from pathlib import Path
for path in Path("configs").glob("flir_yolo*.yaml"):
    lines = path.read_text().splitlines()
    lines = ["path: /workspace/data/flir_yolo" if line.startswith("path: ") else line for line in lines]
    path.write_text("\n".join(lines) + "\n")
PY
python - <<'PY'
from pathlib import Path
pairs = {
    "train_images": Path("/workspace/data/flir_yolo/train/images"),
    "train_labels": Path("/workspace/data/flir_yolo/train/labels"),
    "val_images": Path("/workspace/data/flir_yolo/val/images"),
    "val_labels": Path("/workspace/data/flir_yolo/val/labels"),
}
for name, path in pairs.items():
    print(f"{name}={len(list(path.glob('*')))}")
print("train_txt", len(list(pairs["train_labels"].glob("*.txt"))))
print("val_txt", len(list(pairs["val_labels"].glob("*.txt"))))
PY
touch "$STATUS_DIR/prepare_yolo_layout.done"

mark py_compile
python -m py_compile scripts/06_finish_retinanet.py scripts/04_eval_detector.py scripts/05_risk_eval.py scripts/07_detector_comparison.py
touch "$STATUS_DIR/py_compile.done"

mark retinanet_finish
python scripts/06_finish_retinanet.py \
  --flir-root "$FLIR_ROOT" \
  --split-json configs/flir_val_split.json \
  --ckpt-dir results/retinanet_flir_seed0 \
  --start-epoch 39 \
  --end-epoch 50 \
  --batch 4 \
  --workers 4 \
  --lr 0.0005 \
  --eval-every 5 \
  --seed 0
touch "$STATUS_DIR/retinanet_finish.done"

mark retinanet_eval
python scripts/04_eval_detector.py \
  --detector retinanet \
  --weights results/retinanet_flir_seed0/retinanet_best.pth \
  --img-dir /workspace/data/flir_yolo/val/images \
  --label-dir /workspace/data/flir_yolo/val/labels \
  --split-json configs/flir_val_split.json \
  --which test \
  --out results/retinanet_flir_seed0/eval/retinanet_test

python scripts/04_eval_detector.py \
  --detector retinanet \
  --weights results/retinanet_flir_seed0/retinanet_best.pth \
  --img-dir /workspace/data/flir_yolo/val/images \
  --label-dir /workspace/data/flir_yolo/val/labels \
  --split-json configs/flir_val_split.json \
  --which calibration \
  --out results/retinanet_flir_seed0/eval/retinanet_calib
touch "$STATUS_DIR/retinanet_eval.done"

mark retinanet_risk
python scripts/05_risk_eval.py \
  --calib-json results/retinanet_flir_seed0/eval/retinanet_calib_predictions.json \
  --test-json results/retinanet_flir_seed0/eval/retinanet_test_predictions.json \
  --map-json results/retinanet_flir_seed0/eval/retinanet_test_map.json \
  --out results/retinanet_flir_seed0/eval/retinanet_risk_summary.json
touch "$STATUS_DIR/retinanet_risk.done"

mark detector_comparison
python scripts/07_detector_comparison.py --root .
touch "$STATUS_DIR/detector_comparison.done"

mark complete
touch "$STATUS_DIR/complete.done"
