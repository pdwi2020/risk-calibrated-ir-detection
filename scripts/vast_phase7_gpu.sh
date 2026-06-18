#!/usr/bin/env bash
# vast_phase7_gpu.sh — Phase 7 + 8 + SOTA GPU session (RunPod/vast.ai)
#
# Stages:
#   1. Runtime setup (pip deps, CUDA check)
#   2. Download datasets (public FLIR COCO + private FLIR YOLO labels + LLVIP)
#   3. Prepare YOLO layout (symlinks + patched configs)
#   4. Train YOLOv11m on FLIR (SOTA comparison)
#   5. Eval YOLOv11m → calib + test prediction JSONs
#   6. Corruption inference (5 models × 6 types × 4 sev + LLVIP)
#   7. CACH training (Phase 8.2, frozen YOLOv8m)
#
# Run from project root (/workspace/repo):
#   bash scripts/vast_phase7_gpu.sh 2>&1 | tee /workspace/phase7_log.txt
#
# Rsync results back after:
#   rsync -avz --progress <user>@<host>:/workspace/repo/results/ \
#       "/Volumes/Crucial X9/Research Projects/MV_Paper/results/"

set -euo pipefail

STATUS=/workspace/status
mkdir -p "$STATUS"

mark() { printf '\n========== %s  %s ==========\n' "$(date '+%H:%M:%S')" "$1"; echo "$1" > "$STATUS/current"; }

# ─── 1. Runtime ──────────────────────────────────────────────────────────────
mark runtime_setup

find /workspace/repo -name '._*' -delete 2>/dev/null || true
chmod 600 /root/.kaggle/kaggle.json 2>/dev/null || true

if command -v apt-get >/dev/null 2>&1; then
  apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends \
    rsync libxcb1 libgl1 libglib2.0-0 libx11-6 libxext6 libsm6 libxrender1
fi

python -m pip install -q --upgrade pip
python -m pip install -q \
  'ultralytics==8.3.253' pycocotools opencv-python-headless \
  albumentations scikit-learn scipy kaggle timm

python - <<'PY'
import torch
print(f"torch {torch.__version__}  CUDA: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("ERROR: no CUDA")
p = torch.cuda.get_device_properties(0)
print(f"GPU: {p.name}  VRAM: {p.total_memory/1e9:.1f} GB")
PY
touch "$STATUS/runtime_setup.done"

# ─── 2. Datasets ─────────────────────────────────────────────────────────────
mark datasets

mkdir -p /workspace/data/public /workspace/data/flir_labels /workspace/data/llvip

download_once() {
  local slug="$1" dest="$2"
  if [ -d "$dest" ] && [ -n "$(ls -A "$dest" 2>/dev/null)" ]; then
    echo "[$slug] already present at $dest"; return
  fi
  kaggle datasets download -d "$slug" -p "$dest" --unzip
}

# Public FLIR ADAS v2 (COCO format — needed by 14a corruption inference)
download_once "samdazel/teledyne-flir-adas-thermal-dataset-v2" /workspace/data/public

# FLIR YOLO labels: generate from COCO JSON (avoids private Kaggle dataset auth)
FLIR_ROOT=$(find /workspace/data/public -maxdepth 4 -type d -name FLIR_ADAS_v2 | head -1)
YOLO_LABELS_TRAIN=/workspace/data/flir_labels/labels/train
YOLO_LABELS_VAL=/workspace/data/flir_labels/labels/val
if [ -d "$YOLO_LABELS_TRAIN" ] && [ -n "$(ls -A "$YOLO_LABELS_TRAIN" 2>/dev/null)" ]; then
  echo "[flir_labels] already generated"
else
  echo "[flir_labels] converting COCO JSON → YOLO txt ..."
  mkdir -p "$YOLO_LABELS_TRAIN" "$YOLO_LABELS_VAL"
  python3 - <<'PY'
import json, os
from pathlib import Path

CAT_MAP = {}  # coco cat_id -> yolo class idx (0-based, person/bike/car only)

def convert_split(json_path, out_dir):
    data = json.loads(Path(json_path).read_text())
    # build cat_id -> yolo_idx mapping (person=0, bike=1, car=2)
    keep = {'person': 0, 'bike': 1, 'bicycle': 1, 'car': 2}
    cat_remap = {}
    for c in data['categories']:
        name = c['name'].lower()
        if name in keep:
            cat_remap[c['id']] = keep[name]
    # image id -> filename stem
    id2stem = {img['id']: Path(img['file_name']).stem for img in data['images']}
    id2wh   = {img['id']: (img['width'], img['height']) for img in data['images']}
    # accumulate per-image annotations
    anns = {}
    for ann in data['annotations']:
        cid = cat_remap.get(ann['category_id'])
        if cid is None:
            continue
        iid = ann['image_id']
        if iid not in anns:
            anns[iid] = []
        x, y, w, h = ann['bbox']
        iw, ih = id2wh[iid]
        xc = (x + w / 2) / iw
        yc = (y + h / 2) / ih
        wn = w / iw
        hn = h / ih
        anns[iid].append(f"{cid} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")
    # write txt files
    out_dir = Path(out_dir)
    written = 0
    for iid, lines in anns.items():
        stem = id2stem.get(iid)
        if not stem:
            continue
        (out_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n")
        written += 1
    # also write empty files for images with no kept annotations
    for img in data['images']:
        stem = Path(img['file_name']).stem
        p = out_dir / f"{stem}.txt"
        if not p.exists():
            p.write_text("")
    print(f"  {json_path}: wrote {written} label files to {out_dir}")

import os
flir_root = os.environ.get('FLIR_ROOT', '')
if not flir_root:
    import subprocess
    flir_root = subprocess.check_output(
        "find /workspace/data/public -maxdepth 4 -type d -name FLIR_ADAS_v2 | head -1",
        shell=True).decode().strip()

for split, out in [('train', os.environ.get('YOLO_LABELS_TRAIN','/workspace/data/flir_labels/labels/train')),
                   ('val',   os.environ.get('YOLO_LABELS_VAL',  '/workspace/data/flir_labels/labels/val'))]:
    json_candidates = [
        f"{flir_root}/images_thermal_{split}/coco.json",
        f"{flir_root}/images_thermal_{split}/thermal_{split}_coco_format.json",
        f"{flir_root}/images_thermal_{split}/data/coco.json",
    ]
    found = next((p for p in json_candidates if os.path.exists(p)), None)
    if found is None:
        import glob
        found = next(iter(glob.glob(f"{flir_root}/images_thermal_{split}/**/*.json", recursive=True)), None)
    if found is None:
        print(f"ERROR: no COCO JSON found for split={split} under {flir_root}/images_thermal_{split}/")
        raise SystemExit(1)
    print(f"Found JSON: {found}")
    convert_split(found, out)
PY
  echo "[flir_labels] conversion done"
fi

# LLVIP: use pre-rsynced data from X9 (already at /workspace/data/llvip/LLVIP-YOLO)
# Wait for rsync to finish if still in progress
LLVIP_CHECK=/workspace/data/llvip/LLVIP-YOLO
echo "[llvip] waiting for LLVIP-YOLO data at $LLVIP_CHECK ..."
for i in $(seq 1 60); do
  if [ -d "$LLVIP_CHECK/test/lwir/images" ] && [ -n "$(ls -A "$LLVIP_CHECK/test/lwir/images" 2>/dev/null)" ]; then
    echo "[llvip] data ready (waited ${i}x10s)"; break
  fi
  echo "[llvip] not ready yet, waiting 10s (attempt $i/60)..."
  sleep 10
done

LLVIP_ROOT=$(find /workspace/data/llvip -maxdepth 3 -type d -name LLVIP-YOLO | head -1)
echo "FLIR_ROOT=$FLIR_ROOT"
echo "LLVIP_ROOT=$LLVIP_ROOT"
touch "$STATUS/datasets.done"

# ─── 3. YOLO layout (symlinks + patched configs) ─────────────────────────────
mark yolo_layout

mkdir -p /workspace/data/flir_yolo/train /workspace/data/flir_yolo/val

# Symlink raw images from COCO download
rm -rf /workspace/data/flir_yolo/train/images /workspace/data/flir_yolo/val/images
ln -s "$FLIR_ROOT/images_thermal_train/data" /workspace/data/flir_yolo/train/images
ln -s "$FLIR_ROOT/images_thermal_val/data"   /workspace/data/flir_yolo/val/images

# Copy/symlink YOLO labels from private dataset
FLIR_LABELS_ROOT=$(find /workspace/data/flir_labels -maxdepth 3 \( -type d -name labels \) | head -1 | xargs -I{} dirname {})
if [ -z "$FLIR_LABELS_ROOT" ]; then
  FLIR_LABELS_ROOT=$(find /workspace/data/flir_labels -maxdepth 2 -type d | head -1)
fi
echo "FLIR_LABELS_ROOT=$FLIR_LABELS_ROOT"

if [ -d "$FLIR_LABELS_ROOT/labels" ]; then
  ln -sf "$FLIR_LABELS_ROOT/labels/train" /workspace/data/flir_yolo/train/labels 2>/dev/null || \
    cp -r "$FLIR_LABELS_ROOT/labels/train" /workspace/data/flir_yolo/train/labels
  ln -sf "$FLIR_LABELS_ROOT/labels/val"   /workspace/data/flir_yolo/val/labels 2>/dev/null || \
    cp -r "$FLIR_LABELS_ROOT/labels/val"   /workspace/data/flir_yolo/val/labels
elif [ -d "$FLIR_LABELS_ROOT/train/labels" ]; then
  ln -sf "$FLIR_LABELS_ROOT/train/labels" /workspace/data/flir_yolo/train/labels 2>/dev/null || \
    cp -r "$FLIR_LABELS_ROOT/train/labels" /workspace/data/flir_yolo/train/labels
  ln -sf "$FLIR_LABELS_ROOT/val/labels"   /workspace/data/flir_yolo/val/labels 2>/dev/null || \
    cp -r "$FLIR_LABELS_ROOT/val/labels"   /workspace/data/flir_yolo/val/labels
else
  echo "WARNING: could not find YOLO labels dir — check FLIR_LABELS_ROOT=$FLIR_LABELS_ROOT"
fi

# Verify
echo "train images: $(ls /workspace/data/flir_yolo/train/images/ 2>/dev/null | wc -l)"
echo "train labels: $(ls /workspace/data/flir_yolo/train/labels/ 2>/dev/null | wc -l)"
echo "val images:   $(ls /workspace/data/flir_yolo/val/images/   2>/dev/null | wc -l)"
echo "val labels:   $(ls /workspace/data/flir_yolo/val/labels/   2>/dev/null | wc -l)"

# Patch all flir_yolo*.yaml files to use the workspace path
python - <<'PY'
from pathlib import Path
for p in Path("configs").glob("flir_yolo*.yaml"):
    lines = p.read_text().splitlines()
    lines = ["path: /workspace/data/flir_yolo" if l.startswith("path: ") else l for l in lines]
    p.write_text("\n".join(lines) + "\n")
    print(f"patched {p}")
PY

touch "$STATUS/yolo_layout.done"

# ─── 4. Train YOLOv11m ────────────────────────────────────────────────────────
mark yolov11_train

python scripts/21_train_yolov11.py \
  --flir-root  "$FLIR_ROOT" \
  --data-yaml  configs/flir_yolo.yaml \
  --ckpt-dir   /workspace/checkpoints \
  --epochs 100 \
  --batch  16 \
  --seed   0

mkdir -p results/yolov11m_flir_seed0/weights
cp /workspace/checkpoints/yolov11m_flir/weights/best.pt \
   results/yolov11m_flir_seed0/weights/best.pt

touch "$STATUS/yolov11_train.done"

# ─── 5. Eval YOLOv11m ─────────────────────────────────────────────────────────
mark yolov11_eval

YOLO11_WEIGHTS=results/yolov11m_flir_seed0/weights/best.pt
mkdir -p results/yolov11m_flir_seed0/eval

python scripts/04_eval_detector.py \
  --detector   yolov11 \
  --weights    "$YOLO11_WEIGHTS" \
  --img-dir    /workspace/data/flir_yolo/val/images \
  --label-dir  /workspace/data/flir_yolo/val/labels \
  --split-json configs/flir_val_split.json \
  --which calibration \
  --out results/yolov11m_flir_seed0/eval/yolov11m_calib

python scripts/04_eval_detector.py \
  --detector   yolov11 \
  --weights    "$YOLO11_WEIGHTS" \
  --img-dir    /workspace/data/flir_yolo/val/images \
  --label-dir  /workspace/data/flir_yolo/val/labels \
  --split-json configs/flir_val_split.json \
  --which test \
  --out results/yolov11m_flir_seed0/eval/yolov11m_test

# Extract and save mAP50 via ultralytics val
python - <<'PY'
import json
from pathlib import Path
from ultralytics import YOLO
m = YOLO("results/yolov11m_flir_seed0/weights/best.pt")
metrics = m.val(data="configs/flir_yolo.yaml", verbose=False)
d = metrics.results_dict
map50 = float(d.get("metrics/mAP50(B)", 0.0))
map5095 = float(d.get("metrics/mAP50-95(B)", 0.0))
Path("results/yolov11m_flir_seed0/eval/yolov11m_test_map.json").write_text(
    json.dumps({"mAP50": map50, "mAP50_95": map5095}))
print(f"YOLOv11m mAP50={map50:.4f}  mAP50-95={map5095:.4f}")
PY

touch "$STATUS/yolov11_eval.done"

# ─── 6. Corruption inference (Phase 7.0) ──────────────────────────────────────
mark corruption_infer

python scripts/14a_corruption_infer.py \
  --flir-root        "$FLIR_ROOT" \
  --llvip-root       "$LLVIP_ROOT" \
  --split-json       configs/flir_val_split.json \
  --llvip-split-json configs/llvip_val_split.json

touch "$STATUS/corruption_infer.done"

# ─── 7. CACH training (Phase 8.2) ────────────────────────────────────────────
mark cach_train

python scripts/20_train_cach.py \
  --flir-root  "$FLIR_ROOT" \
  --preds-dir  results/corruption_preds \
  --split-json configs/flir_val_split.json \
  --out        results/cach \
  --epochs 30 \
  --batch 256 \
  --lr 1e-3

touch "$STATUS/cach_train.done"

# ─── Done ─────────────────────────────────────────────────────────────────────
mark complete
echo ""
echo "All done. Rsync to X9:"
echo "  rsync -avz --progress <user>@<host>:/workspace/repo/results/ \\"
echo "      '/Volumes/Crucial X9/Research Projects/MV_Paper/results/'"
echo ""
echo "Files produced:"
echo "  results/yolov11m_flir_seed0/    weights + eval JSONs"
echo "  results/corruption_preds/       120 condition JSONs + LLVIP"
echo "  results/llvip_transfer_preds.json"
echo "  results/cach/                   CACH checkpoint + training log"
