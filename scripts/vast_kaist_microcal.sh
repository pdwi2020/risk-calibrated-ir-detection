#!/usr/bin/env bash
# KAIST micro-calibration pod bootstrap script
# Run on vast.ai pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
# Budget: ~2h × $0.11/hr = $0.22
# Usage: bash vast_kaist_microcal.sh

set -euo pipefail
ROOT=/workspace/mv_paper
RESULTS=$ROOT/kaist_results
mkdir -p $RESULTS

echo "=== [1/6] Install dependencies ==="
pip install -q ultralytics kaggle scikit-learn tqdm 2>/dev/null | tail -3

echo "=== [2/6] Kaggle credentials ==="
mkdir -p ~/.kaggle
cat > ~/.kaggle/kaggle.json << 'KEOF'
{"username":"paritoshdwivedi","key":"538587e44b6482b78dc8c596c635359c"}
KEOF
chmod 600 ~/.kaggle/kaggle.json

echo "=== [3/6] Download KAIST dataset (adlteam/kaist-dataset, ~18GB) ==="
mkdir -p /workspace/kaist_dl
cd /workspace/kaist_dl

# Use aria2c for parallel download via Kaggle API
# Kaggle API will redirect to actual URLs
kaggle datasets download adlteam/kaist-dataset -p /workspace/kaist_dl --unzip &
DL_PID=$!

echo "Download PID: $DL_PID, waiting..."
# Check progress every 2 minutes
for i in $(seq 1 30); do
    sleep 120
    if ! kill -0 $DL_PID 2>/dev/null; then
        echo "Download completed!"
        break
    fi
    du -sh /workspace/kaist_dl/ 2>/dev/null || true
    echo "Still downloading... (${i}×2 min elapsed)"
done
wait $DL_PID || true

echo "Download done. Structure:"
ls /workspace/kaist_dl/ | head -20

echo "=== [4/6] Extract test sets (set06-set11 lwir images + annotations) ==="
mkdir -p $ROOT/kaist_thermal/images
mkdir -p $ROOT/kaist_thermal/annotations

# Find the dataset root (may be nested)
KAIST_ROOT=$(find /workspace/kaist_dl -name "set06" -type d | head -1 | xargs dirname 2>/dev/null || echo "")
if [ -z "$KAIST_ROOT" ]; then
    echo "set06 not found — trying alternatives..."
    KAIST_ROOT=$(find /workspace/kaist_dl -maxdepth 3 -type d | head -20)
    echo "$KAIST_ROOT"
fi
echo "KAIST root: $KAIST_ROOT"

# Copy lwir images from test sets (set06-set11), every frame
IMGCOUNT=0
for SET in set06 set07 set08 set09 set10 set11; do
    SETDIR="$KAIST_ROOT/$SET"
    if [ ! -d "$SETDIR" ]; then
        echo "WARNING: $SET not found at $SETDIR"
        continue
    fi
    for SEQ in "$SETDIR"/V*/; do
        SEQNAME=$(basename $SEQ)
        LWIRDIR="$SEQ/lwir"
        if [ ! -d "$LWIRDIR" ]; then continue; fi
        for IMGFILE in "$LWIRDIR"/I*.jpg; do
            [ -f "$IMGFILE" ] || continue
            BASENAME="${SET}_${SEQNAME}_$(basename $IMGFILE)"
            cp "$IMGFILE" "$ROOT/kaist_thermal/images/$BASENAME"
            IMGCOUNT=$((IMGCOUNT+1))
        done
    done
    echo "$SET done"
done
echo "Total thermal images copied: $IMGCOUNT"

# Copy annotation files (set06-set11)
ANN_ROOT=$(find /workspace/kaist_dl -name "*.txt" -path "*/annotations/*" | head -1 | xargs dirname 2>/dev/null | xargs dirname 2>/dev/null || echo "")
if [ -n "$ANN_ROOT" ]; then
    echo "Annotations root: $ANN_ROOT"
    for SET in set06 set07 set08 set09 set10 set11; do
        ANN_DIR="$ANN_ROOT/$SET"
        if [ -d "$ANN_DIR" ]; then
            cp -r "$ANN_DIR" "$ROOT/kaist_thermal/annotations/$SET"
        fi
    done
else
    echo "WARNING: annotations not found in standard path — searching..."
    find /workspace/kaist_dl -name "*.txt" | grep -i "annot\|label" | head -5
fi

ls $ROOT/kaist_thermal/images/ | wc -l
echo "Images ready."

echo "=== [5/6] Run YOLOv8m inference on KAIST thermal images ==="
# Pull model from X9 (rsync from host) — must be set up externally
# or the model weights should already be copied here

# Write inference script inline
python3 << 'PYEOF'
import json
import sys
from pathlib import Path
import torch

ROOT = Path("/workspace/mv_paper")
IMG_DIR = ROOT / "kaist_thermal" / "images"
ANN_DIR = ROOT / "kaist_thermal" / "annotations"
RESULTS = ROOT / "kaist_results"
RESULTS.mkdir(exist_ok=True)

# Find model
MODEL_PATH = ROOT / "yolov8m_flir_seed0_best.pt"
if not MODEL_PATH.exists():
    print(f"ERROR: model not found at {MODEL_PATH}")
    print("Available pt files:", list(ROOT.glob("*.pt")))
    sys.exit(1)

from ultralytics import YOLO
model = YOLO(str(MODEL_PATH))
model.overrides["verbose"] = False

imgs = sorted(IMG_DIR.glob("*.jpg"))
print(f"Running inference on {len(imgs)} images...")

preds = {}  # img_stem -> list of [x1,y1,x2,y2,score,cls]
BATCH = 32
for i in range(0, len(imgs), BATCH):
    batch = imgs[i:i+BATCH]
    results = model(batch, imgsz=640, conf=0.001, verbose=False)
    for img_path, r in zip(batch, results):
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            preds[img_path.stem] = []
            continue
        dets = []
        for j in range(len(boxes)):
            x1, y1, x2, y2 = boxes.xyxy[j].tolist()
            score = float(boxes.conf[j])
            cls = int(boxes.cls[j])
            dets.append([x1, y1, x2, y2, score, cls])
        preds[img_path.stem] = dets
    if (i // BATCH) % 5 == 0:
        print(f"  {i+len(batch)}/{len(imgs)} done")

print(f"Inference complete. {len(preds)} images processed.")

# Parse KAIST annotations (format: text files with one box per line: classname x y w h ignore occlusion)
# KAIST annotation format per file per frame:
# % [set_name] [seq_name] [frame_id]
# person x y w h occl
gt = {}  # img_stem -> list of [x1,y1,x2,y2]

def find_annotation(stem, ann_dir):
    """stem like set06_V000_I00000, look for annotation file"""
    parts = stem.split("_")
    if len(parts) < 3:
        return []
    set_id = parts[0]   # set06
    seq_id = parts[1]   # V000
    frame_id = parts[2] # I00000

    # Try different annotation layouts
    for layout in [
        ann_dir / set_id / seq_id / f"{frame_id}.txt",
        ann_dir / set_id / f"{seq_id}_{frame_id}.txt",
        ann_dir / f"{set_id}_{seq_id}" / f"{frame_id}.txt",
    ]:
        if layout.exists():
            boxes = []
            for line in layout.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith('%'):
                    continue
                parts_l = line.split()
                if len(parts_l) < 5:
                    continue
                cls_name = parts_l[0].lower()
                if cls_name not in ('person', 'people', 'cyclist'):
                    continue
                try:
                    x, y, w, h = float(parts_l[1]), float(parts_l[2]), float(parts_l[3]), float(parts_l[4])
                    # Some annotations have occlusion flags; skip ignore=1
                    if len(parts_l) >= 6 and parts_l[5] in ('1', '2'):
                        continue  # ignore/heavily occluded
                    boxes.append([x, y, x+w, y+h])
                except ValueError:
                    continue
            return boxes
    return []

for stem in list(preds.keys())[:10]:
    boxes = find_annotation(stem, ANN_DIR)
    print(f"  {stem}: pred={len(preds[stem])}, gt={len(boxes)}")

ann_found = 0
for stem in preds:
    boxes = find_annotation(stem, ANN_DIR)
    gt[stem] = boxes
    if boxes:
        ann_found += 1

print(f"GT annotations: {ann_found}/{len(gt)} images have GT boxes")

# Save predictions + GT
out = {"preds": preds, "gt": gt}
with open(RESULTS / "kaist_preds.json", "w") as f:
    json.dump(out, f)
print(f"Saved: {RESULTS}/kaist_preds.json")
PYEOF

echo "=== [6/6] Run KAIST micro-calibration ==="
python3 << 'PYEOF'
import json
import random
import csv
import numpy as np
from pathlib import Path
from sklearn.isotonic import IsotonicRegression

RESULTS = Path("/workspace/mv_paper/kaist_results")
IOU_THRESH = 0.5
LABEL_FRACS = [0, 1, 5, 10, 100]
N_SEEDS = 10

data = json.loads((RESULTS / "kaist_preds.json").read_text())
preds = data["preds"]
gt_all = data["gt"]

def iou(b1, b2):
    ix1, iy1, ix2, iy2 = max(b1[0],b2[0]), max(b1[1],b2[1]), min(b1[2],b2[2]), min(b1[3],b2[3])
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    a1 = (b1[2]-b1[0])*(b1[3]-b1[1])
    a2 = (b2[2]-b2[0])*(b2[3]-b2[1])
    return inter / (a1 + a2 - inter + 1e-8)

def match_dets(dets, gts, iou_thresh=0.5):
    """Returns list of (score, is_tp) tuples"""
    if not dets:
        return []
    dets_sorted = sorted(dets, key=lambda x: -x[4])  # sort by score desc
    gt_matched = [False] * len(gts)
    results = []
    for det in dets_sorted:
        x1,y1,x2,y2,score,cls = det
        best_iou = 0
        best_j = -1
        for j, gt in enumerate(gts):
            if gt_matched[j]:
                continue
            i = iou([x1,y1,x2,y2], gt)
            if i > best_iou:
                best_iou = i
                best_j = j
        if best_iou >= iou_thresh and best_j >= 0:
            gt_matched[best_j] = True
            results.append((score, 1))
        else:
            results.append((score, 0))
    return results

def compute_ece(pairs, n_bins=15):
    if not pairs:
        return 0.0
    scores = np.array([p[0] for p in pairs])
    is_tp = np.array([p[1] for p in pairs])
    bins = np.linspace(0, 1, n_bins+1)
    ece = 0.0
    for b in range(n_bins):
        mask = (scores >= bins[b]) & (scores < bins[b+1])
        if mask.sum() == 0:
            continue
        avg_conf = scores[mask].mean()
        avg_prec = is_tp[mask].mean()
        ece += mask.sum() / len(scores) * abs(avg_conf - avg_prec)
    return ece

# Build all (score, is_tp) pairs
all_pairs = []
img_pairs = {}  # img -> list of (score, is_tp)
for stem, dets in preds.items():
    gts = gt_all.get(stem, [])
    # Filter person class (cls=0)
    person_dets = [d for d in dets if d[5] == 0]
    pairs = match_dets(person_dets, gts)
    if pairs:
        img_pairs[stem] = pairs
        all_pairs.extend(pairs)

print(f"Total detections: {len(all_pairs)}, images with detections: {len(img_pairs)}")
print(f"TP rate: {sum(p[1] for p in all_pairs)/len(all_pairs):.3f}" if all_pairs else "No detections!")

# No-cal ECE
ece_nocal = compute_ece(all_pairs)
print(f"No-cal ECE: {ece_nocal:.4f}")

# Split images into calib (50%) and test (50%)
all_img_stems = sorted(img_pairs.keys())
random.seed(42)
random.shuffle(all_img_stems)
n_cal = len(all_img_stems) // 2
cal_imgs = set(all_img_stems[:n_cal])
test_imgs = set(all_img_stems[n_cal:])
test_pairs = []
for stem in test_imgs:
    test_pairs.extend(img_pairs[stem])
print(f"Split: {len(cal_imgs)} calib, {len(test_imgs)} test images")
print(f"No-cal ECE on test: {compute_ece(test_pairs):.4f}")

rows = []
for frac in LABEL_FRACS:
    ece_list = []
    for seed in range(N_SEEDS):
        random.seed(seed)
        if frac == 0:
            cal_sample = []
        elif frac == 100:
            cal_sample = list(cal_imgs)
        else:
            n_sample = max(1, int(len(cal_imgs) * frac / 100))
            cal_sample = random.sample(list(cal_imgs), n_sample)

        cal_pairs = []
        for stem in cal_sample:
            cal_pairs.extend(img_pairs.get(stem, []))

        if not cal_pairs or frac == 0:
            # No calibration — use raw scores
            ece = compute_ece(test_pairs)
        else:
            scores_cal = np.array([p[0] for p in cal_pairs])
            tp_cal = np.array([p[1] for p in cal_pairs])
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(scores_cal, tp_cal)
            # Apply to test
            scores_test = np.array([p[0] for p in test_pairs])
            tp_test = np.array([p[1] for p in test_pairs])
            scores_cal_test = iso.predict(scores_test)
            ece = compute_ece(list(zip(scores_cal_test.tolist(), tp_test.tolist())))

        ece_list.append(ece)
        if seed == 0:
            n_used = len(cal_sample)

    mean_ece = np.mean(ece_list)
    std_ece = np.std(ece_list)
    rows.append({
        "frac_pct": frac,
        "n_images": n_used if frac > 0 else 0,
        "ece_mean": mean_ece,
        "ece_std": std_ece
    })
    print(f"  frac={frac:3d}%  n={n_used if frac > 0 else 0:4d}  ECE={mean_ece:.4f}±{std_ece:.4f}")

# Save CSV
with open(RESULTS / "kaist_micro_cal.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["frac_pct","n_images","ece_mean","ece_std"])
    writer.writeheader()
    writer.writerows(rows)

# Summary JSON
summary = {
    "n_test_images": len(test_imgs),
    "n_total_dets": len(all_pairs),
    "tp_rate": sum(p[1] for p in all_pairs)/len(all_pairs) if all_pairs else 0,
    "ece_nocal": ece_nocal,
    "results": rows
}
import json
with open(RESULTS / "kaist_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\nKAIST micro-calibration complete!")
print(f"Results: {RESULTS}/kaist_micro_cal.csv")
PYEOF

echo ""
echo "=== DONE ==="
echo "Results in: $RESULTS/"
ls $RESULTS/
echo ""
echo "Rsync back with:"
echo "  rsync -avz --progress /workspace/mv_paper/kaist_results/ <X9_RSYNC_TARGET>"
