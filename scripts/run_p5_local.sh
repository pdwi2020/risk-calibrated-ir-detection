#!/usr/bin/env bash
# P5 — per-detector CACH, fully local (CPU), no GPU rental.
# Trains one CACH head per detector on the calibration split, evaluates each on
# the test split, and writes results/cach_eval.csv (Table XIII source).
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

FROOT="${FLIR_ROOT:?set FLIR_ROOT to your FLIR_ADAS_v2 directory}"
EPOCHS=30
mkdir -p results/cach

# Preserve the pod-trained yolov8m head for a GPU-vs-CPU reproducibility check.
if [ -f results/cach/yolov8m_cach_best.pt ]; then
  cp -f results/cach/yolov8m_cach_best.pt results/cach/yolov8m_cach_best.podGPU.pt
fi

echo "=== P5 LOCAL START $(date) epochs=$EPOCHS ==="
for m in yolov8m rtdetr faster_rcnn retinanet; do
  echo ""
  echo "=== TRAIN $m  $(date) ==="
  python3 scripts/20e_train_cach_perdet.py \
    --flir-root "$FROOT" \
    --preds-dir results/corruption_preds \
    --detectors "$m" \
    --out "results/cach_$m" \
    --epochs "$EPOCHS" 2>&1 || echo "TRAIN_FAILED $m"
  if [ -f "results/cach_$m/cach_best.pt" ]; then
    cp -f "results/cach_$m/cach_best.pt" "results/cach/${m}_cach_best.pt"
    echo "COPIED ${m}_cach_best.pt"
  else
    echo "NO_CKPT $m"
  fi
done

echo ""
echo "=== EVAL  $(date) ==="
python3 scripts/24_evaluate_cach.py --root . --device cpu 2>&1 || echo "EVAL_FAILED"

echo ""
echo "=== TABLE XIII (per-detector mean ECE) ==="
python3 - <<'PY'
import csv, collections, math
try:
    rows = list(csv.DictReader(open("results/cach_eval.csv")))
except FileNotFoundError:
    print("cach_eval.csv not produced"); raise SystemExit
cols = ["ece_nocal","ece_clean","ece_pooled","ece_oracle","ece_cach"]
agg = collections.defaultdict(lambda: collections.defaultdict(list))
for r in rows:
    for c in cols:
        try:
            v = float(r[c])
            if not math.isnan(v):
                agg[r["model"]][c].append(v)
        except (ValueError, KeyError):
            pass
hdr = ["No-cal","Clean","Pooled","Oracle","CACH"]
print(f'{"model":13} ' + " ".join(f"{h:>7}" for h in hdr))
allm = collections.defaultdict(list)
order = ["yolov8m","rtdetr","faster_rcnn","retinanet"]
for m in order:
    if m not in agg: continue
    d = agg[m]
    vals = [ (sum(d[c])/len(d[c])) if d[c] else float("nan") for c in cols ]
    for c,v in zip(cols,vals):
        if v==v: allm[c].append(v)
    print(f'{m:13} ' + " ".join(f"{v:7.3f}" for v in vals))
mean = [ (sum(allm[c])/len(allm[c])) if allm[c] else float("nan") for c in cols ]
print(f'{"MEAN":13} ' + " ".join(f"{v:7.3f}" for v in mean))
print(f"\nrows: {len(rows)}")
PY

echo ""
echo "=== P5 LOCAL DONE $(date) ==="
touch results/P5_LOCAL_DONE
