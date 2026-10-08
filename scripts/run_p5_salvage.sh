#!/usr/bin/env bash
# P5 salvage: retrain CACH per-detector with the identity-anchoring regulariser
# (T->1,b->0) + fixed identity init + mini-val selection, then re-evaluate.
# Compares against the unregularised result (cach_eval_unreg.csv).
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

FROOT="${FLIR_ROOT:?set FLIR_ROOT to your FLIR_ADAS_v2 directory}"
EPOCHS=8
LAMBDA_ID=1.0
mkdir -p results/cach

# Preserve the unregularised result + checkpoints for comparison.
[ -f results/cach_eval.csv ] && cp -f results/cach_eval.csv results/cach_eval_unreg.csv
for m in yolov8m rtdetr faster_rcnn retinanet; do
  [ -f results/cach/${m}_cach_best.pt ] && cp -f results/cach/${m}_cach_best.pt results/cach/${m}_cach_best.unreg.pt
done

echo "=== P5 SALVAGE START $(date)  epochs=$EPOCHS lambda_id=$LAMBDA_ID ==="
for m in yolov8m rtdetr faster_rcnn retinanet; do
  echo ""
  echo "=== TRAIN(reg) $m  $(date) ==="
  python3 scripts/20f_train_cach_perdet_reg.py \
    --flir-root "$FROOT" \
    --preds-dir results/corruption_preds \
    --detectors "$m" \
    --out "results/cach_${m}_reg" \
    --epochs "$EPOCHS" --lambda-id "$LAMBDA_ID" 2>&1 || echo "TRAIN_FAILED $m"
  if [ -f "results/cach_${m}_reg/cach_best.pt" ]; then
    cp -f "results/cach_${m}_reg/cach_best.pt" "results/cach/${m}_cach_best.pt"
    echo "COPIED ${m}_cach_best.pt (reg)"
  else
    echo "NO_CKPT $m"
  fi
done

echo ""
echo "=== EVAL(reg)  $(date) ==="
python3 scripts/24_evaluate_cach.py --root . --device cpu 2>&1 \
  | grep -vE "creating index|index created|Loading and prep|^DONE|Running per|Accumulating|Average (Precision|Recall)" \
  || echo "EVAL_FAILED"

echo ""
echo "=== TABLE XIII: unreg vs reg (per-detector mean ECE) ==="
python3 - <<'PY'
import csv, collections, math
def means(path):
    try: rows=list(csv.DictReader(open(path)))
    except FileNotFoundError: return None
    cols=["ece_nocal","ece_clean","ece_pooled","ece_oracle","ece_cach"]
    agg=collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        for c in cols:
            try:
                v=float(r[c])
                if not math.isnan(v): agg[r["model"]][c].append(v)
            except (ValueError,KeyError): pass
    out={}
    for m in ["yolov8m","rtdetr","faster_rcnn","retinanet"]:
        if m in agg:
            d=agg[m]; out[m]=[ (sum(d[c])/len(d[c])) if d[c] else float('nan') for c in cols]
    return out
unreg=means("results/cach_eval_unreg.csv"); reg=means("results/cach_eval.csv")
hdr=["No-cal","Clean","Pooled","Oracle","CACH"]
print(f'{"detector":12} {"":>8}'+" ".join(f"{h:>8}" for h in hdr))
for m in ["yolov8m","rtdetr","faster_rcnn","retinanet"]:
    if reg and m in reg:
        if unreg and m in unreg:
            print(f'{m:12} {"unreg":>8}'+" ".join(f"{v:8.3f}" for v in unreg[m]))
        print(f'{m:12} {"REG":>8}'+" ".join(f"{v:8.3f}" for v in reg[m]))
def grand(d):
    if not d: return None
    cols=list(zip(*d.values())); return [sum(c)/len(c) for c in cols]
gu,gr=grand(unreg),grand(reg)
if gu: print(f'{"MEAN":12} {"unreg":>8}'+" ".join(f"{v:8.3f}" for v in gu))
if gr: print(f'{"MEAN":12} {"REG":>8}'+" ".join(f"{v:8.3f}" for v in gr))
PY

echo ""
echo "=== P5 SALVAGE DONE $(date) ==="
touch results/P5_SALVAGE_DONE
