#!/usr/bin/env bash
# P5 lambda sweep: confirm CACH never beats pooled isotonic at any identity-
# regularization strength.  Combined with lambda=0 (cach_eval_unreg.csv) and
# lambda=1.0 (current cach_eval.csv), this gives a 4-point response curve.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

FROOT="${FLIR_ROOT:?set FLIR_ROOT to your FLIR_ADAS_v2 directory}"
EPOCHS=8

# Preserve the lambda=1.0 eval already on disk.
[ -f results/cach_eval.csv ] && cp -f results/cach_eval.csv results/cach_eval_lam1.0.csv

echo "=== LAMBDA SWEEP START $(date)  (lambda in 0.1 0.3) ==="
for LAM in 0.1 0.3; do
  echo ""
  echo "########## LAMBDA = $LAM ##########"
  for m in yolov8m rtdetr faster_rcnn retinanet; do
    echo "--- TRAIN $m lambda=$LAM  $(date) ---"
    python3 scripts/20f_train_cach_perdet_reg.py \
      --flir-root "$FROOT" \
      --preds-dir results/corruption_preds \
      --detectors "$m" \
      --out "results/cach_${m}_lam${LAM}" \
      --epochs "$EPOCHS" --lambda-id "$LAM" 2>&1 | tail -2 || echo "TRAIN_FAILED $m"
    if [ -f "results/cach_${m}_lam${LAM}/cach_best.pt" ]; then
      cp -f "results/cach_${m}_lam${LAM}/cach_best.pt" "results/cach/${m}_cach_best.pt"
    fi
  done
  echo "--- EVAL lambda=$LAM  $(date) ---"
  python3 scripts/24_evaluate_cach.py --root . --device cpu 2>&1 \
    | grep -vE "creating index|index created|Loading and prep|^DONE|Running per|Accumulating|Average (Precision|Recall)" \
    || echo "EVAL_FAILED"
  cp -f results/cach_eval.csv "results/cach_eval_lam${LAM}.csv"
done

echo ""
echo "=== LAMBDA RESPONSE: mean CACH ECE vs baselines ==="
python3 - <<'PY'
import csv, collections, math
def col_means(path, col):
    try: rows=list(csv.DictReader(open(path)))
    except FileNotFoundError: return None
    agg=collections.defaultdict(list)
    for r in rows:
        try:
            v=float(r[col])
            if not math.isnan(v): agg[r["model"]].append(v)
        except (ValueError,KeyError): pass
    per={m:(sum(v)/len(v)) for m,v in agg.items() if v}
    if not per: return None
    per["MEAN"]=sum(per[m] for m in ["yolov8m","rtdetr","faster_rcnn","retinanet"] if m in per)/4
    return per
ref="results/cach_eval_lam1.0.csv"
base={c:col_means(ref,f"ece_{c}") for c in ["nocal","clean","pooled","oracle"]}
lam_files=[("0.0 (unreg)","results/cach_eval_unreg.csv"),
           ("0.1","results/cach_eval_lam0.1.csv"),
           ("0.3","results/cach_eval_lam0.3.csv"),
           ("1.0","results/cach_eval_lam1.0.csv")]
order=["yolov8m","rtdetr","faster_rcnn","retinanet","MEAN"]
print(f'{"":16}'+" ".join(f"{m:>12}" for m in order))
for name in ["nocal","clean","pooled","oracle"]:
    b=base[name]
    if b: print(f'{name:16}'+" ".join(f"{b.get(m,float("nan")):12.3f}" for m in order))
print("-"*16+"-"*65)
for lam,path in lam_files:
    cm=col_means(path,"ece_cach")
    if cm: print(f'{"CACH lam="+lam:16}'+" ".join(f"{cm.get(m,float("nan")):12.3f}" for m in order))
print("\n(pooled isotonic is the target to beat; lower = better)")
PY

echo ""
echo "=== LAMBDA SWEEP DONE $(date) ==="
touch results/P5_SWEEP_DONE
