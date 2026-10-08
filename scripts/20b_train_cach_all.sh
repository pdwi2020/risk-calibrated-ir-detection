#!/usr/bin/env bash
# 20b_train_cach_all.sh — Train one CACH checkpoint per detector.
#
# Usage (from project root):
#   bash scripts/20b_train_cach_all.sh --flir-root /workspace/data/flir/FLIR_ADAS_v2 \
#       --preds-dir results/corruption_preds --split-json configs/flir_val_split.json \
#       [--epochs 30] [--batch 256] [--lr 1e-3] ...
#
# Extra args (anything after the script name) are forwarded verbatim to
# 20_train_cach.py, so --epochs, --batch, --lr, etc. all work as expected.
#
# Outputs (one pair per detector):
#   results/cach/yolov8m_cach_best.pt
#   results/cach/rtdetr_cach_best.pt
#   results/cach/faster_rcnn_cach_best.pt
#   results/cach/retinanet_cach_best.pt

set -euo pipefail

DETECTORS=(yolov8m rtdetr faster_rcnn retinanet)

for m in "${DETECTORS[@]}"; do
    echo ""
    echo "============================================================"
    echo "  Training CACH for detector: ${m}"
    echo "============================================================"
    python3 scripts/20_train_cach.py --model "${m}" "$@"
done

echo ""
echo "All CACH checkpoints trained successfully."
