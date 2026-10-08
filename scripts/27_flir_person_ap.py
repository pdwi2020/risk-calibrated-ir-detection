#!/usr/bin/env python3
"""FLIR person-class AP@0.5 for the in-domain reference of D3/D4 (P6).

D3/D4 (LLVIP->FLIR) are evaluated on the FLIR PERSON class only, so the
"fraction of in-domain performance retained" must be divided by the FLIR
detector's PERSON-class AP@0.5 -- NOT the 3-class mAP (0.769), which mixes in
bicycle and car. This script recomputes the person-only AP@0.5 from the cached
FLIR YOLOv8m test predictions with the same pycocotools settings as
04_eval_detector.py (category_id = label+1; person -> category 1).

Usage:
  python scripts/27_flir_person_ap.py \
      --preds results/yolov8m_flir_seed0/eval/yolov8m_test_predictions.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

CLASS_NAMES = ("person", "bike", "car")
PERSON_CAT = 1  # label 0 + 1


def xyxy_to_xywh(b):
    x1, y1, x2, y2 = b
    return [x1, y1, x2 - x1, y2 - y1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, help="cached *_predictions.json (FLIR test)")
    args = ap.parse_args()

    recs = json.loads(Path(args.preds).read_text())
    coco_images, coco_anns, dts = [], [], []
    ann_id = 1
    for rec in recs:
        image_id = int(rec["image_id"])
        coco_images.append({"id": image_id})
        for b, l in zip(rec["gt_boxes"], rec["gt_labels"]):
            x, y, w, h = xyxy_to_xywh(b)
            coco_anns.append({"id": ann_id, "image_id": image_id,
                              "category_id": int(l) + 1, "bbox": [x, y, w, h],
                              "area": w * h, "iscrowd": 0})
            ann_id += 1
        for b, s, l in zip(rec["pred_boxes"], rec["pred_scores"], rec["pred_labels"]):
            x, y, w, h = xyxy_to_xywh(b)
            dts.append({"image_id": image_id, "category_id": int(l) + 1,
                        "bbox": [x, y, w, h], "score": float(s)})

    cats = [{"id": i + 1, "name": n} for i, n in enumerate(CLASS_NAMES)]
    coco_gt = COCO()
    coco_gt.dataset = {"images": coco_images, "annotations": coco_anns, "categories": cats}
    coco_gt.createIndex()
    coco_dt = coco_gt.loadRes(dts)

    ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
    ev.params.catIds = [PERSON_CAT]          # PERSON only
    ev.evaluate(); ev.accumulate(); ev.summarize()
    person_ap50 = float(ev.stats[1])          # AP @ IoU=0.50, person category
    person_ap5095 = float(ev.stats[0])

    # all-class mAP for context (sanity vs stored 0.769)
    ev_all = COCOeval(coco_gt, coco_dt, iouType="bbox")
    ev_all.evaluate(); ev_all.accumulate(); ev_all.summarize()
    map_all_50 = float(ev_all.stats[1])

    n_person_gt = sum(1 for a in coco_anns if a["category_id"] == PERSON_CAT)
    print("\n=== FLIR YOLOv8m (seed0) test split ===")
    print(f"person AP@0.5      = {person_ap50:.4f}")
    print(f"person AP@0.5:0.95 = {person_ap5095:.4f}")
    print(f"3-class mAP@0.5    = {map_all_50:.4f}  (stored ref 0.769)")
    print(f"n person GT boxes  = {n_person_gt}")

    out = Path(args.preds).parent / "yolov8m_test_person_ap.json"
    out.write_text(json.dumps({
        "person_ap50": round(person_ap50, 4),
        "person_ap50_95": round(person_ap5095, 4),
        "map_all_50": round(map_all_50, 4),
        "n_person_gt": n_person_gt,
    }, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
