"""Post-hoc detector evaluation: COCO mAP + cached predictions.

Runs a trained detector over an (images, labels) split, computes mAP@0.5 and
mAP@0.5:0.95 with **pycocotools** (independent of ultralytics' buggy in-training
validator), and dumps per-image predictions + ground truth to JSON for the
risk-calibration pipeline (cost_sensitive / temperature_scaling / conformal).

Predictions are kept at a LOW confidence threshold (default 0.001) so the
downstream risk sweep over theta and the calibration fit have the full score
distribution to work with.

Label class indices are 0=person,1=bike,2=car (our YOLO convention); COCO
category_ids are stored as cls+1 (1..3) for both GT and detections.

Usage (on the GPU box):
  python scripts/04_eval_detector.py --detector yolov8 \
      --weights /workspace/best.pt \
      --img-dir /workspace/data/flir_yolo_real/val/images \
      --label-dir /workspace/data/flir_yolo_real/val/labels \
      --split-json configs/flir_val_split.json --which test \
      --out /workspace/eval/yolov8m_test
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

CLASS_NAMES = ["person", "bike", "car"]


def _build_torchvision(detector: str, weights: str):
    """Build a torchvision detector (frcnn/retinanet) with a 4-class head
    (background + person/bike/car) and load our trained state_dict. Lowers the
    internal score threshold so the risk sweep sees the full low-confidence tail."""
    import torch
    if detector == "faster_rcnn":
        from torchvision.models.detection import fasterrcnn_resnet50_fpn
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        m = fasterrcnn_resnet50_fpn(weights=None)
        in_f = m.roi_heads.box_predictor.cls_score.in_features
        m.roi_heads.box_predictor = FastRCNNPredictor(in_f, 4)
        m.roi_heads.score_thresh = 0.001
        m.roi_heads.detections_per_img = 300
    elif detector == "retinanet":
        from torchvision.models.detection import retinanet_resnet50_fpn
        from torchvision.models.detection.retinanet import RetinaNetClassificationHead
        m = retinanet_resnet50_fpn(weights=None)
        na = m.head.classification_head.num_anchors
        m.head.classification_head = RetinaNetClassificationHead(256, na, 4)
        m.score_thresh = 0.001
        m.detections_per_img = 300
    else:
        raise ValueError(f"unknown torchvision detector: {detector}")
    state = torch.load(weights, map_location="cpu")
    if isinstance(state, dict):
        for k in ("model_state_dict", "model", "state_dict"):
            if k in state and isinstance(state[k], dict):
                state = state[k]
                break
    m.load_state_dict(state)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return m.to(dev).eval()


def _predict_torchvision(model, path, conf: float):
    """Per-image inference -> (boxes xyxy px, scores, labels in {0,1,2})."""
    import numpy as np
    import torch
    from PIL import Image
    with Image.open(path) as im:
        arr = np.asarray(im.convert("RGB"), dtype="float32") / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1)
    dev = next(model.parameters()).device
    with torch.no_grad():
        out = model([t.to(dev)])[0]
    boxes = out["boxes"].cpu().numpy()
    scores = out["scores"].cpu().numpy()
    labels = out["labels"].cpu().numpy().astype(int) - 1  # {1,2,3} -> {0,1,2}
    keep = scores >= conf
    return boxes[keep].tolist(), scores[keep].tolist(), labels[keep].tolist()


def load_gt(label_path: str, W: int, H: int):
    """Parse a YOLO label file -> (boxes xyxy pixels, labels 0/1/2)."""
    boxes, labels = [], []
    if os.path.exists(label_path):
        with open(label_path) as f:
            for line in f:
                parts = line.split()
                if len(parts) != 5:
                    continue
                cls = int(float(parts[0]))
                cx, cy, bw, bh = (float(v) for v in parts[1:5])
                x1 = (cx - bw / 2) * W
                y1 = (cy - bh / 2) * H
                x2 = (cx + bw / 2) * W
                y2 = (cy + bh / 2) * H
                boxes.append([x1, y1, x2, y2])
                labels.append(cls)
    return boxes, labels


def select_images(img_dir: Path, split_json: str, which: str):
    all_imgs = sorted(p for p in img_dir.glob("*.jpg") if not p.name.startswith("."))
    if not split_json:
        return all_imgs
    data = json.loads(Path(split_json).read_text())
    fnames = data["filenames"]  # str(image_id) -> filename
    keep = {fnames[str(i)] for i in data[which]}
    return [p for p in all_imgs if p.name in keep]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--detector", default="yolov8",
                    choices=["yolov8", "yolov11", "rtdetr", "faster_rcnn", "retinanet"])
    ap.add_argument("--weights", required=True)
    ap.add_argument("--img-dir", required=True)
    ap.add_argument("--label-dir", required=True)
    ap.add_argument("--split-json", default=None, help="configs/flir_val_split.json")
    ap.add_argument("--which", default="test", choices=["test", "calibration", "val"])
    ap.add_argument("--conf", type=float, default=0.001, help="keep low to retain all dets")
    ap.add_argument("--iou", type=float, default=0.7, help="NMS IoU")
    ap.add_argument("--out", required=True, help="output path prefix")
    args = ap.parse_args()

    from PIL import Image

    img_dir = Path(args.img_dir)
    label_dir = Path(args.label_dir)
    imgs = select_images(img_dir, args.split_json, args.which)
    print(f"[eval] detector={args.detector} split={args.which} images={len(imgs)}")
    if not imgs:
        sys.exit("No images selected — check --img-dir / --split-json / --which")

    ultra = args.detector in ("yolov8", "yolov11", "rtdetr")
    if args.detector in ("yolov8", "yolov11"):
        from ultralytics import YOLO
        model = YOLO(args.weights)
    elif args.detector == "rtdetr":
        from ultralytics import RTDETR
        model = RTDETR(args.weights)
    else:
        model = _build_torchvision(args.detector, args.weights)

    coco_images, coco_anns, dts = [], [], []
    cached = []
    ann_id = 1

    for idx, p in enumerate(imgs, 1):
        with Image.open(p) as im:
            W, H = im.size
        gt_boxes, gt_labels = load_gt(str(label_dir / (p.stem + ".txt")), W, H)

        image_id = idx
        coco_images.append({"id": image_id})
        for b, l in zip(gt_boxes, gt_labels):
            x1, y1, x2, y2 = b
            coco_anns.append({
                "id": ann_id, "image_id": image_id, "category_id": l + 1,
                "bbox": [x1, y1, x2 - x1, y2 - y1], "area": (x2 - x1) * (y2 - y1),
                "iscrowd": 0,
            })
            ann_id += 1

        if ultra:
            r = model.predict(str(p), conf=args.conf, iou=args.iou, verbose=False)[0]
            pb = r.boxes.xyxy.cpu().numpy().tolist()
            ps = r.boxes.conf.cpu().numpy().tolist()
            pl = r.boxes.cls.cpu().numpy().astype(int).tolist()
        else:
            pb, ps, pl = _predict_torchvision(model, p, args.conf)
        for b, s, l in zip(pb, ps, pl):
            x1, y1, x2, y2 = b
            dts.append({"image_id": image_id, "category_id": int(l) + 1,
                        "bbox": [x1, y1, x2 - x1, y2 - y1], "score": float(s)})

        cached.append({
            "image_id": image_id, "file": p.name, "width": W, "height": H,
            "pred_boxes": pb, "pred_scores": ps, "pred_labels": pl,
            "gt_boxes": gt_boxes, "gt_labels": gt_labels,
        })
        if idx % 100 == 0:
            print(f"  {idx}/{len(imgs)}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pred_path = str(out) + "_predictions.json"
    Path(pred_path).write_text(json.dumps(cached))
    print(f"[eval] cached predictions -> {pred_path}  (n_det={len(dts)})")

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    cats = [{"id": i + 1, "name": n} for i, n in enumerate(CLASS_NAMES)]
    coco_gt = COCO()
    coco_gt.dataset = {"images": coco_images, "annotations": coco_anns, "categories": cats}
    coco_gt.createIndex()
    if not dts:
        print("[eval] NO DETECTIONS — mAP=0")
        return
    coco_dt = coco_gt.loadRes(dts)
    ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
    ev.evaluate(); ev.accumulate(); ev.summarize()
    res = {"detector": args.detector, "split": args.which, "n_images": len(imgs),
           "n_gt": len(coco_anns), "mAP50-95": float(ev.stats[0]), "mAP50": float(ev.stats[1])}
    map_path = str(out) + "_map.json"
    Path(map_path).write_text(json.dumps(res, indent=2))
    print(f"[eval] RESULT: {res}\n[eval] saved -> {map_path}")


if __name__ == "__main__":
    main()
