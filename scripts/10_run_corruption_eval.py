"""Driver: run full corruption robustness evaluation for all 4 detectors.
Run from project root: python /tmp/mv_paper_scripts/10_run_corruption_eval.py

Requires:
  - GPU (or CPU will be slow) — run on vast.ai box
  - model weights: results/*/eval/*.pt or best.pt
  - FLIR test split: configs/flir_val_split.json + images

Outputs per detector:
  results/<det>_flir_seed0/eval/corruption/corruption_results.json
  results/<det>_flir_seed0/eval/corruption/mce_summary.csv

Summary across all detectors:
  results/corruption_comparison.csv
"""
from __future__ import annotations
import json, sys, csv
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np
try:
    import cv2
except ImportError:
    raise SystemExit("cv2 required: pip install opencv-python-headless")

from src.corruption.corruption_pipeline import run_corruption_eval, CORRUPTION_REGISTRY

SPLIT_JSON = REPO / "configs/flir_val_split.json"

DETECTORS = {
    "yolov8m": {
        "weights": REPO / "results/yolov8m_flir_seed0/weights/best.pt",
        "type": "ultralytics",
        "clean_map": 0.7686,
    },
    "rtdetr": {
        "weights": REPO / "results/rtdetr_flir_seed0/rtdetr_flir/weights/best.pt",
        "type": "ultralytics_rtdetr",
        "clean_map": 0.7721,
    },
    "faster_rcnn": {
        "weights": REPO / "results/faster_rcnn_flir_seed0/weights/frcnn_best.pth",
        "type": "frcnn",
        "clean_map": 0.7153,
    },
    "retinanet": {
        "weights": REPO / "results/retinanet_flir_seed0/retinanet_best.pth",
        "type": "retinanet",
        "clean_map": 0.6176,
    },
}

# ── Image loading from FLIR test split ────────────────────────────────────
def _get_test_images_and_gt():
    split = json.loads(SPLIT_JSON.read_text())
    test_ids = set(split["test"])
    # GT from FLIR val COCO annotation
    coco_ann = REPO / "datasets/flir_adas_v2/FLIR_ADAS_v2/images_thermal_val/coco.json"
    gt_data = json.loads(coco_ann.read_text())
    id2path = {img["id"]: img["file_name"] for img in gt_data["images"]}
    img_ids = [i for i in test_ids if i in id2path]

    # Build per-image GT in detector format
    from collections import defaultdict
    ann_map = defaultdict(list)
    for ann in gt_data["annotations"]:
        if ann["image_id"] in test_ids:
            ann_map[ann["image_id"]].append(ann)

    # FLIR COCO category IDs: person=1, bike=2, car=3  →  our labels: 0, 1, 2
    CLASS_MAP = {1: 0, 2: 1, 3: 2}
    VALID = set(CLASS_MAP.keys())

    image_paths, gt_records = [], []
    # file_name values are relative to images_thermal_val/ (e.g. "data/FLIR_00001.jpg")
    base = REPO / "datasets/flir_adas_v2/FLIR_ADAS_v2/images_thermal_val"
    for img_id in sorted(img_ids):
        anns = [a for a in ann_map[img_id] if a["category_id"] in VALID]
        if not anns:
            continue  # skip background-only images
        fpath = base / id2path[img_id]
        if not fpath.exists():
            continue
        boxes = [[a["bbox"][0], a["bbox"][1],
                  a["bbox"][0]+a["bbox"][2], a["bbox"][1]+a["bbox"][3]] for a in anns]
        labels = [CLASS_MAP[a["category_id"]] for a in anns]
        image_paths.append(str(fpath))
        gt_records.append({"boxes": boxes, "labels": labels})
    return image_paths, gt_records


def _make_predict_fn(det_info: dict):
    """Return a predict function for the given detector config."""
    dtype = det_info["type"]
    weights = det_info["weights"]

    if dtype in ("ultralytics", "ultralytics_rtdetr"):
        from ultralytics import YOLO
        model = YOLO(str(weights))
        def predict(imgs):
            results = []
            for img in imgs:
                if img.ndim == 2:
                    img = np.stack([img]*3, axis=-1)
                elif img.ndim == 3 and img.shape[2] == 1:
                    img = np.concatenate([img]*3, axis=-1)
                r = model.predict(img, conf=0.001, verbose=False)[0]
                boxes = r.boxes.xyxy.cpu().numpy().tolist() if len(r.boxes) else []
                scores = r.boxes.conf.cpu().numpy().tolist() if len(r.boxes) else []
                labels = r.boxes.cls.cpu().numpy().astype(int).tolist() if len(r.boxes) else []
                results.append({"boxes": boxes, "scores": scores, "labels": labels})
            return results
        return predict

    elif dtype in ("frcnn", "retinanet"):
        import torch
        import torchvision
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if dtype == "frcnn":
            from src.detectors.faster_rcnn_wrapper import FasterRCNNDetector
            wrapper = FasterRCNNDetector()
        else:
            from src.detectors.retinanet_wrapper import RetinaNetDetector
            wrapper = RetinaNetDetector()
        state = torch.load(str(weights), map_location=device)
        wrapper.model.load_state_dict(state)
        wrapper.model.to(device)
        wrapper.model.eval()
        def predict(imgs):
            import torchvision.transforms.functional as TF
            results = []
            for img in imgs:
                if img.ndim == 2:
                    img_rgb = np.stack([img]*3, axis=-1)
                elif img.ndim == 3 and img.shape[2] == 1:
                    img_rgb = np.concatenate([img]*3, axis=-1)
                else:
                    img_rgb = img
                tensor = TF.to_tensor(img_rgb).to(device)
                with torch.no_grad():
                    pred = wrapper.model([tensor])[0]
                boxes  = pred["boxes"].cpu().numpy().tolist()
                scores = pred["scores"].cpu().numpy().tolist()
                labels = (pred["labels"].cpu().numpy() - 1).tolist()  # 1-indexed -> 0-indexed
                results.append({"boxes": boxes, "scores": scores, "labels": labels})
            return results
        return predict

    raise ValueError(f"Unknown detector type: {dtype}")


def main():
    print("Loading test split images and GT ...")
    image_paths, gt_records = _get_test_images_and_gt()
    print(f"  {len(image_paths)} test images loaded")

    summary_rows = []
    for det_name, det_info in DETECTORS.items():
        print(f"\n{'='*60}")
        print(f"  Detector: {det_name}")
        print(f"{'='*60}")
        out_dir = REPO / f"results/{det_name}_flir_seed0/eval/corruption"
        predict_fn = _make_predict_fn(det_info)
        results = run_corruption_eval(
            predict_fn,
            image_paths,
            gt_records,
            str(out_dir),
            clean_map=det_info["clean_map"],
        )
        for cname, mce in results["mce"].items():
            summary_rows.append({
                "detector": det_name, "corruption": cname, "mCE": mce,
                **{f"sev{s}": results["corruptions"][cname][str(s)]["mAP50"]
                   for s in [1,2,3,4]},
            })
        print(f"  mean mCE = {results['mean_mce']:.4f}")

    # Summary CSV
    out_csv = REPO / "results/corruption_comparison.csv"
    if summary_rows:
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            w.writeheader(); w.writerows(summary_rows)
        print(f"\nSummary: {out_csv}")


if __name__ == "__main__":
    main()
