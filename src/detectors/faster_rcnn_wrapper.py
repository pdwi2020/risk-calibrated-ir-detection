"""Faster R-CNN wrapper using torchvision for FLIR thermal detection."""

import json
import tempfile
from typing import Dict, List

import torch
from torchvision.models.detection import FasterRCNN, fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import yaml

# ---------------------------------------------------------------------------
# Label-mapping convention (torchvision reserves 0 for background)
# ---------------------------------------------------------------------------
# FLIR internal indices (from FLIRDataset): person=0, bike=1, car=2
# Torchvision foreground indices:            person=1, bike=2, car=3
#   → train/evaluate: target['labels'] += 1  (dataset → model)
#   → predict:        output['labels'] -= 1  (model → dataset)
# n_classes must therefore be 4  (background + 3 foreground)
# ---------------------------------------------------------------------------


class FasterRCNNDetector:
    """ResNet-50 FPN Faster R-CNN fine-tuned on thermal detection datasets."""

    def __init__(
        self,
        n_classes: int = 4,
        config_path: str = "configs/faster_rcnn.yaml",
        pretrained: bool = True,
    ):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._build_model(n_classes, pretrained)
        self.model.to(self.device)

    def _build_model(self, n_classes: int, pretrained: bool) -> FasterRCNN:
        """Build Faster R-CNN with custom head for n_classes (including background)."""
        model = fasterrcnn_resnet50_fpn(pretrained=pretrained)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, n_classes)
        return model

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_one_epoch(
        self,
        dataloader,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        """Train for one epoch. Returns mean loss over all batches.

        Label mapping: FLIRDataset emits labels in {0,1,2}.  Torchvision
        Faster R-CNN treats 0 as background, so foreground labels must be
        in {1, 2, 3}.  We shift here: label += 1.
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for images, targets in dataloader:
            # Move images to device; each image is a float Tensor [C, H, W] in [0,1]
            images = [img.to(self.device) for img in images]

            # Build per-image target dicts expected by torchvision:
            #   'boxes'  : FloatTensor [N, 4] xyxy absolute pixels
            #   'labels' : Int64Tensor [N]   foreground ids in [1..n_classes-1]
            tv_targets = []
            for t in targets:
                tv_targets.append(
                    {
                        # boxes come from FLIRDataset as xyxy float32 — keep as-is
                        "boxes": t["boxes"].to(self.device, dtype=torch.float32),
                        # Shift dataset labels {0,1,2} → torchvision labels {1,2,3}
                        "labels": (t["labels"] + 1).to(self.device, dtype=torch.int64),
                    }
                )

            optimizer.zero_grad()

            # Forward pass in train mode returns a dict of scalar losses
            loss_dict = self.model(images, tv_targets)
            loss = sum(loss_dict.values())

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, dataloader) -> Dict:
        """Evaluate using COCO mAP metrics via pycocotools.

        Returns:
            Dict with at minimum:
                'mAP50-95' : float  (COCOeval.stats[0])
                'mAP50'    : float  (COCOeval.stats[1])

        Label mapping: predictions come out of the model with labels in
        {1,2,3}.  We convert them to category_id {1,2,3} for pycocotools
        (no further shift needed since COCO category_ids are 1-based).
        Ground-truth labels from FLIRDataset are in {0,1,2}; we shift
        to {1,2,3} here as well so that GT and DT share the same ID space.
        """
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        self.model.eval()

        # -------------------------------------------------------------------
        # Accumulators for COCO-format GT and detections
        # -------------------------------------------------------------------
        # GT structure expected by pycocotools:
        #   {
        #     "images":      [{"id": image_id}, ...],
        #     "annotations": [{"id", "image_id", "category_id", "bbox":[x,y,w,h], "area", "iscrowd"}, ...],
        #     "categories":  [{"id": 1, "name": "person"}, {"id": 2, ...}, {"id": 3, ...}],
        #   }
        gt_images = []       # list of {"id": image_id}
        gt_annotations = []  # list of COCO annotation dicts
        dt_list = []         # list of detection result dicts

        ann_id = 0  # running annotation id for GT (must be unique)

        with torch.no_grad():
            for images, targets in dataloader:
                images = [img.to(self.device) for img in images]

                # -------------------------------------------------------
                # Build torchvision targets (needed only for image_id here,
                # but we also read GT boxes/labels for the COCO GT dict).
                # -------------------------------------------------------
                tv_targets = []
                for t in targets:
                    tv_targets.append(
                        {
                            "boxes": t["boxes"].to(self.device, dtype=torch.float32),
                            # Shift {0,1,2} → {1,2,3} so GT category_ids are 1-based
                            "labels": (t["labels"] + 1).to(self.device, dtype=torch.int64),
                        }
                    )

                # -------------------------------------------------------
                # Forward pass — eval mode returns list of prediction dicts
                # -------------------------------------------------------
                outputs = self.model(images)

                # -------------------------------------------------------
                # Populate GT and DT for each image in the batch
                # -------------------------------------------------------
                for output, tv_tgt, raw_tgt in zip(outputs, tv_targets, targets):
                    image_id = int(raw_tgt["image_id"])

                    # -- Ground-truth side --
                    gt_images.append({"id": image_id})

                    gt_boxes = tv_tgt["boxes"].cpu()    # xyxy, absolute pixels
                    gt_labels = tv_tgt["labels"].cpu()  # category_id in {1,2,3}

                    for box, cat_id in zip(gt_boxes, gt_labels):
                        x1, y1, x2, y2 = box.tolist()
                        w = x2 - x1
                        h = y2 - y1
                        gt_annotations.append(
                            {
                                "id": ann_id,
                                "image_id": image_id,
                                "category_id": int(cat_id),
                                # COCO annotation bbox format: [x, y, width, height]
                                "bbox": [x1, y1, w, h],
                                "area": float(w * h),
                                "iscrowd": 0,
                            }
                        )
                        ann_id += 1

                    # -- Detection side --
                    pred_boxes = output["boxes"].cpu()    # xyxy
                    pred_scores = output["scores"].cpu()
                    pred_labels = output["labels"].cpu()  # model output labels in {1,2,3}

                    for box, score, cat_id in zip(pred_boxes, pred_scores, pred_labels):
                        x1, y1, x2, y2 = box.tolist()
                        w = x2 - x1
                        h = y2 - y1
                        dt_list.append(
                            {
                                "image_id": image_id,
                                # category_id stays in {1,2,3} — matches GT convention above
                                "category_id": int(cat_id),
                                # COCO detection bbox format: [x, y, width, height]
                                "bbox": [x1, y1, w, h],
                                "score": float(score),
                            }
                        )

        # -------------------------------------------------------------------
        # Build pycocotools COCO objects from dicts (no temp-file I/O needed)
        # -------------------------------------------------------------------
        # Category list: ids 1, 2, 3 matching the torchvision foreground label space
        categories = [
            {"id": 1, "name": "person", "supercategory": "none"},
            {"id": 2, "name": "bike",   "supercategory": "none"},
            {"id": 3, "name": "car",    "supercategory": "none"},
        ]

        gt_dict = {
            "images":      gt_images,
            "annotations": gt_annotations,
            "categories":  categories,
        }

        # Build COCO GT object by loading the dict directly (avoids file I/O)
        cocoGt = COCO()
        cocoGt.dataset = gt_dict
        cocoGt.createIndex()

        # Build COCO DT object via the standard loadRes interface;
        # loadRes requires a non-empty list so we guard for edge cases.
        if dt_list:
            cocoDt = cocoGt.loadRes(dt_list)
        else:
            # No detections at all — return zeros rather than crashing
            return {"mAP50-95": 0.0, "mAP50": 0.0}

        # -------------------------------------------------------------------
        # Run COCOeval (IoU type 'bbox')
        # -------------------------------------------------------------------
        coco_eval = COCOeval(cocoGt, cocoDt, iouType="bbox")
        coco_eval.evaluate()    # Per-image per-category IoU matching
        coco_eval.accumulate()  # Build precision-recall curve
        coco_eval.summarize()   # Print summary table; fills coco_eval.stats

        # stats[0] = mAP @[IoU=0.50:0.95] (primary metric)
        # stats[1] = mAP @[IoU=0.50]       (PASCAL VOC-style)
        return {
            "mAP50-95": float(coco_eval.stats[0]),
            "mAP50":    float(coco_eval.stats[1]),
        }

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(
        self,
        images: List[torch.Tensor],
        score_threshold: float = 0.5,
    ) -> List[Dict]:
        """Run inference; filter by score_threshold.

        Label mapping: model outputs labels in {1,2,3}; we subtract 1
        to return dataset-convention labels {0,1,2} (person/bike/car).

        Args:
            images: List of float Tensors [C, H, W] in [0, 1].
            score_threshold: Minimum score to include a detection.

        Returns:
            List of dicts (one per image) with keys:
                'boxes'  : FloatTensor [N, 4] xyxy absolute pixels
                'scores' : FloatTensor [N]
                'labels' : Int64Tensor [N]   in {0,1,2} (dataset convention)
        """
        self.model.eval()
        images = [img.to(self.device) for img in images]

        with torch.no_grad():
            outputs = self.model(images)

        results = []
        for output in outputs:
            scores = output["scores"]
            keep = scores >= score_threshold

            results.append(
                {
                    "boxes":  output["boxes"][keep].cpu(),
                    "scores": scores[keep].cpu(),
                    # Shift {1,2,3} → {0,1,2} to match FLIRDataset label convention
                    "labels": (output["labels"][keep] - 1).cpu(),
                }
            )
        return results
