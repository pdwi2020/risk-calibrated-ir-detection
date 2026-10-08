"""FLIR ADAS v2 thermal dataset loader.

Ground-truth annotation format: COCO JSON at images_thermal_{split}/coco.json
Classes kept: person (FLIR cat_id=1), bike (cat_id=2), car (cat_id=3)
Internal class indices: person=0, bike=1, car=2
Image size: 640×512 JPEG under images_thermal_{split}/data/
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2


class FLIRDataset(Dataset):
    """FLIR ADAS v2 thermal image dataset for object detection.

    Annotation format: COCO JSON (coco.json per split)
    Classes: 0=person, 1=bike, 2=car  (cyclist/dog/etc. excluded)
    Image size: 640×512 thermal JPEG, resized to 640×640 for YOLOv8
    """

    # FLIR COCO category names → internal 0-based class index
    KEEP_CLASSES: Dict[str, int] = {"person": 0, "bike": 1, "car": 2}
    CLASS_NAMES: List[str] = ["person", "bike", "car"]

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[A.Compose] = None,
        return_format: str = "yolo",
    ) -> None:
        """
        Args:
            root: Path to FLIR_ADAS_v2 directory (contains images_thermal_train/, etc.)
            split: 'train' or 'val'
            transform: Albumentations Compose pipeline (must include bbox_params).
            return_format: 'yolo' or 'coco' — controls box format in target dict.
        """
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.return_format = return_format
        self._load_annotations()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _split_dir(self) -> str:
        return "images_thermal_train" if self.split == "train" else "images_thermal_val"

    def _load_annotations(self) -> None:
        """Parse coco.json; filter to KEEP_CLASSES; build image → annotations index."""
        ann_file = self.root / self._split_dir() / "coco.json"
        if not ann_file.exists():
            raise FileNotFoundError(f"Annotation file not found: {ann_file}")

        with open(ann_file) as f:
            coco = json.load(f)

        # Map FLIR COCO category_id → internal class index
        self.cat_id_to_idx: Dict[int, int] = {}
        for cat in coco["categories"]:
            if cat["name"] in self.KEEP_CLASSES:
                self.cat_id_to_idx[cat["id"]] = self.KEEP_CLASSES[cat["name"]]

        # Image directory (actual JPEG files live here)
        self.image_dir = self.root / self._split_dir() / "data"

        # id → filename (strip subdirs prefix, keep only basename)
        id_to_fname: Dict[int, str] = {
            img["id"]: Path(img["file_name"]).name
            for img in coco["images"]
        }

        # Build annotation index filtered to KEEP_CLASSES
        ann_by_image: Dict[int, List] = defaultdict(list)
        for ann in coco["annotations"]:
            if ann["category_id"] in self.cat_id_to_idx:
                ann_by_image[ann["image_id"]].append(ann)

        # Ordered list of (image_id, filename) and annotation lookup
        self.images: List[Tuple[int, str]] = []
        self.annotations: Dict[int, List] = {}
        for img in coco["images"]:
            iid = img["id"]
            self.images.append((iid, id_to_fname[iid]))
            self.annotations[iid] = ann_by_image[iid]

        # Store image dims for YOLO normalization
        self._img_dims: Dict[int, Tuple[int, int]] = {
            img["id"]: (img["width"], img["height"])
            for img in coco["images"]
        }

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        """Return (image_tensor, target_dict).

        target_dict keys:
            'boxes'    — (N, 4) float32 xyxy (pixels) or xywhn if return_format='yolo'
            'labels'   — (N,) int64 class indices
            'image_id' — int
        """
        iid, fname = self.images[idx]
        img_path = self.image_dir / fname

        # Load as RGB (thermal JPEGs are grayscale-valued but stored as uint8 JPEG)
        img = np.array(Image.open(img_path).convert("RGB"))
        H, W = img.shape[:2]

        anns = self.annotations[iid]

        # COCO bbox: [x_min, y_min, width, height] → xyxy
        boxes: List[List[float]] = []
        labels: List[int] = []
        for ann in anns:
            x, y, bw, bh = ann["bbox"]
            # Clamp to image bounds
            x1 = max(0.0, float(x))
            y1 = max(0.0, float(y))
            x2 = min(float(W), float(x + bw))
            y2 = min(float(H), float(y + bh))
            if x2 > x1 and y2 > y1:  # skip degenerate boxes
                boxes.append([x1, y1, x2, y2])
                labels.append(self.cat_id_to_idx[ann["category_id"]])

        boxes_arr = np.array(boxes, dtype=np.float32).reshape(-1, 4)
        labels_arr = np.array(labels, dtype=np.int64)

        if self.transform is not None:
            result = self.transform(
                image=img,
                bboxes=boxes_arr.tolist(),
                class_labels=labels_arr.tolist(),
            )
            img_tensor: torch.Tensor = result["image"]
            boxes_arr = np.array(result["bboxes"], dtype=np.float32).reshape(-1, 4)
            labels_arr = np.array(result["class_labels"], dtype=np.int64)
        else:
            # Default: resize to 640×640, normalize to [0,1], CHW
            img_resized = np.array(
                Image.fromarray(img).resize((640, 640), Image.BILINEAR)
            )
            img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
            # Scale boxes to new image size
            if len(boxes_arr):
                scale_x = 640.0 / W
                scale_y = 640.0 / H
                boxes_arr[:, [0, 2]] *= scale_x
                boxes_arr[:, [1, 3]] *= scale_y

        if self.return_format == "yolo":
            # Convert xyxy to normalized xywh (relative to image after transform)
            _h, _w = (640, 640)  # after resize
            if len(boxes_arr):
                cx = (boxes_arr[:, 0] + boxes_arr[:, 2]) / 2 / _w
                cy = (boxes_arr[:, 1] + boxes_arr[:, 3]) / 2 / _h
                bw_ = (boxes_arr[:, 2] - boxes_arr[:, 0]) / _w
                bh_ = (boxes_arr[:, 3] - boxes_arr[:, 1]) / _h
                boxes_arr = np.stack([cx, cy, bw_, bh_], axis=1).astype(np.float32)

        target = {
            "boxes": torch.tensor(boxes_arr),
            "labels": torch.tensor(labels_arr),
            "image_id": iid,
        }
        return img_tensor, target

    # ------------------------------------------------------------------
    # Class statistics helpers
    # ------------------------------------------------------------------

    def class_counts(self) -> Dict[str, int]:
        """Count annotations per class across the entire split."""
        counts: Dict[str, int] = {name: 0 for name in self.CLASS_NAMES}
        idx_to_name = {v: k for k, v in self.KEEP_CLASSES.items()}
        for iid, _ in self.images:
            for ann in self.annotations[iid]:
                cidx = self.cat_id_to_idx.get(ann["category_id"])
                if cidx is not None:
                    counts[idx_to_name[cidx]] += 1
        return counts

    def total_annotations(self) -> int:
        return sum(len(self.annotations[iid]) for iid, _ in self.images)

    # ------------------------------------------------------------------
    # Augmentation presets
    # ------------------------------------------------------------------

    @staticmethod
    def get_default_transform(train: bool = True) -> A.Compose:
        """Albumentations augmentation pipeline for thermal images."""
        bbox_params = A.BboxParams(
            format="pascal_voc",
            label_fields=["class_labels"],
            min_visibility=0.1,
        )
        if train:
            return A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.RandomScale(scale_limit=0.1, p=0.5),
                    A.RandomBrightnessContrast(
                        brightness_limit=0.1, contrast_limit=0.1, p=0.3
                    ),
                    A.Resize(640, 640),
                    ToTensorV2(),
                ],
                bbox_params=bbox_params,
            )
        return A.Compose(
            [A.Resize(640, 640), ToTensorV2()],
            bbox_params=bbox_params,
        )
