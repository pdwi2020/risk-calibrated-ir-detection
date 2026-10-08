"""LLVIP thermal dataset loader (YOLO format).

Actual directory layout on disk:
    LLVIP-YOLO/
        {split}/
            lwir/
                images/   ← IR images (*.jpg) — 12025 train / 3463 test
            visible/
                images/   ← paired RGB images
                labels/   ← YOLO txt annotations (shared for both modalities)

The labels live under visible/labels/ even for the IR modality.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2


class LLVIPDataset(Dataset):
    """LLVIP infrared pedestrian dataset.

    Annotation format: YOLO txt — one file per image:
        class_id  cx  cy  w  h   (all normalized to [0,1])
    Classes: 0=person (single class)
    IR image path:  LLVIP-YOLO/{split}/lwir/images/{id}.jpg
    Label path:     LLVIP-YOLO/{split}/visible/labels/{id}.txt
    """

    CLASS_NAMES: List[str] = ["person"]

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[A.Compose] = None,
        use_visible: bool = False,
    ) -> None:
        """
        Args:
            root: Path to the llvip dataset directory (contains LLVIP-YOLO/).
            split: 'train' or 'test'
            transform: Albumentations Compose pipeline (must include bbox_params).
            use_visible: If True, load visible-spectrum image instead of IR.
        """
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.use_visible = use_visible
        self._load_file_list()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _yolo_root(self) -> Path:
        return self.root / "LLVIP-YOLO" / self.split

    def _load_file_list(self) -> None:
        """Build sorted list of (img_path, lbl_path) pairs.

        IR images: lwir/images/*.jpg
        Labels:    visible/labels/*.txt  (same filename stem)
        """
        yolo_root = self._yolo_root()
        img_dir = yolo_root / "lwir" / "images"
        lbl_dir = yolo_root / "visible" / "labels"

        if not img_dir.exists():
            raise FileNotFoundError(f"IR image directory not found: {img_dir}")

        self.samples: List[Tuple[Path, Optional[Path]]] = []
        for img_path in sorted(img_dir.glob("*.jpg")):
            if img_path.name.startswith("."):  # skip macOS ._* shadow files
                continue
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            if lbl_path.exists():
                self.samples.append((img_path, lbl_path))
            else:
                # Test set may have no labels — include anyway (inference mode)
                self.samples.append((img_path, None))

        if not self.samples:
            raise RuntimeError(f"No images found in {img_dir}")

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        """Return (image_tensor, target_dict).

        target_dict keys:
            'boxes'    — (N, 4) float32 xyxy pixels
            'labels'   — (N,) int64 (all 0 = person)
            'image_id' — str stem, e.g. '010001'
        """
        img_path, lbl_path = self.samples[idx]

        if self.use_visible:
            vis_path = (
                self._yolo_root() / "visible" / "images" / img_path.name
            )
            img = np.array(Image.open(vis_path).convert("RGB"))
        else:
            img = np.array(Image.open(img_path).convert("RGB"))

        H, W = img.shape[:2]

        boxes: List[List[float]] = []
        labels: List[int] = []

        if lbl_path is not None and lbl_path.exists():
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    cls_id, cx, cy, bw, bh = map(float, parts)
                    # YOLO normalized → xyxy pixel coords
                    x1 = (cx - bw / 2) * W
                    y1 = (cy - bh / 2) * H
                    x2 = (cx + bw / 2) * W
                    y2 = (cy + bh / 2) * H
                    x1, y1 = max(0.0, x1), max(0.0, y1)
                    x2, y2 = min(float(W), x2), min(float(H), y2)
                    if x2 > x1 and y2 > y1:
                        boxes.append([x1, y1, x2, y2])
                        labels.append(int(cls_id))

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
            img_resized = np.array(
                Image.fromarray(img).resize((640, 640), Image.BILINEAR)
            )
            img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
            if len(boxes_arr):
                scale_x = 640.0 / W
                scale_y = 640.0 / H
                boxes_arr[:, [0, 2]] *= scale_x
                boxes_arr[:, [1, 3]] *= scale_y

        target = {
            "boxes": torch.tensor(boxes_arr),
            "labels": torch.tensor(labels_arr),
            "image_id": img_path.stem,
        }
        return img_tensor, target

    # ------------------------------------------------------------------
    # Statistics helpers
    # ------------------------------------------------------------------

    def total_annotations(self) -> int:
        """Count all bounding boxes across the split (reads all label files)."""
        total = 0
        for _, lbl_path in self.samples:
            if lbl_path and lbl_path.exists():
                with open(lbl_path) as f:
                    total += sum(1 for ln in f if len(ln.strip().split()) == 5)
        return total

    # ------------------------------------------------------------------
    # Augmentation presets
    # ------------------------------------------------------------------

    @staticmethod
    def get_default_transform(train: bool = True) -> A.Compose:
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
                    A.Resize(640, 640),
                    ToTensorV2(),
                ],
                bbox_params=bbox_params,
            )
        return A.Compose(
            [A.Resize(640, 640), ToTensorV2()],
            bbox_params=bbox_params,
        )
