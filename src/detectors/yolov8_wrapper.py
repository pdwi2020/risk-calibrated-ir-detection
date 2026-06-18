"""YOLOv8m wrapper for training and inference on thermal datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


class YOLOv8Detector:
    """Thin wrapper around ultralytics YOLO for thermal detection experiments."""

    def __init__(
        self,
        model_path: str = "yolov8m.pt",
        config_path: str = "configs/yolov8.yaml",
    ) -> None:
        from ultralytics import YOLO

        self.model = YOLO(model_path)
        config_file = Path(config_path)
        if config_file.exists():
            with open(config_file) as f:
                self.cfg: Dict[str, Any] = yaml.safe_load(f)
        else:
            self.cfg = {}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, data_yaml: str, **kwargs) -> Dict:
        """Train YOLOv8m on dataset described by data_yaml.

        Args:
            data_yaml: Path to YOLO data.yaml with train/val paths and class names.
            **kwargs: Override any config values (e.g. epochs=50, batch=8).

        Returns:
            Results dict from ultralytics training run.
        """
        train_args = {**self.cfg, "data": data_yaml, **kwargs}
        results = self.model.train(**train_args)
        return results.results_dict

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, data_yaml: str, **kwargs) -> Dict:
        """Evaluate on the validation split defined in data_yaml.

        Returns:
            Metrics dict with keys: metrics/mAP50, metrics/mAP50-95, etc.
        """
        metrics = self.model.val(data=data_yaml, **kwargs)
        return metrics.results_dict

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(
        self,
        images: List,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        **kwargs,
    ) -> List:
        """Run inference on a list of images or file paths.

        Args:
            images: List of image arrays (H,W,C uint8) or file path strings.
            conf_threshold: Minimum confidence to keep a detection.
            iou_threshold: NMS IoU threshold.

        Returns:
            List of ultralytics Results objects with .boxes, .conf, .cls.
        """
        return self.model.predict(
            images,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Uncertainty estimation support
    # ------------------------------------------------------------------

    def enable_mc_dropout(self) -> None:
        """Enable dropout layers in backbone for MC Dropout uncertainty estimation.

        Call before running multiple forward passes to obtain stochastic predictions.
        Must be called once; model.eval() is the default (dropout off) — this
        re-enables dropout only in Dropout modules.
        """
        import torch.nn as nn

        for m in self.model.model.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    def disable_mc_dropout(self) -> None:
        """Return all dropout layers to eval mode (deterministic inference)."""
        import torch.nn as nn

        for m in self.model.model.modules():
            if isinstance(m, nn.Dropout):
                m.eval()

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save model weights."""
        self.model.save(path)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        config_path: str = "configs/yolov8.yaml",
    ) -> "YOLOv8Detector":
        """Load from a saved .pt checkpoint."""
        obj = cls.__new__(cls)
        from ultralytics import YOLO

        obj.model = YOLO(checkpoint_path)
        config_file = Path(config_path)
        if config_file.exists():
            with open(config_file) as f:
                obj.cfg = yaml.safe_load(f)
        else:
            obj.cfg = {}
        return obj
