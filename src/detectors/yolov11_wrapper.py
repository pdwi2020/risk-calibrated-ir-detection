"""YOLOv11m wrapper — identical API to yolov8_wrapper; uses yolo11m.pt."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import yaml


class YOLOv11Detector:
    """Thin wrapper around ultralytics YOLO for YOLOv11m on thermal datasets."""

    def __init__(
        self,
        model_path: str = "yolo11m.pt",
        config_path: str = "configs/yolov11.yaml",
    ) -> None:
        from ultralytics import YOLO

        self.model = YOLO(model_path)
        config_file = Path(config_path)
        self.cfg: Dict[str, Any] = yaml.safe_load(config_file.read_text()) if config_file.exists() else {}

    def train(self, data_yaml: str, **kwargs) -> Dict:
        train_args = {**self.cfg, "data": data_yaml, **kwargs}
        results = self.model.train(**train_args)
        return results.results_dict

    def evaluate(self, data_yaml: str, **kwargs) -> Dict:
        metrics = self.model.val(data=data_yaml, **kwargs)
        return metrics.results_dict

    def predict(
        self,
        images: List,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        **kwargs,
    ) -> List:
        return self.model.predict(
            images,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False,
            **kwargs,
        )

    def save(self, path: str) -> None:
        self.model.save(path)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        config_path: str = "configs/yolov11.yaml",
    ) -> "YOLOv11Detector":
        obj = cls.__new__(cls)
        from ultralytics import YOLO

        obj.model = YOLO(checkpoint_path)
        config_file = Path(config_path)
        obj.cfg = yaml.safe_load(config_file.read_text()) if config_file.exists() else {}
        return obj
