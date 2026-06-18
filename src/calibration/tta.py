"""Test-Time Augmentation (TTA) for object detectors.

Runs K augmentation passes (hflip × scale) and fuses predictions with
Weighted Box Fusion (WBF).  Unlike MC Dropout, TTA works for any detector
with no dropout layers and is the primary ensemble-diversity mechanism for
the standard detectors in this paper.

Default augmentations: scales ∈ [0.83, 1.0, 1.20] × flip ∈ [False, True] = 6 passes.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import cv2
except ImportError as e:
    raise ImportError(
        "OpenCV is required for TTA augmentation. "
        "Install with: pip install opencv-python-headless"
    ) from e


# ---------------------------------------------------------------------------
# Augmentation helpers
# ---------------------------------------------------------------------------

def _apply_augmentation(image: np.ndarray, scale: float, flip: bool) -> np.ndarray:
    """Return a new array with scale then hflip applied. Input is unchanged."""
    if scale != 1.0:
        h, w = image.shape[:2]
        nw, nh = int(round(w * scale)), int(round(h * scale))
        image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    if flip:
        image = np.ascontiguousarray(image[:, ::-1])
    return image


def _invert_boxes(
    boxes: np.ndarray,
    scale: float,
    flip: bool,
    aug_w: int,
) -> np.ndarray:
    """Map xyxy boxes from augmented-image coordinates back to original coords.

    Augmentation order was: scale → flip, so inversion order is: unflip → unscale.
    """
    if len(boxes) == 0:
        return boxes
    boxes = boxes.astype(np.float32).copy()
    if flip:
        x1 = boxes[:, 0].copy()
        x2 = boxes[:, 2].copy()
        boxes[:, 0] = aug_w - x2
        boxes[:, 2] = aug_w - x1
    if scale != 1.0:
        boxes[:, [0, 2]] /= scale
        boxes[:, [1, 3]] /= scale
    return boxes


# ---------------------------------------------------------------------------
# Weighted Box Fusion
# ---------------------------------------------------------------------------

def _iou_boxes(a: np.ndarray, b: np.ndarray) -> float:
    """IoU of two xyxy boxes."""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter == 0.0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1])
    ub = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (ua + ub - inter + 1e-6)


def weighted_box_fusion(
    all_boxes: List[np.ndarray],
    all_scores: List[float],
    all_labels: List[int],
    iou_thr: float = 0.55,
    skip_box_thr: float = 0.0,
) -> Dict[str, np.ndarray]:
    """Weighted Boxes Fusion: fuse overlapping predictions into weighted means.

    Reference: Solovyev et al. (2021), "Weighted boxes fusion: Ensembling boxes
    from different object detection models" — Pattern Recognition Letters.

    Args:
        all_boxes:    List of (4,) xyxy float boxes from all augmentation passes.
        all_scores:   Corresponding confidence scores.
        all_labels:   Corresponding class-id labels.
        iou_thr:      IoU threshold for grouping overlapping boxes (default 0.55).
        skip_box_thr: Discard boxes below this confidence before fusion.

    Returns:
        Dict with 'boxes' (N,4) float32, 'scores' (N,) float32, 'labels' (N,) int32.
    """
    # Filter below threshold
    keep = [i for i, s in enumerate(all_scores) if s >= skip_box_thr]
    if not keep:
        return {
            "boxes": np.empty((0, 4), dtype=np.float32),
            "scores": np.empty((0,), dtype=np.float32),
            "labels": np.empty((0,), dtype=np.int32),
        }

    boxes = np.stack([all_boxes[i] for i in keep]).astype(np.float64)
    scores = np.array([all_scores[i] for i in keep], dtype=np.float64)
    labels = np.array([all_labels[i] for i in keep], dtype=np.int32)

    order = np.argsort(-scores)
    boxes, scores, labels = boxes[order], scores[order], labels[order]
    used = np.zeros(len(boxes), dtype=bool)

    out_boxes: List[np.ndarray] = []
    out_scores: List[float] = []
    out_labels: List[int] = []

    for i in range(len(boxes)):
        if used[i]:
            continue
        cluster: List[int] = [i]
        used[i] = True
        for j in range(i + 1, len(boxes)):
            if used[j] or labels[j] != labels[i]:
                continue
            if _iou_boxes(boxes[i], boxes[j]) >= iou_thr:
                cluster.append(j)
                used[j] = True

        w = scores[cluster]
        w = w / w.sum()
        fused_box = (boxes[cluster] * w[:, None]).sum(axis=0)
        out_boxes.append(fused_box.astype(np.float32))
        out_scores.append(float(scores[cluster].mean()))
        out_labels.append(int(labels[i]))

    return {
        "boxes": np.stack(out_boxes) if out_boxes else np.empty((0, 4), dtype=np.float32),
        "scores": np.array(out_scores, dtype=np.float32),
        "labels": np.array(out_labels, dtype=np.int32),
    }


# ---------------------------------------------------------------------------
# Reliability diagram helper
# ---------------------------------------------------------------------------

def compute_reliability_bins(
    confidences: Sequence[float],
    is_tp: Sequence[int],
    n_bins: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute reliability diagram data: mean confidence vs empirical precision per bin.

    Args:
        confidences: Per-detection confidence scores.
        is_tp:       1 if the detection is a true positive (IoU ≥ 0.5), else 0.
        n_bins:      Number of equal-width bins over [0, 1].

    Returns:
        (bin_centers, mean_conf, empirical_precision, bin_count)
        Empty bins have NaN for mean_conf and empirical_precision.
    """
    conf = np.asarray(confidences, dtype=np.float64)
    tp = np.asarray(is_tp, dtype=np.float64)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    mean_conf = np.full(n_bins, np.nan)
    precision = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=np.int64)

    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        mask = (conf >= lo) & (conf < hi) if b < n_bins - 1 else (conf >= lo) & (conf <= hi)
        n = int(mask.sum())
        counts[b] = n
        if n > 0:
            mean_conf[b] = conf[mask].mean()
            precision[b] = tp[mask].mean()

    return centers, mean_conf, precision, counts


# ---------------------------------------------------------------------------
# TTADetection
# ---------------------------------------------------------------------------

class TTADetection:
    """Test-Time Augmentation ensemble for object detectors.

    Runs n_passes augmentation passes (scale × hflip combinations) and fuses
    predictions with Weighted Box Fusion.  Interface mirrors MCDropoutDetection
    so both modules are interchangeable in driver scripts.

    Args:
        detector:       Any detector with .predict(images, conf_threshold=…).
        scales:         Scale factors (default [0.83, 1.0, 1.20]).
        use_flip:       Include hflip augmentation (default True).
        iou_thr:        IoU threshold for WBF box grouping (default 0.55).
        skip_box_thr:   Discard boxes below this confidence before WBF.
        conf_thr:       Confidence threshold passed to detector.predict().
    """

    DEFAULT_SCALES: List[float] = [0.83, 1.0, 1.20]

    def __init__(
        self,
        detector,
        scales: Optional[List[float]] = None,
        use_flip: bool = True,
        iou_thr: float = 0.55,
        skip_box_thr: float = 0.0,
        conf_thr: float = 0.25,
    ) -> None:
        self.detector = detector
        self.scales = scales if scales is not None else list(self.DEFAULT_SCALES)
        self.use_flip = use_flip
        self.iou_thr = iou_thr
        self.skip_box_thr = skip_box_thr
        self.conf_thr = conf_thr

        flips = [False, True] if use_flip else [False]
        self.augmentations: List[Tuple[float, bool]] = [
            (s, f) for s in self.scales for f in flips
        ]

    # ------------------------------------------------------------------

    @property
    def n_passes(self) -> int:
        return len(self.augmentations)

    # ------------------------------------------------------------------

    def predict(
        self,
        images: List[np.ndarray],
    ) -> Tuple[List[Dict[str, np.ndarray]], np.ndarray]:
        """Run TTA inference and fuse predictions with WBF.

        Args:
            images: List of (H,W,C) uint8 numpy arrays.
                    File paths are not supported — augmentation requires pixel arrays.

        Returns:
            (fused_detections, aug_consistency)
            - fused_detections:  List[dict] with 'boxes','scores','labels'.
            - aug_consistency:   (N_images,) float32 — fraction of augmentation
                passes that produced at least one detection.  1.0 = all passes
                detected something; lower = detector fires inconsistently across views.
        """
        n_images = len(images)
        images_np = [np.asarray(img) for img in images]
        orig_shapes = [(img.shape[0], img.shape[1]) for img in images_np]

        # aug_preds[aug_idx][img_idx] = parsed dict with boxes in original coords
        aug_preds: List[List[Dict[str, np.ndarray]]] = []

        for scale, flip in self.augmentations:
            aug_batch = [_apply_augmentation(img, scale, flip) for img in images_np]
            raw_results = self.detector.predict(aug_batch, conf_threshold=self.conf_thr)

            aug_img_preds: List[Dict[str, np.ndarray]] = []
            for img_idx, result in enumerate(raw_results):
                orig_w = orig_shapes[img_idx][1]
                aug_w = int(round(orig_w * scale))
                d = self._parse_result(result)
                d["boxes"] = _invert_boxes(d["boxes"], scale, flip, aug_w)
                aug_img_preds.append(d)
            aug_preds.append(aug_img_preds)

        fused: List[Dict[str, np.ndarray]] = []
        aug_consistency = np.zeros(n_images, dtype=np.float32)

        for img_idx in range(n_images):
            all_b: List[np.ndarray] = []
            all_s: List[float] = []
            all_l: List[int] = []
            passes_with_det = 0

            for aug_idx in range(self.n_passes):
                d = aug_preds[aug_idx][img_idx]
                if len(d["boxes"]) > 0:
                    passes_with_det += 1
                for b, s, l in zip(d["boxes"], d["scores"], d["labels"]):
                    all_b.append(b)
                    all_s.append(float(s))
                    all_l.append(int(l))

            fused.append(
                weighted_box_fusion(all_b, all_s, all_l,
                                    iou_thr=self.iou_thr,
                                    skip_box_thr=self.skip_box_thr)
            )
            aug_consistency[img_idx] = passes_with_det / self.n_passes

        return fused, aug_consistency

    # ------------------------------------------------------------------
    @staticmethod
    def _parse_result(raw_result) -> Dict[str, np.ndarray]:
        """Convert a detector result to a plain dict (same logic as mc_dropout.py)."""
        if isinstance(raw_result, dict):
            return {
                "boxes": np.asarray(raw_result.get("boxes", np.empty((0, 4))), dtype=np.float32),
                "scores": np.asarray(raw_result.get("scores", np.empty((0,))), dtype=np.float32),
                "labels": np.asarray(raw_result.get("labels", np.empty((0,))), dtype=np.int32),
            }
        try:
            boxes_obj = raw_result.boxes
            if boxes_obj is None or len(boxes_obj) == 0:
                return {
                    "boxes": np.empty((0, 4), dtype=np.float32),
                    "scores": np.empty((0,), dtype=np.float32),
                    "labels": np.empty((0,), dtype=np.int32),
                }
            return {
                "boxes": boxes_obj.xyxy.cpu().numpy().astype(np.float32),
                "scores": boxes_obj.conf.cpu().numpy().astype(np.float32),
                "labels": boxes_obj.cls.cpu().numpy().astype(np.int32),
            }
        except AttributeError:
            return {
                "boxes": np.empty((0, 4), dtype=np.float32),
                "scores": np.empty((0,), dtype=np.float32),
                "labels": np.empty((0,), dtype=np.int32),
            }


# ---------------------------------------------------------------------------
# Self-test: python3 src/calibration/tta.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("TTA self-test — synthetic augmented predictions")

    rng = np.random.default_rng(42)

    class _MockDetector:
        """Returns two boxes (with jitter) at fixed relative positions."""
        def predict(self, images, conf_threshold=0.25):
            results = []
            for img in images:
                h, w = img.shape[:2]
                boxes = np.array([
                    [w * 0.3, h * 0.2, w * 0.6, h * 0.8],   # large box, conf 0.85
                    [w * 0.8, h * 0.1, w * 0.95, h * 0.3],  # small box, conf 0.30
                ], dtype=np.float32)
                boxes += rng.uniform(-3.0, 3.0, boxes.shape).astype(np.float32)
                scores = np.array([0.85, 0.30], dtype=np.float32)
                labels = np.array([0, 1], dtype=np.int32)
                mask = scores >= conf_threshold
                results.append({
                    "boxes": boxes[mask],
                    "scores": scores[mask],
                    "labels": labels[mask],
                })
            return results

    tta = TTADetection(_MockDetector(), scales=[0.83, 1.0, 1.20], use_flip=True)
    print(f"  Augmentations ({tta.n_passes} passes): {tta.augmentations}")

    img = np.zeros((480, 640, 3), dtype=np.uint8)
    fused, consistency = tta.predict([img, img])

    print(f"\n  Image 0: {len(fused[0]['boxes'])} fused boxes")
    for i in range(len(fused[0]["boxes"])):
        b = fused[0]["boxes"][i]
        print(f"    box {i}: [{b[0]:.1f},{b[1]:.1f},{b[2]:.1f},{b[3]:.1f}]  "
              f"score={fused[0]['scores'][i]:.3f}  label={fused[0]['labels'][i]}")
    print(f"  aug_consistency[0] = {consistency[0]:.2f} "
          f"(expect 1.0 — all passes detect the 0.85-conf box)")

    assert len(fused[0]["boxes"]) >= 1, "Expected at least one fused box"
    assert consistency[0] > 0.8, f"Low consistency: {consistency[0]}"

    # Box inversion round-trip: scale=1.2 + hflip should recover original coords
    orig_box = np.array([[100.0, 50.0, 300.0, 400.0]], dtype=np.float32)
    orig_w = 640
    scale, flip = 1.2, True
    aug_w = int(round(orig_w * scale))   # 768
    # Forward transform: scale
    scaled = orig_box * scale
    # Forward transform: hflip in aug_w space
    flipped = scaled.copy()
    flipped[:, 0] = aug_w - scaled[:, 2]
    flipped[:, 2] = aug_w - scaled[:, 0]
    # Invert
    recovered = _invert_boxes(flipped, scale=scale, flip=flip, aug_w=aug_w)
    assert np.allclose(recovered, orig_box, atol=1e-3), (
        f"Box inversion failed: {recovered} != {orig_box}"
    )
    print("\n  Box inversion round-trip (scale=1.2 + hflip): OK")

    # Reliability diagram
    conf_vals = list(np.linspace(0.05, 0.95, 200))
    # Toy oracle: a detection is TP if its confidence > 0.5
    tp_vals = [1 if c > 0.5 else 0 for c in conf_vals]
    centers, mean_c, prec, counts = compute_reliability_bins(conf_vals, tp_vals, n_bins=10)
    print("\n  Reliability bins (10):")
    for b in range(10):
        if counts[b] > 0:
            print(f"    [{centers[b]:.2f}]  mean_conf={mean_c[b]:.3f}  "
                  f"prec={prec[b]:.3f}  n={counts[b]}")
    # Bins below 0.5 should have prec ≈ 0; bins above 0.5 should have prec ≈ 1
    low_bins = [b for b in range(10) if not np.isnan(prec[b]) and centers[b] < 0.5]
    high_bins = [b for b in range(10) if not np.isnan(prec[b]) and centers[b] > 0.5]
    assert all(prec[b] < 0.3 for b in low_bins), "Low-conf bins should have low precision"
    assert all(prec[b] > 0.7 for b in high_bins), "High-conf bins should have high precision"
    print("  Reliability bins: OK")

    print("\nAll assertions passed.")
    sys.exit(0)
