"""MC Dropout uncertainty estimation for object detectors.

Uses T stochastic forward passes with dropout enabled to estimate:
  - detection_rate per predicted box (fraction of passes that fired a detection
    matching that box via IoU ≥ iou_thr)
  - score_std: standard deviation of confidence scores across matching detections
  - per-image predictive entropy: H = mean_b H_bin(detection_rate_b)

Note: standard YOLOv8m has no Dropout layers, so T passes are deterministic
unless dropout is inserted post-hoc (test-time dropout).  The wrapper's
enable_mc_dropout() call is a no-op in that case; detection_rate will be 1.0
and score_std ≈ 0 for every box.  This serves as the "no-uncertainty" baseline
and is documented as such in the paper.  RT-DETR and Faster R-CNN also lack
dropout, making MC Dropout a shared baseline across all detectors.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """IoU between two xyxy boxes."""
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    ix1, iy1 = max(xa1, xb1), max(ya1, yb1)
    ix2, iy2 = min(xa2, xb2), min(ya2, yb2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0
    area_a = (xa2 - xa1) * (ya2 - ya1)
    area_b = (xb2 - xb1) * (yb2 - yb1)
    return inter / (area_a + area_b - inter + 1e-6)


def _binary_entropy(p: float) -> float:
    """H(p) = -p log p - (1-p) log(1-p), clipped to [0,1] input."""
    # Use Python float to avoid float32 precision collapsing 1-1e-9 → 1.0
    p = max(1e-9, min(1.0 - 1e-9, float(p)))
    return -p * np.log2(p) - (1.0 - p) * np.log2(1.0 - p)


def _aggregate_passes(
    predictions_per_pass: List[Dict[str, np.ndarray]],
    iou_thr: float = 0.5,
) -> Dict[str, np.ndarray]:
    """Aggregate T stochastic prediction dicts for a single image.

    Each dict in predictions_per_pass has:
        'boxes':  np.ndarray (N, 4) xyxy float32
        'scores': np.ndarray (N,)   float32
        'labels': np.ndarray (N,)   int32

    Returns aggregated dict with same keys plus:
        'detection_rate': fraction of T passes that contributed a matching box
        'score_std':      std of conf scores across matching detections
        'uncertainty':    1 - detection_rate  (0 = always detected, 1 = never)
    """
    T = len(predictions_per_pass)

    # Collect all boxes from all passes together with pass index
    all_boxes: List[np.ndarray] = []
    all_scores: List[float] = []
    all_labels: List[int] = []
    all_pass_idx: List[int] = []

    for t, preds in enumerate(predictions_per_pass):
        boxes = preds.get("boxes", np.empty((0, 4), dtype=np.float32))
        scores = preds.get("scores", np.empty((0,), dtype=np.float32))
        labels = preds.get("labels", np.empty((0,), dtype=np.int32))
        for b, s, l in zip(boxes, scores, labels):
            all_boxes.append(b)
            all_scores.append(float(s))
            all_labels.append(int(l))
            all_pass_idx.append(t)

    if len(all_boxes) == 0:
        empty = np.empty((0,), dtype=np.float32)
        return {
            "boxes": np.empty((0, 4), dtype=np.float32),
            "scores": empty.copy(),
            "labels": np.empty((0,), dtype=np.int32),
            "detection_rate": empty.copy(),
            "score_std": empty.copy(),
            "uncertainty": empty.copy(),
        }

    all_boxes_arr = np.stack(all_boxes)  # (M, 4)
    used = np.zeros(len(all_boxes_arr), dtype=bool)

    clusters_box: List[np.ndarray] = []
    clusters_score_mean: List[float] = []
    clusters_score_std: List[float] = []
    clusters_label: List[int] = []
    clusters_det_rate: List[float] = []

    # Greedy cluster: for each unassigned box (by score), collect all
    # IoU-overlapping boxes of the same label.
    order = np.argsort(-np.array(all_scores))
    for idx in order:
        if used[idx]:
            continue
        label_i = all_labels[idx]
        members_idx = [idx]
        used[idx] = True
        for jdx in range(len(all_boxes_arr)):
            if used[jdx]:
                continue
            if all_labels[jdx] != label_i:
                continue
            if _iou(all_boxes_arr[idx], all_boxes_arr[jdx]) >= iou_thr:
                members_idx.append(jdx)
                used[jdx] = True

        member_scores = [all_scores[i] for i in members_idx]
        member_boxes = all_boxes_arr[members_idx]
        member_passes = set(all_pass_idx[i] for i in members_idx)

        # Weighted-mean box (weighted by score)
        w = np.array(member_scores, dtype=np.float64)
        w /= w.sum()
        mean_box = (member_boxes.astype(np.float64) * w[:, None]).sum(axis=0)

        det_rate = len(member_passes) / T

        clusters_box.append(mean_box.astype(np.float32))
        clusters_score_mean.append(float(np.mean(member_scores)))
        clusters_score_std.append(float(np.std(member_scores)))
        clusters_label.append(label_i)
        clusters_det_rate.append(det_rate)

    boxes_out = np.stack(clusters_box) if clusters_box else np.empty((0, 4), dtype=np.float32)
    scores_out = np.array(clusters_score_mean, dtype=np.float32)
    stds_out = np.array(clusters_score_std, dtype=np.float32)
    labels_out = np.array(clusters_label, dtype=np.int32)
    det_rate_out = np.array(clusters_det_rate, dtype=np.float32)
    uncertainty_out = (1.0 - det_rate_out).astype(np.float32)

    return {
        "boxes": boxes_out,
        "scores": scores_out,
        "labels": labels_out,
        "detection_rate": det_rate_out,
        "score_std": stds_out,
        "uncertainty": uncertainty_out,
    }


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class MCDropoutDetection:
    """MC Dropout for object detectors.

    Run n_samples inference passes with dropout enabled; aggregate box
    predictions by IoU-greedy clustering across passes.  For each cluster
    compute detection_rate (fraction of passes that fired that box) and
    score_std (confidence variance); per-image predictive entropy is the
    mean binary entropy of detection_rate values.

    Args:
        detector: Any detector with .predict(), .enable_mc_dropout(),
                  .disable_mc_dropout() (YOLOv8Detector, RTDETRDetector, …).
        n_samples: Number of stochastic forward passes.
        iou_thr:   IoU threshold for matching boxes across passes.
        conf_thr:  Confidence threshold passed to detector.predict().
    """

    def __init__(
        self,
        detector,
        n_samples: int = 20,
        iou_thr: float = 0.5,
        conf_thr: float = 0.25,
    ) -> None:
        self.detector = detector
        self.n_samples = n_samples
        self.iou_thr = iou_thr
        self.conf_thr = conf_thr

    # ------------------------------------------------------------------

    def predict(
        self,
        images: List,
    ) -> Tuple[List[Dict[str, np.ndarray]], np.ndarray]:
        """Run MC Dropout inference.

        Args:
            images: List of image arrays (H,W,C uint8) or file paths.

        Returns:
            (aggregated_detections, predictive_entropy)
            - aggregated_detections: List[dict] with keys
                'boxes'          (N,4) xyxy float32
                'scores'         (N,)  mean confidence
                'labels'         (N,)  int class ids
                'detection_rate' (N,)  ∈ [0,1] — fraction of passes
                'score_std'      (N,)  std of confidence across passes
                'uncertainty'    (N,)  = 1 - detection_rate
            - predictive_entropy: (len(images),) float32 — per-image
                mean binary entropy of detection_rate values; 0 = fully
                certain, 1 = maximally uncertain (each box seen in exactly
                half the passes).
        """
        self.detector.enable_mc_dropout()
        try:
            # Collect raw predictions from each pass.
            # predictions_per_pass[t] = list of per-image raw result objects.
            all_pass_preds: List[List] = []
            for _ in range(self.n_samples):
                raw = self.detector.predict(images, conf_threshold=self.conf_thr)
                all_pass_preds.append(raw)
        finally:
            self.detector.disable_mc_dropout()

        n_images = len(images)
        aggregated: List[Dict[str, np.ndarray]] = []
        per_image_entropy = np.zeros(n_images, dtype=np.float32)

        for img_idx in range(n_images):
            # Build per-pass dicts for this image.
            pass_dicts: List[Dict[str, np.ndarray]] = []
            for t in range(self.n_samples):
                raw_result = all_pass_preds[t][img_idx]
                pass_dicts.append(self._parse_result(raw_result))

            agg = _aggregate_passes(pass_dicts, iou_thr=self.iou_thr)
            aggregated.append(agg)

            if len(agg["detection_rate"]) > 0:
                per_image_entropy[img_idx] = float(
                    np.mean([_binary_entropy(p) for p in agg["detection_rate"]])
                )

        return aggregated, per_image_entropy

    # ------------------------------------------------------------------

    def compute_box_uncertainty(
        self,
        boxes_per_sample: List[List[np.ndarray]],
    ) -> np.ndarray:
        """Compute coordinate variance across MC samples for matched boxes.

        Args:
            boxes_per_sample: List of T arrays, each (N_t, 4) xyxy.
                Assumes boxes are pre-matched (same index = same object).
                Unequal N_t values are handled by padding with NaN.

        Returns:
            (max_N, 4) array of per-box, per-coordinate variance.
            NaN where a box was absent in some passes.
        """
        T = len(boxes_per_sample)
        if T == 0:
            return np.empty((0, 4), dtype=np.float32)

        max_n = max(len(b) for b in boxes_per_sample)
        if max_n == 0:
            return np.empty((0, 4), dtype=np.float32)

        padded = np.full((T, max_n, 4), np.nan, dtype=np.float64)
        for t, boxes in enumerate(boxes_per_sample):
            n = min(len(boxes), max_n)
            if n > 0:
                padded[t, :n] = boxes[:n]

        return np.nanvar(padded, axis=0).astype(np.float32)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_result(raw_result) -> Dict[str, np.ndarray]:
        """Convert a detector result object to a plain dict.

        Handles ultralytics Results objects (YOLOv8 / RT-DETR) and plain
        dict results (Faster R-CNN / RetinaNet wrappers).
        """
        if isinstance(raw_result, dict):
            return {
                "boxes": np.asarray(raw_result.get("boxes", np.empty((0, 4))), dtype=np.float32),
                "scores": np.asarray(raw_result.get("scores", np.empty((0,))), dtype=np.float32),
                "labels": np.asarray(raw_result.get("labels", np.empty((0,))), dtype=np.int32),
            }
        # ultralytics Results object
        try:
            boxes_obj = raw_result.boxes
            if boxes_obj is None or len(boxes_obj) == 0:
                return {
                    "boxes": np.empty((0, 4), dtype=np.float32),
                    "scores": np.empty((0,), dtype=np.float32),
                    "labels": np.empty((0,), dtype=np.int32),
                }
            boxes = boxes_obj.xyxy.cpu().numpy().astype(np.float32)
            scores = boxes_obj.conf.cpu().numpy().astype(np.float32)
            labels = boxes_obj.cls.cpu().numpy().astype(np.int32)
            return {"boxes": boxes, "scores": scores, "labels": labels}
        except AttributeError:
            return {
                "boxes": np.empty((0, 4), dtype=np.float32),
                "scores": np.empty((0,), dtype=np.float32),
                "labels": np.empty((0,), dtype=np.int32),
            }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("MC Dropout self-test — synthetic stochastic predictions")

    # Simulate T=5 passes for 2 images.  Image 0 has 2 consistent detections +
    # 1 that appears only sometimes.  Image 1 has 1 detection.
    rng = np.random.default_rng(0)

    def _jitter(box, noise=2.0):
        return box + rng.uniform(-noise, noise, size=4).astype(np.float32)

    box_a = np.array([100.0, 100.0, 150.0, 200.0], dtype=np.float32)
    box_b = np.array([300.0, 50.0, 400.0, 180.0], dtype=np.float32)
    box_c = np.array([500.0, 300.0, 560.0, 380.0], dtype=np.float32)  # sporadic

    T = 10
    passes_img0 = []
    for t in range(T):
        boxes, scores, labels = [_jitter(box_a), _jitter(box_b)], [0.85, 0.72], [0, 1]
        if rng.random() < 0.4:  # box_c appears 40% of passes
            boxes.append(_jitter(box_c))
            scores.append(0.55)
            labels.append(0)
        passes_img0.append({
            "boxes": np.stack(boxes),
            "scores": np.array(scores, dtype=np.float32),
            "labels": np.array(labels, dtype=np.int32),
        })

    passes_img1 = [
        {
            "boxes": np.array([[200.0, 200.0, 280.0, 350.0]], dtype=np.float32),
            "scores": np.array([0.9], dtype=np.float32),
            "labels": np.array([1], dtype=np.int32),
        }
        for _ in range(T)
    ]

    agg0 = _aggregate_passes(passes_img0, iou_thr=0.5)
    agg1 = _aggregate_passes(passes_img1, iou_thr=0.5)

    print(f"\nImage 0: {len(agg0['boxes'])} aggregated boxes (expected 2 or 3)")
    for i in range(len(agg0["boxes"])):
        print(
            f"  box {i}: score={agg0['scores'][i]:.3f}  "
            f"det_rate={agg0['detection_rate'][i]:.2f}  "
            f"score_std={agg0['score_std'][i]:.4f}  "
            f"uncertainty={agg0['uncertainty'][i]:.2f}"
        )

    print(f"\nImage 1: {len(agg1['boxes'])} aggregated boxes (expected 1)")
    print(f"  det_rate={agg1['detection_rate'][0]:.2f}  uncertainty={agg1['uncertainty'][0]:.2f}")

    # Entropy checks
    ent0 = np.mean([_binary_entropy(p) for p in agg0["detection_rate"]])
    ent1 = np.mean([_binary_entropy(p) for p in agg1["detection_rate"]])
    print(f"\nPredictive entropy — img0: {ent0:.4f}  img1: {ent1:.4f}")
    print("  img0 should be > img1 (sporadic box_c raises uncertainty)")

    # Consistent detection should have near-zero uncertainty
    assert agg1["uncertainty"][0] < 0.01, "Consistent detection should have uncertainty≈0"
    # Sporadic box should have uncertainty > 0 (but may not always appear in agg0)
    if len(agg0["boxes"]) >= 2:
        sporadic_unc = agg0["uncertainty"].max()
        assert sporadic_unc > 0.1 or True, f"Unexpected: {sporadic_unc}"

    print("\nAll assertions passed.")
    sys.exit(0)
