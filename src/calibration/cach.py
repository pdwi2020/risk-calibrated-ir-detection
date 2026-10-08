"""src/calibration/cach.py — Corruption-Adaptive Calibration Head (CACH).

Architecture:
  1. CorruptionEmbedNet  — small CNN over the input image → e ∈ R^d
     Encodes corruption type / severity from image statistics without any
     explicit corruption label.  Only 8–16 K parameters.
  2. CalibrationHead     — monotonic MLP: (conf, e) → calibrated confidence
     Monotone in detection confidence (preserves ranking).

Training:
  - Backbone (YOLOv8m) is FROZEN throughout.
  - Training data: pooled clean + 6×4 corrupted FLIR predictions (from
    results/corruption_preds/) plus the input images to feed the embed net.
  - Loss: TP-NLL (negative log-likelihood on is_TP labels) + focal calibration
    regulariser to spread confidence mass.

Novel claim:
  First learned test-time calibration adaptation for thermal-IR detection that
  recovers reliability under unseen low-SNR corruptions without corruption
  labels or detector retraining.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Corruption embedding network
# ---------------------------------------------------------------------------

class CorruptionEmbedNet(nn.Module):
    """Small CNN: grayscale image → corruption embedding e ∈ R^embed_dim.

    Input:  (B, 1, 64, 64)  — centre-cropped / resized patch of input image.
    Output: (B, embed_dim)
    """

    def __init__(self, embed_dim: int = 32):
        super().__init__()
        # ~12K parameters total
        self.conv = nn.Sequential(
            nn.Conv2d(1,  16, 5, stride=2, padding=2), nn.BatchNorm2d(16), nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(32, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, H, W) — float32, range [0, 1]."""
        return self.fc(self.conv(x).flatten(1))


# ---------------------------------------------------------------------------
# Monotone calibration head
# ---------------------------------------------------------------------------

class MonotoneCalibrationHead(nn.Module):
    """Inputs:  (conf, e) where conf ∈ (0,1), e ∈ R^embed_dim.
    Output:  calibrated confidence ∈ (0,1).

    Architecture: corruption-adaptive temperature scaling.
      logit_cal = T(e) · logit(conf) + b(e)
      cal_conf  = sigmoid(logit_cal)

    where T(e) = softplus(FC_T(e)) > 0 and b(e) = FC_b(e) ∈ R.
    This guarantees strict monotonicity in conf since:
      d(cal_conf)/d(conf) = sigmoid' · T(e) / (conf·(1−conf)) > 0 ∀ T > 0.

    Interpretation: the corruption embedding modulates the confidence
    temperature (spread) and bias (shift) of the calibration map — a
    generalisation of temperature scaling that adapts per image.
    """

    def __init__(self, embed_dim: int = 32, hidden: int = 64):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden = hidden

        # Temperature branch: T(e) > 0
        self.T_net = nn.Sequential(
            nn.Linear(embed_dim, hidden), nn.ReLU(True),
            nn.Linear(hidden, 1),
        )
        # Bias branch: b(e) ∈ R (unconstrained shift)
        self.b_net = nn.Sequential(
            nn.Linear(embed_dim, hidden), nn.ReLU(True),
            nn.Linear(hidden, 1),
        )
        # Initialise so the head is the TRUE identity at init (no-op):
        # T = softplus(bias) + 1e-4 = 1  ⇒  bias = ln(e^1 − 1) ≈ 0.54132.
        # (Previously bias=0 gave T=softplus(0)=ln2≈0.693, i.e. the head started
        #  ALREADY mis-calibrated — spreading confidences toward 0.5 — which is a
        #  large part of why CACH degraded ECE.  With T=1,b=0 it starts as a pass-
        #  through and only deviates if the data loss + identity reg justify it.)
        nn.init.constant_(self.T_net[-1].weight, 0.0)
        nn.init.constant_(self.T_net[-1].bias,   0.54132)
        nn.init.constant_(self.b_net[-1].weight,  0.0)
        nn.init.constant_(self.b_net[-1].bias,    0.0)

    def forward(self, conf: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        """conf: (N,) raw confidences; e: (N, embed_dim). Returns calibrated (N,)."""
        cal, _, _ = self.forward_with_params(conf, e)
        return cal

    def forward_with_params(
        self, conf: torch.Tensor, e: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Like forward but also returns (T, b) so the trainer can apply an
        identity-anchoring regulariser (T→1, b→0)."""
        conf = conf.clamp(1e-6, 1 - 1e-6)
        logit_conf = torch.log(conf / (1.0 - conf))          # (N,)
        T = F.softplus(self.T_net(e)).squeeze(1) + 1e-4      # (N,) > 0
        b = self.b_net(e).squeeze(1)                          # (N,)
        return torch.sigmoid(T * logit_conf + b), T, b        # (N,), (N,), (N,)


# ---------------------------------------------------------------------------
# Full CACH model
# ---------------------------------------------------------------------------

class CACH(nn.Module):
    """Corruption-Adaptive Calibration Head.

    Usage at inference:
        cach = CACH.load("results/cach/cach_best.pt")
        cach.eval()
        patch = preprocess_image(img)          # (1, 1, 64, 64) float32 [0,1]
        raw_scores = tensor(detector_scores)   # (N,) float32
        cal_scores = cach(patch, raw_scores)   # (N,) float32
    """

    def __init__(self, embed_dim: int = 32, hidden: int = 64):
        super().__init__()
        self.embed_net  = CorruptionEmbedNet(embed_dim=embed_dim)
        self.calib_head = MonotoneCalibrationHead(embed_dim=embed_dim, hidden=hidden)

    def forward(
        self,
        image_patch: torch.Tensor,  # (B, 1, 64, 64) — one patch per image
        conf: torch.Tensor,         # (N,) raw detection confidences
        img_idx: Optional[torch.Tensor] = None,  # (N,) which image each det belongs to
    ) -> torch.Tensor:
        """Return calibrated confidences (N,).

        If img_idx is None, assumes all detections belong to image 0 (single-image mode).
        """
        e = self.embed_net(image_patch)   # (B, embed_dim)
        if img_idx is None:
            img_idx = torch.zeros(conf.shape[0], dtype=torch.long, device=conf.device)
        e_per_det = e[img_idx]            # (N, embed_dim)
        return self.calib_head(conf, e_per_det)

    # ------------------------------------------------------------------
    # Convenience: single-image calibration (no batching overhead)
    # ------------------------------------------------------------------
    def calibrate(
        self,
        image: torch.Tensor,   # (1, 64, 64) or (1, 1, 64, 64) float32 [0,1]
        scores: torch.Tensor,  # (N,) float32
    ) -> torch.Tensor:
        if image.dim() == 3:
            image = image.unsqueeze(0)
        e = self.embed_net(image)          # (1, embed_dim)
        e_rep = e.expand(scores.shape[0], -1)
        return self.calib_head(scores, e_rep)

    def calibrate_with_params(
        self,
        image: torch.Tensor,
        scores: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Like calibrate() but also returns (T, b) for identity regularization."""
        if image.dim() == 3:
            image = image.unsqueeze(0)
        e = self.embed_net(image)
        e_rep = e.expand(scores.shape[0], -1)
        return self.calib_head.forward_with_params(scores, e_rep)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        torch.save({
            "state_dict": self.state_dict(),
            "embed_dim":  self.embed_net.fc.out_features,
            "hidden":     self.calib_head.hidden,
        }, path)

    @classmethod
    def load(cls, path: str, map_location="cpu") -> "CACH":
        ckpt = torch.load(path, map_location=map_location, weights_only=True)
        model = cls(embed_dim=ckpt["embed_dim"], hidden=ckpt["hidden"])
        model.load_state_dict(ckpt["state_dict"])
        return model

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def tp_nll_loss(cal_conf: torch.Tensor, is_tp: torch.Tensor) -> torch.Tensor:
    """Binary cross-entropy: calibrated confidence should match is_TP labels."""
    return F.binary_cross_entropy(cal_conf.clamp(1e-6, 1 - 1e-6), is_tp.float())


def focal_calibration_loss(
    cal_conf: torch.Tensor, is_tp: torch.Tensor, gamma: float = 2.0
) -> torch.Tensor:
    """Focal variant — down-weights easy, well-calibrated predictions."""
    bce = F.binary_cross_entropy(cal_conf.clamp(1e-6, 1 - 1e-6),
                                 is_tp.float(), reduction="none")
    p_t = torch.where(is_tp.bool(), cal_conf, 1.0 - cal_conf)
    weight = (1.0 - p_t) ** gamma
    return (weight * bce).mean()


def combined_loss(
    cal_conf: torch.Tensor,
    is_tp: torch.Tensor,
    lambda_focal: float = 0.5,
    gamma: float = 2.0,
) -> torch.Tensor:
    return ((1 - lambda_focal) * tp_nll_loss(cal_conf, is_tp)
            + lambda_focal * focal_calibration_loss(cal_conf, is_tp, gamma))


def identity_reg_loss(T: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Anchor the corruption-adaptive map to the identity calibration (T=1, b=0).

    Penalising deviation from identity makes the head default to a pass-through
    (no worse than no-calibration) and only move where the data loss provides
    enough evidence — preventing the head from de-calibrating well-calibrated
    detectors (the failure mode observed without this term)."""
    return ((T - 1.0) ** 2 + b ** 2).mean()


# ---------------------------------------------------------------------------
# Image preprocessing helper
# ---------------------------------------------------------------------------

def preprocess_patch(img: "np.ndarray", size: int = 64) -> "torch.Tensor":
    """Convert a grayscale numpy image to a (1,1,size,size) float32 tensor."""
    import numpy as np
    try:
        import cv2
        patch = cv2.resize(img.astype(np.float32), (size, size),
                           interpolation=cv2.INTER_LINEAR)
    except ImportError:
        from PIL import Image
        patch = np.array(Image.fromarray(img).resize((size, size), Image.BILINEAR),
                         dtype=np.float32)
    patch = patch / 255.0
    return torch.tensor(patch, dtype=torch.float32).unsqueeze(0).unsqueeze(0)


# ---------------------------------------------------------------------------
# IoU matching helper (re-used in training data builder)
# ---------------------------------------------------------------------------

def match_detections_to_gt(
    pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels,
    iou_threshold: float = 0.5,
) -> Tuple["np.ndarray", "np.ndarray"]:
    """Return (matched_scores, is_tp) arrays for all kept predictions.

    Greedy matching by descending score, class-aware, IoU ≥ iou_threshold.
    """
    import numpy as np

    if len(pred_boxes) == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    scores  = np.array(pred_scores,  dtype=np.float32)
    p_boxes = np.array(pred_boxes,   dtype=np.float32)
    p_labs  = np.array(pred_labels,  dtype=np.int32)
    g_boxes = np.array(gt_boxes,     dtype=np.float32) if gt_boxes else np.zeros((0, 4))
    g_labs  = np.array(gt_labels,    dtype=np.int32)   if gt_labels else np.zeros(0, dtype=np.int32)

    order = np.argsort(-scores)
    is_tp = np.zeros(len(pred_boxes), dtype=np.float32)
    matched = np.zeros(len(g_boxes), dtype=bool)

    def _iou(a, b):
        ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
        ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
        iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        ua = max(0.0, a[2]-a[0]) * max(0.0, a[3]-a[1])
        ub = max(0.0, b[2]-b[0]) * max(0.0, b[3]-b[1])
        denom = ua + ub - inter
        return inter / denom if denom > 0 else 0.0

    for pi in order:
        best_iou, best_g = 0.0, -1
        for gi in range(len(g_boxes)):
            if matched[gi] or g_labs[gi] != p_labs[pi]:
                continue
            iou = _iou(p_boxes[pi], g_boxes[gi])
            if iou >= iou_threshold and iou > best_iou:
                best_iou, best_g = iou, gi
        if best_g >= 0:
            matched[best_g] = True
            is_tp[pi] = 1.0

    return scores, is_tp


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import torch
    model = CACH(embed_dim=32, hidden=64)
    print(f"CACH parameters: {model.param_count():,}")

    # Forward pass
    patch = torch.rand(2, 1, 64, 64)
    conf  = torch.tensor([0.9, 0.3, 0.7, 0.1, 0.8, 0.5], dtype=torch.float32)
    idx   = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
    out   = model(patch, conf, idx)
    print(f"Output shape: {out.shape}  range: [{out.min():.3f}, {out.max():.3f}]")
    assert out.shape == (6,), f"Expected (6,) got {out.shape}"

    # Monotonicity check: increasing raw conf → non-decreasing calibrated conf
    patch_single = torch.rand(1, 1, 64, 64)
    confs_inc = torch.linspace(0.05, 0.95, 19)
    cal_inc = model.calibrate(patch_single, confs_inc)
    diffs = cal_inc[1:] - cal_inc[:-1]
    assert (diffs >= -1e-4).all(), f"Monotonicity violated: {diffs.min():.6f}"
    print(f"Monotonicity OK (min diff: {diffs.min():.6f})")

    # Loss
    is_tp = (conf > 0.5).float()
    loss = combined_loss(out, is_tp)
    print(f"Combined loss: {loss.item():.4f}")
    print("CACH self-test PASSED")
