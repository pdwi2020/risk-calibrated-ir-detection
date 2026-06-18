"""Conformal risk control for detection confidence thresholds.

Implements Conformal Risk Control (Angelopoulos, Bates, Fisch, Lei, Schuster,
2022): choose the confidence threshold ``lambda`` on a held-out CALIBRATION
split so that the expected value of a bounded, monotone risk on unseen TEST
data is provably controlled at level ``alpha``.

Risk used here = the **missed-detection rate** (a safety-critical quantity for
IR pedestrian/vehicle detection):

    R(lambda) = (# ground-truth boxes NOT matched by any prediction with
                 score >= lambda) / (total # ground-truth boxes)

matched class-aware by IoU >= iou_threshold. R is non-decreasing in lambda
(a stricter threshold keeps fewer detections, so more GT is missed), so the
calibration is a 1-D threshold search.

Finite-sample guarantee (risk bounded in [0, 1], B = 1):

    lambda_hat = max { lambda :  (n * Rhat(lambda) + 1) / (n + 1) <= alpha }

i.e. the *strictest* (highest-precision) threshold whose conservative,
finite-sample-corrected calibration risk still respects the budget alpha. By
the CRC theorem, E[R_test(lambda_hat)] <= alpha.

Pure-Python (no numpy/torch) so it runs on cached predictions anywhere.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = (float(v) for v in a)
    bx1, by1, bx2, by2 = (float(v) for v in b)
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    ua = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ub = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = ua + ub - inter
    return inter / union if union > 0 else 0.0


class ConformalRiskController:
    """Choose a confidence threshold that controls expected missed-detection rate."""

    def __init__(self, alpha: float = 0.1, iou_threshold: float = 0.5):
        self.alpha = alpha
        self.iou_threshold = iou_threshold
        self.lambda_hat: Optional[float] = None

    # ------------------------------------------------------------------
    def _missed_and_total(self, predictions: List[Dict], ground_truths: List[Dict],
                          lam: float) -> Tuple[int, int]:
        """Return (missed_gt, total_gt) when keeping predictions with score >= lam."""
        missed = 0
        total = 0
        for pred, gt in zip(predictions, ground_truths):
            g_boxes = list(gt.get("boxes", []))
            g_labels = [int(x) for x in gt.get("labels", [])]
            total += len(g_boxes)
            if not g_boxes:
                continue

            p_boxes = list(pred.get("boxes", []))
            p_scores = [float(s) for s in pred.get("scores", [])]
            p_labels = [int(x) for x in pred.get("labels", [])]
            # keep confident predictions, strongest first
            kept = sorted((i for i in range(len(p_boxes)) if p_scores[i] >= lam),
                          key=lambda i: -p_scores[i])

            matched = [False] * len(g_boxes)
            for pi in kept:
                best_iou, best_g = 0.0, -1
                for gi in range(len(g_boxes)):
                    if matched[gi] or g_labels[gi] != p_labels[pi]:
                        continue
                    iou = _iou(p_boxes[pi], g_boxes[gi])
                    if iou >= self.iou_threshold and iou > best_iou:
                        best_iou, best_g = iou, gi
                if best_g >= 0:
                    matched[best_g] = True
            missed += matched.count(False)
        return missed, total

    def empirical_risk(self, predictions: List[Dict], ground_truths: List[Dict],
                       lam: float) -> float:
        missed, total = self._missed_and_total(predictions, ground_truths, lam)
        return missed / total if total else 0.0

    # ------------------------------------------------------------------
    def calibrate(self, cal_predictions: List[Dict], cal_ground_truths: List[Dict],
                  lambdas: Optional[Sequence[float]] = None) -> float:
        """Fit lambda_hat on the calibration split. Returns the threshold.

        Picks the largest lambda whose finite-sample-corrected calibration risk
        is <= alpha. Falls back to the smallest candidate if none qualifies.
        """
        if lambdas is None:
            lambdas = [round(0.01 * i, 2) for i in range(1, 100)]  # 0.01..0.99
        lambdas = sorted(lambdas)

        n = sum(len(gt.get("boxes", [])) for gt in cal_ground_truths)
        n = max(n, 1)

        qualifying = []
        for lam in lambdas:
            r_hat = self.empirical_risk(cal_predictions, cal_ground_truths, lam)
            corrected = (n * r_hat + 1.0) / (n + 1.0)
            if corrected <= self.alpha:
                qualifying.append(lam)
        self.lambda_hat = max(qualifying) if qualifying else lambdas[0]
        return self.lambda_hat

    def validate_coverage(self, test_predictions: List[Dict],
                          test_ground_truths: List[Dict],
                          lam: Optional[float] = None) -> Dict[str, float]:
        """Empirical test risk at lambda_hat; controlled if <= alpha."""
        lam = self.lambda_hat if lam is None else lam
        if lam is None:
            raise RuntimeError("calibrate() must be called before validate_coverage()")
        risk = self.empirical_risk(test_predictions, test_ground_truths, lam)
        return {
            "lambda_hat": lam,
            "alpha": self.alpha,
            "test_risk": risk,                       # missed-detection rate
            "test_coverage": 1.0 - risk,             # detection rate
            "controlled": risk <= self.alpha,        # risk budget respected
        }


# ----------------------------------------------------------------------------
# Unit test: python3 src/risk/conformal.py
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    import random

    def make_split(n_imgs, seed):
        rng = random.Random(seed)
        preds, gts = [], []
        for _ in range(n_imgs):
            # one GT person box, one perfectly-localized prediction with random score
            box = [10.0, 10.0, 30.0, 50.0]
            score = rng.random()  # uniform[0,1]
            gts.append({"boxes": [box], "labels": [0]})
            preds.append({"boxes": [box], "scores": [score], "labels": [0]})
        return preds, gts

    cp, cg = make_split(400, seed=1)
    tp, tg = make_split(400, seed=2)

    crc = ConformalRiskController(alpha=0.2, iou_threshold=0.5)
    lam = crc.calibrate(cp, cg)
    cov = crc.validate_coverage(tp, tg)

    # Risk monotone non-decreasing in lambda
    r_lo = crc.empirical_risk(cp, cg, 0.1)
    r_hi = crc.empirical_risk(cp, cg, 0.9)
    assert r_hi >= r_lo, (r_lo, r_hi)
    # With uniform scores, miss-rate(lambda) ~= lambda, so lambda_hat ~= alpha (<= alpha budget)
    assert 0.05 <= lam <= 0.25, lam
    # Test risk should be controlled near/under alpha
    assert cov["test_risk"] <= crc.alpha + 0.05, cov
    assert cov["controlled"] or cov["test_risk"] <= crc.alpha + 0.05, cov

    print("conformal self-test OK: alpha=%.2f lambda_hat=%.2f test_risk=%.3f coverage=%.3f"
          % (crc.alpha, lam, cov["test_risk"], cov["test_coverage"]))
